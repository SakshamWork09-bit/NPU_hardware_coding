import os
import time
import queue
import threading
import cv2
import numpy as np
import openvino as ov

MODEL_XML = r"C:\Users\saksh\HardwareLab\yolov8n_openvino_model\yolov8n.xml"
MODEL_ONNX = r"C:\Users\saksh\HardwareLab\yolov8n.onnx"

# Auto-detect INT8 IR model if present, otherwise fallback to ONNX
if os.path.exists(MODEL_XML):
    MODEL_PATH = MODEL_XML
    MODEL_TYPE = "INT8 OpenVINO IR"
elif os.path.exists(MODEL_ONNX):
    MODEL_PATH = MODEL_ONNX
    MODEL_TYPE = "FP32/FP16 ONNX"
else:
    raise FileNotFoundError(
        "No YOLOv8 model found. Ensure yolov8n.onnx or yolov8n_openvino_model exists."
    )

INPUT_W, INPUT_H = 640, 640
CONF_THRESHOLD = 0.40
IOU_THRESHOLD = 0.45

# Worker allocation: 1 GPU worker + 2 NPU workers for maximum concurrent throughput
WORKER_CONFIG = {
    "GPU": 1,
    "NPU": 2
}

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

core = ov.Core()

print("=" * 85)
print("HETEROGENEOUS PARALLEL OBJECT DETECTION (GPU + NPU CO-EXECUTION)")
print("=" * 85)
print(f"Model File       : {MODEL_PATH} ({MODEL_TYPE})")
print("Available Devices:", core.available_devices)
print("Worker Pool      :", WORKER_CONFIG)
print("Press 'q' in the video window to quit.")
print("=" * 85)

compiled_models = {}
for dev in WORKER_CONFIG:
    if dev in core.available_devices:
        print(f"Compiling YOLOv8 for {dev}...")
        model = core.read_model(MODEL_PATH)
        compiled_models[dev] = core.compile_model(model, dev)
    else:
        raise RuntimeError(f"Required device {dev} not available.")

# Shared bounded task queue
task_queue = queue.Queue(maxsize=16)
stop_event = threading.Event()

# Telemetry and state trackers
device_counts = {"GPU": 0, "NPU": 0}
counter_lock = threading.Lock()
latest_boxes = []
box_lock = threading.Lock()

# ------------------------------------------------------------
# Vectorized NMS and Box Decoder (High-Speed NumPy)
# ------------------------------------------------------------
def decode_and_nms(raw_output, scale, dx, dy):
    # raw_output is [84, 8400] -> transpose to [8400, 84]
    out = raw_output.T
    classes_scores = out[:, 4:]
    max_scores = np.max(classes_scores, axis=1)
    mask = max_scores >= CONF_THRESHOLD

    if not np.any(mask):
        return []

    filtered_out = out[mask]
    filtered_scores = max_scores[mask]
    class_ids = np.argmax(filtered_out[:, 4:], axis=1)

    boxes_raw = filtered_out[:, :4]
    cx = boxes_raw[:, 0]
    cy = boxes_raw[:, 1]
    bw = boxes_raw[:, 2]
    bh = boxes_raw[:, 3]

    x1 = np.round(((cx - bw / 2.0) - dx) / scale).astype(int)
    y1 = np.round(((cy - bh / 2.0) - dy) / scale).astype(int)
    w_box = np.round(bw / scale).astype(int)
    h_box = np.round(bh / scale).astype(int)

    boxes = np.stack([x1, y1, w_box, h_box], axis=1).tolist()
    confidences = filtered_scores.tolist()
    class_ids = class_ids.tolist()

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, IOU_THRESHOLD)

    detected = []
    if len(indices) > 0:
        for idx in indices.flatten():
            detected.append((boxes[idx], confidences[idx], class_ids[idx]))

    return detected

# ------------------------------------------------------------
# Engine Worker Thread
# ------------------------------------------------------------
def engine_worker(device_name, compiled_model):
    infer_request = compiled_model.create_infer_request()

    while not stop_event.is_set():
        try:
            item = task_queue.get(timeout=0.01)
        except queue.Empty:
            continue

        tensor, scale, dx, dy = item

        # Run inference on target accelerator
        infer_request.infer({0: tensor})
        raw_output = infer_request.get_output_tensor(0).data[0]

        # Decode detections using fast vectorized operations
        detected = decode_and_nms(raw_output, scale, dx, dy)

        with counter_lock:
            device_counts[device_name] += 1

        with box_lock:
            global latest_boxes
            latest_boxes = detected

        task_queue.task_done()

# Start worker pool
for dev, count in WORKER_CONFIG.items():
    for i in range(count):
        t = threading.Thread(
            target=engine_worker,
            args=(dev, compiled_models[dev]),
            daemon=True,
            name=f"{dev}_Worker_{i}"
        )
        t.start()

# ------------------------------------------------------------
# Threaded Camera Ingestion
# ------------------------------------------------------------
class CameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.use_synthetic = not self.cap.isOpened()
        self.running = True
        self.frame = None
        self.lock = threading.Lock()

        if self.use_synthetic:
            print("\n[NOTE] No webcam found. Running synthetic video generator.")
            self.frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        else:
            _, self.frame = self.cap.read()

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        while self.running:
            if not self.use_synthetic:
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.frame = frame
                else:
                    time.sleep(0.01)
            else:
                t = time.perf_counter()
                f = np.zeros((1080, 1920, 3), dtype=np.uint8)
                cv2.circle(f, (int((t * 400) % 1920), 540), 110, (0, 255, 0), -1)
                with self.lock:
                    self.frame = f
                time.sleep(0.005)

    def read_latest(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if not self.use_synthetic:
            self.cap.release()

cam = CameraStream(0)

# Letterbox resize to 640x640
def preprocess(frame):
    h, w = frame.shape[:2]
    scale = min(INPUT_W / w, INPUT_H / h)
    nw, nh = int(w * scale), int(h * scale)

    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((INPUT_H, INPUT_W, 3), dtype=np.uint8)
    dx = (INPUT_W - nw) // 2
    dy = (INPUT_H - nh) // 2
    canvas[dy:dy+nh, dx:dx+nw] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return np.expand_dims(tensor, axis=0), scale, dx, dy

# ------------------------------------------------------------
# Main Dispatcher & Display Loop
# ------------------------------------------------------------
fps_timer = time.perf_counter()
display_timer = time.perf_counter()
prev_gpu = 0
prev_npu = 0

total_fps = 0.0
gpu_fps = 0.0
npu_fps = 0.0

print("\nStarting Heterogeneous Parallel YOLOv8 Detection...\n")

try:
    while True:
        frame = cam.read_latest()
        if frame is None:
            time.sleep(0.001)
            continue

        tensor, scale, dx, dy = preprocess(frame)
        try:
            task_queue.put((tensor, scale, dx, dy), block=False)
        except queue.Full:
            pass

        now = time.perf_counter()

        # Update telemetry every 1.0 second
        if now - fps_timer >= 1.0:
            elapsed = now - fps_timer
            with counter_lock:
                c_gpu = device_counts["GPU"]
                c_npu = device_counts["NPU"]

            gpu_fps = (c_gpu - prev_gpu) / elapsed
            npu_fps = (c_npu - prev_npu) / elapsed
            total_fps = gpu_fps + npu_fps

            prev_gpu = c_gpu
            prev_npu = c_npu
            fps_timer = now

            with box_lock:
                n_objs = len(latest_boxes)

            print(f"[HETERO YOLO] Total: {total_fps:6.1f} FPS | "
                  f"GPU: {gpu_fps:5.1f} FPS | NPU: {npu_fps:5.1f} FPS | "
                  f"Objects: {n_objs}")

        # Render display at 30 Hz
        if now - display_timer >= 0.033:
            display_frame = frame.copy()

            with box_lock:
                boxes_to_draw = list(latest_boxes)

            # Draw bounding boxes and labels
            for (box, score, cls_id) in boxes_to_draw:
                bx, by, bw, bh = box
                cls_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"ID {cls_id}"
                label = f"{cls_name}: {score:.2f}"

                cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                cv2.putText(display_frame, label, (bx, max(20, by - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # HUD Status Overlay
            cv2.rectangle(display_frame, (10, 10), (440, 115), (20, 20, 20), -1)
            cv2.putText(display_frame, f"ENGINE: GPU + NPU ({MODEL_TYPE})", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)
            cv2.putText(display_frame, f"TOTAL SPEED : {total_fps:5.1f} FPS", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 0), 2)
            cv2.putText(display_frame, f"GPU: {gpu_fps:4.1f} FPS | NPU: {npu_fps:4.1f} FPS", (20, 92),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

            cv2.imshow("OpenVINO Heterogeneous YOLOv8 Detection", display_frame)
            display_timer = now

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    stop_event.set()
    cam.stop()
    cv2.destroyAllWindows()
    print("Heterogeneous YOLO pipeline stopped cleanly.")