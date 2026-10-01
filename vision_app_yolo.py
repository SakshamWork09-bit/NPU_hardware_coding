import os
import time
import threading
import cv2
import numpy as np
import openvino as ov

MODEL_PATH = r"C:\Users\saksh\HardwareLab\yolov8n.onnx"
INPUT_W, INPUT_H = 640, 640
CONF_THRESHOLD = 0.40
IOU_THRESHOLD = 0.45

# COCO 80 Class Names
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

print("=" * 80)
print("REAL-TIME OBJECT DETECTION ENGINE (YOLOv8 + OPENVINO)")
print("=" * 80)
print("Available Devices:", core.available_devices)
print("\nControls:")
print("  Press 'g' -> Run on GPU (Intel Arc 130T)")
print("  Press 'n' -> Run on NPU (Intel AI Boost)")
print("  Press 'c' -> Run on CPU (Core Ultra 5 225H)")
print("  Press 'q' -> Quit Application")
print("=" * 80)

# Compile YOLOv8 for available devices
compiled_engines = {}
for dev in core.available_devices:
    try:
        print(f"Compiling YOLOv8 for {dev}...")
        model = core.read_model(MODEL_PATH)
        compiled = core.compile_model(model, dev)
        queue = ov.AsyncInferQueue(compiled, jobs=2)
        compiled_engines[dev] = (compiled, queue)
    except Exception as e:
        print(f"Could not compile for {dev}: {e}")

current_device = "GPU" if "GPU" in compiled_engines else "CPU"
_, current_queue = compiled_engines[current_device]

# ------------------------------------------------------------
# Threaded Camera Capture
# ------------------------------------------------------------
class CameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.use_synthetic = not self.cap.isOpened()
        self.running = True
        self.frame = None
        self.lock = threading.Lock()

        if self.use_synthetic:
            print("\n[NOTE] No webcam found. Generating synthetic video.")
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
                cv2.circle(f, (int((t * 400) % 1920), 540), 100, (0, 255, 0), -1)
                with self.lock:
                    self.frame = f
                time.sleep(0.01)

    def read_latest(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if not self.use_synthetic:
            self.cap.release()

cam = CameraStream(0)

# Preprocessing: Letterbox / Resize 640x640 + Normalization
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

# Postprocessing: Box Decoding + OpenCV NMS
latest_boxes = []
box_lock = threading.Lock()
inference_count = 0

def postprocess_callback(infer_request, user_data):
    global inference_count, latest_boxes
    scale, dx, dy, orig_w, orig_h = user_data
    
    # YOLOv8 output shape: [1, 84, 8400]
    out = infer_request.get_output_tensor(0).data[0]
    
    # Transpose to [8400, 84] (rows: candidate anchors; cols: cx, cy, w, h, 80 class scores)
    out = out.T

    boxes = []
    confidences = []
    class_ids = []

    for row in out:
        classes_scores = row[4:]
        max_idx = np.argmax(classes_scores)
        max_score = classes_scores[max_idx]

        if max_score >= CONF_THRESHOLD:
            cx, cy, bw, bh = row[0], row[1], row[2], row[3]
            
            # Map back from letterbox canvas to original frame coordinates
            x1 = int(((cx - bw / 2) - dx) / scale)
            y1 = int(((cy - bh / 2) - dy) / scale)
            w_box = int(bw / scale)
            h_box = int(bh / scale)

            boxes.append([x1, y1, w_box, h_box])
            confidences.append(float(max_score))
            class_ids.append(int(max_idx))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, IOU_THRESHOLD)

    detected = []
    if len(indices) > 0:
        for idx in indices.flatten():
            detected.append((boxes[idx], confidences[idx], class_ids[idx]))

    with box_lock:
        latest_boxes = detected
    inference_count += 1

# Assign callback to queues
for dev in compiled_engines:
    compiled_engines[dev][1].set_callback(postprocess_callback)

# ------------------------------------------------------------
# Main Dispatch & Render Loop
# ------------------------------------------------------------
fps_timer = time.perf_counter()
display_timer = time.perf_counter()
display_fps = 0.0

print("\nStarting YOLO real-time detection stream...\n")

try:
    while True:
        frame = cam.read_latest()
        if frame is None:
            time.sleep(0.001)
            continue

        orig_h, orig_w = frame.shape[:2]
        tensor, scale, dx, dy = preprocess(frame)
        
        # Dispatch async inference with mapping parameters
        user_meta = (scale, dx, dy, orig_w, orig_h)
        current_queue.start_async({0: tensor}, userdata=user_meta)

        now = time.perf_counter()

        # Update FPS counter
        if now - fps_timer >= 1.0:
            elapsed = now - fps_timer
            display_fps = inference_count / elapsed
            print(f"[{current_device}] Detection Throughput: {display_fps:6.1f} FPS | Objects Tracked: {len(latest_boxes)}")
            inference_count = 0
            fps_timer = now

        # Render display at 30 Hz
        if now - display_timer >= 0.033:
            display_frame = frame.copy()

            with box_lock:
                boxes_to_draw = list(latest_boxes)

            # Draw bounding boxes
            for (box, score, cls_id) in boxes_to_draw:
                bx, by, bw, bh = box
                label = f"{COCO_CLASSES[cls_id]}: {score:.2f}"
                
                # Bounding box & label banner
                cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                cv2.putText(display_frame, label, (bx, max(20, by - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # HUD stats box
            cv2.rectangle(display_frame, (10, 10), (380, 100), (20, 20, 20), -1)
            cv2.putText(display_frame, f"ENGINE: {current_device} (YOLOv8n)", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_frame, f"SPEED : {display_fps:5.1f} FPS", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"DETECTED: {len(boxes_to_draw)} objects", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("OpenVINO Real-Time Object Detection", display_frame)
            display_timer = now

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('g') and "GPU" in compiled_engines:
                current_queue.wait_all()
                current_device = "GPU"
                current_queue = compiled_engines["GPU"][1]
                print("\n>>> Switched YOLO to GPU <<<\n")
            elif key == ord('n') and "NPU" in compiled_engines:
                current_queue.wait_all()
                current_device = "NPU"
                current_queue = compiled_engines["NPU"][1]
                print("\n>>> Switched YOLO to NPU <<<\n")
            elif key == ord('c') and "CPU" in compiled_engines:
                current_queue.wait_all()
                current_device = "CPU"
                current_queue = compiled_engines["CPU"][1]
                print("\n>>> Switched YOLO to CPU <<<\n")

finally:
    cam.stop()
    cv2.destroyAllWindows()
    print("YOLO application stopped cleanly.")