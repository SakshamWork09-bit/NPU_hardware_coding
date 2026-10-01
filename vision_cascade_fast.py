import os
import time
import queue
import threading
import cv2
import numpy as np
import openvino as ov

DET_MODEL_PATH = r"C:\Users\saksh\HardwareLab\yolov8n.onnx"
POSE_MODEL_PATH = r"C:\Users\saksh\HardwareLab\yolov8n-pose.onnx"

INPUT_DIM = 640
DET_CONF_THRESH = 0.40
DET_IOU_THRESH = 0.45

# COCO 17 Skeleton Bones Connection Map
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

core = ov.Core()

print("=" * 85)
print("FULLY PIPELINED ASYMMETRIC CASCADE (NPU DETECTOR || GPU POSE ESTIMATOR)")
print("=" * 85)
print(f"Stage 1 [Detector] : {DET_MODEL_PATH} -> Target: NPU (2 Async Jobs)")
print(f"Stage 2 [Pose]     : {POSE_MODEL_PATH} -> Target: GPU (2 Async Jobs)")
print("Controls           : Press 'q' in video display to quit")
print("=" * 85)

# 1. Compile Stage 1 on NPU with 2 Async Jobs
print("Compiling Stage 1 (NPU Detection)...")
det_model = core.read_model(DET_MODEL_PATH)
compiled_det = core.compile_model(det_model, "NPU")
npu_queue = ov.AsyncInferQueue(compiled_det, jobs=2)

# 2. Compile Stage 2 on Arc 130T GPU with 2 Async Jobs
print("Compiling Stage 2 (GPU Pose)...")
pose_model = core.read_model(POSE_MODEL_PATH)
compiled_pose = core.compile_model(pose_model, "GPU")
gpu_pose_queue = ov.AsyncInferQueue(compiled_pose, jobs=2)

# Threading queues & events
stop_event = threading.Event()
crop_queue = queue.Queue(maxsize=16)

# Telemetry counters
npu_count = 0
gpu_count = 0
count_lock = threading.Lock()

latest_boxes = []
latest_kpts = []
state_lock = threading.Lock()

# ------------------------------------------------------------
# Vectorized NMS for Stage 1 Detection
# ------------------------------------------------------------
def decode_person_detections(raw_out, scale, dx, dy):
    out = raw_out.T
    classes_scores = out[:, 4:]
    max_scores = np.max(classes_scores, axis=1)
    class_ids = np.argmax(classes_scores, axis=1)

    # Class ID 0 = 'person'
    mask = (max_scores >= DET_CONF_THRESH) & (class_ids == 0)
    if not np.any(mask):
        return []

    filtered_out = out[mask]
    filtered_scores = max_scores[mask]

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

    indices = cv2.dnn.NMSBoxes(boxes, confidences, DET_CONF_THRESH, DET_IOU_THRESH)
    detected_people = []
    if len(indices) > 0:
        for idx in indices.flatten():
            detected_people.append((boxes[idx], confidences[idx]))
    return detected_people

# ------------------------------------------------------------
# Stage 1: Async NPU Callback
# ------------------------------------------------------------
def npu_callback(infer_request, user_data):
    global npu_count
    frame_ref, scale, dx, dy = user_data
    raw_out = infer_request.get_output_tensor(0).data[0]
    detected_people = decode_person_detections(raw_out, scale, dx, dy)

    with state_lock:
        global latest_boxes
        latest_boxes = detected_people

    with count_lock:
        npu_count += 1

    # Hand off detected subject to Stage 2 GPU queue
    if detected_people:
        orig_h, orig_w = frame_ref.shape[:2]
        bx, by, bw, bh = detected_people[0][0]
        pad = 15
        x1 = max(0, bx - pad)
        y1 = max(0, by - pad)
        x2 = min(orig_w, bx + bw + pad)
        y2 = min(orig_h, by + bh + pad)

        if x2 > x1 and y2 > y1:
            crop = frame_ref[y1:y2, x1:x2]
            crop_tensor, c_scale, c_dx, c_dy = preprocess(crop)
            meta = (c_scale, c_dx - x1, c_dy - y1)
            try:
                crop_queue.put((crop_tensor, meta), block=False)
            except queue.Full:
                pass

npu_queue.set_callback(npu_callback)

# ------------------------------------------------------------
# Stage 2: Async GPU Pose Callback
# ------------------------------------------------------------
def gpu_pose_callback(infer_request, user_data):
    global gpu_count
    scale, offset_x, offset_y = user_data
    out = infer_request.get_output_tensor(0).data[0].T  # [8400, 56]
    scores = out[:, 4]
    best_idx = np.argmax(scores)

    kpts = []
    if scores[best_idx] >= 0.35:
        raw_k = out[best_idx, 5:]
        for k in range(17):
            kx = int((raw_k[k * 3] - offset_x) / scale)
            ky = int((raw_k[k * 3 + 1] - offset_y) / scale)
            kconf = float(raw_k[k * 3 + 2])
            kpts.append((kx, ky, kconf))

    with state_lock:
        global latest_kpts
        if kpts:
            latest_kpts = kpts

    with count_lock:
        gpu_count += 1

gpu_pose_queue.set_callback(gpu_pose_callback)

# Dedicated thread feeding Stage 2 GPU without delaying Stage 1
def gpu_worker_loop():
    while not stop_event.is_set():
        try:
            crop_tensor, meta = crop_queue.get(timeout=0.01)
        except queue.Empty:
            continue

        gpu_pose_queue.start_async({0: crop_tensor}, userdata=meta)
        crop_queue.task_done()

threading.Thread(target=gpu_worker_loop, daemon=True, name="GPU_Pose_Thread").start()

# ------------------------------------------------------------
# Camera Thread
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
            self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        else:
            _, self.frame = self.cap.read()

        threading.Thread(target=self._capture_loop, daemon=True).start()

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
                f = np.zeros((720, 1280, 3), dtype=np.uint8)
                cx = int((t * 260) % 1280)
                cy = 360
                cv2.circle(f, (cx, cy), 80, (200, 200, 200), -1)
                cv2.circle(f, (cx, cy - 100), 40, (255, 255, 255), -1)
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

# Preprocessing: Letterbox resize to 640x640
def preprocess(frame):
    h, w = frame.shape[:2]
    scale = min(INPUT_DIM / w, INPUT_DIM / h)
    nw, nh = int(w * scale), int(h * scale)

    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((INPUT_DIM, INPUT_DIM, 3), dtype=np.uint8)
    dx = (INPUT_DIM - nw) // 2
    dy = (INPUT_DIM - nh) // 2
    canvas[dy:dy+nh, dx:dx+nw] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return np.expand_dims(tensor, axis=0), scale, dx, dy

# ------------------------------------------------------------
# Main Pipelined Dispatcher & Display Loop
# ------------------------------------------------------------
fps_timer = time.perf_counter()
display_timer = time.perf_counter()
prev_npu, prev_gpu = 0, 0
npu_fps, gpu_fps = 0.0, 0.0

print("\nRunning High-FPS Fully Pipelined Cascade...\n")

try:
    while True:
        frame = cam.read_latest()
        if frame is None:
            time.sleep(0.001)
            continue

        tensor, scale, dx, dy = preprocess(frame)

        # Dispatch async to NPU (Non-blocking: main thread never stalls!)
        user_meta = (frame, scale, dx, dy)
        npu_queue.start_async({0: tensor}, userdata=user_meta)

        now = time.perf_counter()

        # Telemetry updates every 1.0s
        if now - fps_timer >= 1.0:
            elapsed = now - fps_timer
            with count_lock:
                c_npu, c_gpu = npu_count, gpu_count

            npu_fps = (c_npu - prev_npu) / elapsed
            gpu_fps = (c_gpu - prev_gpu) / elapsed
            prev_npu, prev_gpu = c_npu, c_gpu
            fps_timer = now

            print(f"[PIPELINED] NPU Detection: {npu_fps:6.1f} FPS | "
                  f"GPU Pose Estimation: {gpu_fps:6.1f} FPS")

        # Capped display rendering (30 Hz)
        if now - display_timer >= 0.033:
            display_frame = frame.copy()

            with state_lock:
                boxes_to_draw = list(latest_boxes)
                kpts_to_draw = list(latest_kpts)

            # Draw NPU Person Bounding Boxes
            for (box, score) in boxes_to_draw:
                bx, by, bw, bh = box
                cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
                cv2.putText(display_frame, f"NPU: Person ({score:.2f})", (bx, max(20, by - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            # Draw GPU Skeleton Bones
            if kpts_to_draw:
                for pt1, pt2 in SKELETON_CONNECTIONS:
                    if pt1 < len(kpts_to_draw) and pt2 < len(kpts_to_draw):
                        x1, y1, c1 = kpts_to_draw[pt1]
                        x2, y2, c2 = kpts_to_draw[pt2]
                        if c1 > 0.4 and c2 > 0.4:
                            cv2.line(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                for (kx, ky, conf) in kpts_to_draw:
                    if conf > 0.4:
                        cv2.circle(display_frame, (kx, ky), 4, (0, 0, 255), -1)

            # HUD Telemetry
            cv2.rectangle(display_frame, (10, 10), (460, 100), (20, 20, 20), -1)
            cv2.putText(display_frame, "PIPELINED ASYMMETRIC CASCADE", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(display_frame, f"NPU DETECTOR : {npu_fps:5.1f} FPS", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(display_frame, f"GPU POSE EST : {gpu_fps:5.1f} FPS", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

            cv2.imshow("Pipelined Asymmetric Cascade (NPU || GPU)", display_frame)
            display_timer = now

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    stop_event.set()
    cam.stop()
    cv2.destroyAllWindows()
    print("Pipeline stopped cleanly.")