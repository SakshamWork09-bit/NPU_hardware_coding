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

# COCO 17 Skeleton Bones Connection Map for rendering
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # Facial landmarks
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # Upper body / arms
    (5, 11), (6, 12), (11, 12),              # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)   # Lower body / legs
]

core = ov.Core()

print("=" * 85)
print("ASYMMETRIC CASCADED PIPELINE: NPU (DETECTOR) + GPU (POSE ESTIMATOR)")
print("=" * 85)
print(f"Stage 1 [Detector] : {DET_MODEL_PATH} -> Target: NPU")
print(f"Stage 2 [Pose]     : {POSE_MODEL_PATH} -> Target: GPU")
print("Controls           : Press 'q' in video display to quit")
print("=" * 85)

# Compile Stage 1 on NPU
print("Compiling Stage 1 (YOLOv8 Detection) on NPU...")
det_model = core.read_model(DET_MODEL_PATH)
compiled_det = core.compile_model(det_model, "NPU")

# Compile Stage 2 on Arc 130T GPU
print("Compiling Stage 2 (YOLOv8 Pose) on GPU...")
pose_model = core.read_model(POSE_MODEL_PATH)
compiled_pose = core.compile_model(pose_model, "GPU")
pose_queue = ov.AsyncInferQueue(compiled_pose, jobs=2)

# Inter-stage decoupling queues
crop_task_queue = queue.Queue(maxsize=16)
stop_event = threading.Event()

# Telemetry counters
npu_det_count = 0
gpu_pose_count = 0
counter_lock = threading.Lock()

latest_people_boxes = []
latest_keypoints_list = []
results_lock = threading.Lock()

# ------------------------------------------------------------
# Vectorized Box Decoder for Stage 1 Detection
# ------------------------------------------------------------
def decode_person_detections(raw_out, scale, dx, dy):
    out = raw_out.T
    classes_scores = out[:, 4:]
    max_scores = np.max(classes_scores, axis=1)
    class_ids = np.argmax(classes_scores, axis=1)

    # Class ID 0 = 'person' in COCO dataset
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
# Stage 2: GPU Pose Estimation Worker
# ------------------------------------------------------------
def pose_callback(infer_request, user_data):
    global gpu_pose_count
    orig_w, orig_h, scale, dx, dy = user_data
    # Output shape of yolov8-pose: [1, 56, 8400]
    # 56 = 4 box coordinates + 1 box score + 17 keypoints * 3 (x, y, conf)
    out = infer_request.get_output_tensor(0).data[0].T  # [8400, 56]
    
    scores = out[:, 4]
    best_idx = np.argmax(scores)
    
    person_kpts = []
    if scores[best_idx] >= 0.35:
        kpts_raw = out[best_idx, 5:]  # length 51
        for k in range(17):
            kx = ((kpts_raw[k * 3] - dx) / scale)
            ky = ((kpts_raw[k * 3 + 1] - dy) / scale)
            kconf = kpts_raw[k * 3 + 2]
            person_kpts.append((int(kx), int(ky), float(kconf)))

    with results_lock:
        if person_kpts:
            latest_keypoints_list.append(person_kpts)
            if len(latest_keypoints_list) > 4:
                latest_keypoints_list.pop(0)

    with counter_lock:
        gpu_pose_count += 1

pose_queue.set_callback(pose_callback)

def gpu_pose_worker():
    while not stop_event.is_set():
        try:
            crop_item = crop_task_queue.get(timeout=0.01)
        except queue.Empty:
            continue

        tensor, user_meta = crop_item
        pose_queue.start_async({0: tensor}, userdata=user_meta)
        crop_task_queue.task_done()

pose_thread = threading.Thread(target=gpu_pose_worker, daemon=True, name="GPU_Pose_Worker")
pose_thread.start()

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
                f = np.zeros((720, 1280, 3), dtype=np.uint8)
                # Synthetic simulated moving subject
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
# Stage 1: NPU Detection Worker
# ------------------------------------------------------------
npu_infer_request = compiled_det.create_infer_request()

fps_timer = time.perf_counter()
display_timer = time.perf_counter()
prev_det_count = 0
prev_pose_count = 0
npu_fps = 0.0
gpu_fps = 0.0

print("\nStarting Asymmetric Cascade Pipeline...\n")

try:
    while True:
        frame = cam.read_latest()
        if frame is None:
            time.sleep(0.001)
            continue

        orig_h, orig_w = frame.shape[:2]
        tensor, scale, dx, dy = preprocess(frame)

        # 1. Run Stage 1 (NPU Detects People in Wide Angle)
        npu_infer_request.infer({0: tensor})
        raw_det_out = npu_infer_request.get_output_tensor(0).data[0]
        detected_people = decode_person_detections(raw_det_out, scale, dx, dy)

        with counter_lock:
            npu_det_count += 1

        with results_lock:
            latest_people_boxes = detected_people

        # 2. Dispatch to Stage 2 (GPU Pose Estimator)
        if detected_people:
            # Crop the primary detected subject
            bx, by, bw, bh = detected_people[0][0]
            pad = 20
            x1 = max(0, bx - pad)
            y1 = max(0, by - pad)
            x2 = min(orig_w, bx + bw + pad)
            y2 = min(orig_h, by + bh + pad)

            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                crop_tensor, c_scale, c_dx, c_dy = preprocess(crop)
                user_meta = (crop.shape[1], crop.shape[0], c_scale, c_dx - x1, c_dy - y1)
                
                try:
                    crop_task_queue.put((crop_tensor, user_meta), block=False)
                except queue.Full:
                    pass

        now = time.perf_counter()

        # Update FPS telemetry every 1.0s
        if now - fps_timer >= 1.0:
            elapsed = now - fps_timer
            with counter_lock:
                c_det = npu_det_count
                c_pose = gpu_pose_count

            npu_fps = (c_det - prev_det_count) / elapsed
            gpu_fps = (c_pose - prev_pose_count) / elapsed
            prev_det_count = c_det
            prev_pose_count = c_pose
            fps_timer = now

            print(f"[CASCADE] NPU Detection: {npu_fps:5.1f} FPS | "
                  f"GPU Pose Estimation: {gpu_fps:5.1f} FPS | "
                  f"Subjects: {len(detected_people)}")

        # Display Loop (30 Hz)
        if now - display_timer >= 0.033:
            display_frame = frame.copy()

            with results_lock:
                boxes_to_draw = list(latest_people_boxes)
                kpts_to_draw = list(latest_keypoints_list)

            # Draw Stage 1 (NPU) Bounding Boxes
            for (box, score) in boxes_to_draw:
                bx, by, bw, bh = box
                cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
                cv2.putText(display_frame, f"NPU: Person ({score:.2f})", (bx, max(20, by - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            # Draw Stage 2 (GPU) Skeleton Keypoints
            for kpts in kpts_to_draw:
                # Draw bones
                for pt1_idx, pt2_idx in SKELETON_CONNECTIONS:
                    if pt1_idx < len(kpts) and pt2_idx < len(kpts):
                        x1, y1, c1 = kpts[pt1_idx]
                        x2, y2, c2 = kpts[pt2_idx]
                        if c1 > 0.4 and c2 > 0.4:
                            cv2.line(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw joint points
                for (kx, ky, kconf) in kpts:
                    if kconf > 0.4:
                        cv2.circle(display_frame, (kx, ky), 4, (0, 0, 255), -1)

            # HUD Status Display
            cv2.rectangle(display_frame, (10, 10), (450, 115), (20, 20, 20), -1)
            cv2.putText(display_frame, "ASYMMETRIC CASCADED PIPELINE", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)
            cv2.putText(display_frame, f"STAGE 1 (NPU Detector): {npu_fps:5.1f} FPS", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(display_frame, f"STAGE 2 (GPU Pose Est): {gpu_fps:5.1f} FPS", (20, 92),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

            cv2.imshow("Asymmetric Dual-Engine Cascade (NPU + GPU)", display_frame)
            display_timer = now

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    stop_event.set()
    cam.stop()
    cv2.destroyAllWindows()
    print("Cascaded pipeline stopped cleanly.")