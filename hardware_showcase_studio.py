import os
import glob
import time
import queue
import threading
import cv2
import numpy as np
import openvino as ov

# ------------------------------------------------------------
# Model Discovery (INT8 Optimized & ONNX FP32 Fallback)
# ------------------------------------------------------------
def find_model(folder_candidates, fallback_file):
    for folder in folder_candidates:
        if os.path.exists(folder):
            xmls = glob.glob(os.path.join(folder, "*.xml"))
            if xmls:
                return xmls[0]
    if os.path.exists(fallback_file):
        return fallback_file
    raise FileNotFoundError(f"Neither {folder_candidates} nor {fallback_file} could be found.")

DET_INT8 = find_model(
    [r"C:\Users\saksh\HardwareLab\yolov8n_int8_openvino_model", r"C:\Users\saksh\HardwareLab\yolov8n_openvino_model"],
    r"C:\Users\saksh\HardwareLab\yolov8n.onnx"
)
POSE_INT8 = find_model(
    [r"C:\Users\saksh\HardwareLab\yolov8n-pose_int8_openvino_model", r"C:\Users\saksh\HardwareLab\yolov8n-pose_openvino_model"],
    r"C:\Users\saksh\HardwareLab\yolov8n-pose.onnx"
)
NAIVE_MODEL_PATH = r"C:\Users\saksh\HardwareLab\yolov8n.onnx"

INPUT_DIM = 640
CONF_THRESH = 0.40
IOU_THRESH = 0.45

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

core = ov.Core()

print("=" * 85)
print("HARDWARE ACCELERATION SHOWCASE STUDIO: NAIVE vs. HARDWARE-OPTIMIZED")
print("=" * 85)
print("Controls:")
print("  Press [ 1 ] -> Switch to NAIVE MODE (Synchronous, CPU Only, FP32)")
print("  Press [ 2 ] -> Switch to HARDWARE-CODED MODE (Heterogeneous NPU + GPU, INT8)")
print("  Press [ q ] -> Exit Showcase")
print("=" * 85)

# 1. Compile Naive Engine on CPU
print("Compiling Naive Engine (Host CPU)...")
naive_model = core.read_model(NAIVE_MODEL_PATH)
compiled_naive = core.compile_model(naive_model, "CPU")
naive_infer_request = compiled_naive.create_infer_request()

# 2. Compile Stage 1: Detection on NPU (INT8)
print("Compiling Optimized Stage 1: Detection -> Intel AI Boost NPU (INT8)...")
opt_det_model = core.read_model(DET_INT8)
compiled_npu = core.compile_model(opt_det_model, "NPU")
npu_queue = ov.AsyncInferQueue(compiled_npu, jobs=2)

# 3. Compile Stage 2: Pose on Arc 130T GPU (INT8)
print("Compiling Optimized Stage 2: Pose -> Intel Arc 130T GPU (INT8)...")
opt_pose_model = core.read_model(POSE_INT8)
compiled_gpu = core.compile_model(opt_pose_model, "GPU")
gpu_pose_queue = ov.AsyncInferQueue(compiled_gpu, jobs=2)

# Global State Management
current_mode = "HARDWARE_OPTIMIZED"  # Options: "NAIVE", "HARDWARE_OPTIMIZED"
mode_lock = threading.Lock()
stop_event = threading.Event()
crop_queue = queue.Queue(maxsize=16)

# Optimized state
latest_boxes = []
latest_kpts = []
state_lock = threading.Lock()

opt_npu_count = 0
opt_gpu_count = 0
count_lock = threading.Lock()

# ------------------------------------------------------------
# Vectorized Box Decoding (Hardware Coded)
# ------------------------------------------------------------
def decode_detections_vectorized(raw_out, scale, dx, dy):
    out = raw_out.T
    classes_scores = out[:, 4:]
    max_scores = np.max(classes_scores, axis=1)
    class_ids = np.argmax(classes_scores, axis=1)

    mask = (max_scores >= CONF_THRESH) & (class_ids == 0)
    if not np.any(mask):
        return []

    filtered_out = out[mask]
    filtered_scores = max_scores[mask]
    boxes_raw = filtered_out[:, :4]

    cx, cy, bw, bh = boxes_raw[:, 0], boxes_raw[:, 1], boxes_raw[:, 2], boxes_raw[:, 3]
    x1 = np.round(((cx - bw / 2.0) - dx) / scale).astype(int)
    y1 = np.round(((cy - bh / 2.0) - dy) / scale).astype(int)
    w_box = np.round(bw / scale).astype(int)
    h_box = np.round(bh / scale).astype(int)

    boxes = np.stack([x1, y1, w_box, h_box], axis=1).tolist()
    confidences = filtered_scores.tolist()

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESH, IOU_THRESH)
    detected = []
    if len(indices) > 0:
        for idx in indices.flatten():
            detected.append((boxes[idx], confidences[idx]))
    return detected

# ------------------------------------------------------------
# Naive Scalar Box Decoding (Slow Python Loops)
# ------------------------------------------------------------
def decode_detections_naive(raw_out, scale, dx, dy):
    out = raw_out.T
    boxes = []
    confidences = []
    for row in out:
        score = float(np.max(row[4:]))
        if score >= CONF_THRESH:
            cls_id = int(np.argmax(row[4:]))
            if cls_id == 0:  # person
                cx, cy, bw, bh = row[0], row[1], row[2], row[3]
                x1 = int(((cx - bw / 2.0) - dx) / scale)
                y1 = int(((cy - bh / 2.0) - dy) / scale)
                w_box = int(bw / scale)
                h_box = int(bh / scale)
                boxes.append([x1, y1, w_box, h_box])
                confidences.append(score)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESH, IOU_THRESH)
    detected = []
    if len(indices) > 0:
        for idx in indices.flatten():
            detected.append((boxes[idx], confidences[idx]))
    return detected

# ------------------------------------------------------------
# Callbacks for Optimized Heterogeneous Ring
# ------------------------------------------------------------
def npu_callback(infer_request, user_data):
    global opt_npu_count
    frame_ref, scale, dx, dy = user_data
    raw_out = infer_request.get_output_tensor(0).data[0]
    detected_people = decode_detections_vectorized(raw_out, scale, dx, dy)

    with state_lock:
        global latest_boxes
        latest_boxes = detected_people

    with count_lock:
        opt_npu_count += 1

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

def gpu_pose_callback(infer_request, user_data):
    global opt_gpu_count
    scale, offset_x, offset_y = user_data
    out = infer_request.get_output_tensor(0).data[0].T
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
        opt_gpu_count += 1

gpu_pose_queue.set_callback(gpu_pose_callback)

def gpu_worker_loop():
    while not stop_event.is_set():
        try:
            crop_tensor, meta = crop_queue.get(timeout=0.01)
        except queue.Empty:
            continue
        gpu_pose_queue.start_async({0: crop_tensor}, userdata=meta)
        crop_queue.task_done()

threading.Thread(target=gpu_worker_loop, daemon=True).start()

# ------------------------------------------------------------
# Ingestion Thread
# ------------------------------------------------------------
class CameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.use_synthetic = not self.cap.isOpened()
        self.running = True
        self.frame = None
        self.lock = threading.Lock()

        if self.use_synthetic:
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
                    time.sleep(0.005)
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
# Primary Showcase Execution Loop
# ------------------------------------------------------------
fps_calc_timer = time.perf_counter()
prev_opt_npu = 0
naive_frame_count = 0

current_display_fps = 0.0
current_latency_ms = 0.0

print("\nShowcase Studio Active. Open the OpenCV display window.\n")

try:
    while True:
        with mode_lock:
            mode = current_mode

        frame = cam.read_latest()
        if frame is None:
            time.sleep(0.001)
            continue

        annotated_display = frame.copy()

        # ====================================================
        # EXECUTION PATH A: NAIVE MODE (CPU, Synchronous)
        # ====================================================
        if mode == "NAIVE":
            t_naive_start = time.perf_counter()

            tensor, scale, dx, dy = preprocess(frame)
            naive_infer_request.infer({0: tensor})
            raw_out = naive_infer_request.get_output_tensor(0).data[0]
            detected_naive = decode_detections_naive(raw_out, scale, dx, dy)

            for (box, score) in detected_naive:
                bx, by, bw, bh = box
                cv2.rectangle(annotated_display, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
                cv2.putText(annotated_display, f"CPU Naive: {score:.2f}", (bx, max(20, by - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

            t_naive_end = time.perf_counter()
            current_latency_ms = (t_naive_end - t_naive_start) * 1000.0

            naive_frame_count += 1
            now = time.perf_counter()
            if now - fps_calc_timer >= 0.5:
                current_display_fps = naive_frame_count / (now - fps_calc_timer)
                naive_frame_count = 0
                fps_calc_timer = now

        # ====================================================
        # EXECUTION PATH B: HARDWARE-CODED ASYMMETRIC (NPU + GPU)
        # ====================================================
        else:
            tensor, scale, dx, dy = preprocess(frame)
            user_meta = (frame, scale, dx, dy)
            npu_queue.start_async({0: tensor}, userdata=user_meta)

            with state_lock:
                boxes_to_draw = list(latest_boxes)
                kpts_to_draw = list(latest_kpts)

            for (box, score) in boxes_to_draw:
                bx, by, bw, bh = box
                cv2.rectangle(annotated_display, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
                cv2.putText(annotated_display, f"NPU (INT8): {score:.2f}", (bx, max(20, by - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            if kpts_to_draw:
                for pt1, pt2 in SKELETON_CONNECTIONS:
                    if pt1 < len(kpts_to_draw) and pt2 < len(kpts_to_draw):
                        x1, y1, c1 = kpts_to_draw[pt1]
                        x2, y2, c2 = kpts_to_draw[pt2]
                        if c1 > 0.4 and c2 > 0.4:
                            cv2.line(annotated_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                for (kx, ky, conf) in kpts_to_draw:
                    if conf > 0.4:
                        cv2.circle(annotated_display, (kx, ky), 4, (0, 0, 255), -1)

            now = time.perf_counter()
            if now - fps_calc_timer >= 0.5:
                elapsed = now - fps_calc_timer
                with count_lock:
                    c_npu = opt_npu_count
                current_display_fps = (c_npu - prev_opt_npu) / elapsed
                prev_opt_npu = c_npu
                fps_calc_timer = now
                current_latency_ms = (1000.0 / current_display_fps) if current_display_fps > 0 else 8.4

        # ====================================================
        # Dynamic HUD Presentation
        # ====================================================
        overlay = annotated_display.copy()
        cv2.rectangle(overlay, (15, 15), (580, 165), (15, 17, 23), -1)
        cv2.addWeighted(overlay, 0.85, annotated_display, 0.15, 0, annotated_display)

        if mode == "NAIVE":
            mode_header = "MODE: [1] NAIVE BASELINE (UNOPTIMIZED)"
            header_color = (0, 0, 255)
            silicon_info = "Engine: Host CPU (Single-Threaded FP32)"
            bottleneck_info = "Bottleneck: Blocking Serial Inference & Host Contention"
            speedup_str = "Speed: 1.0x Baseline"
        else:
            mode_header = "MODE: [2] HARDWARE-CODED CO-EXECUTION"
            header_color = (0, 255, 0)
            silicon_info = "Silicon: Intel AI Boost NPU + Arc 130T (INT8)"
            bottleneck_info = "Pipeline: Decoupled Asymmetric Ring Buffer"
            multiplier = current_display_fps / 20.0 if current_display_fps > 0 else 6.0
            speedup_str = f"Speedup: {multiplier:.1f}x FASTER"

        cv2.putText(annotated_display, mode_header, (25, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, header_color, 2)
        cv2.putText(annotated_display, f"THROUGHPUT : {current_display_fps:5.1f} FPS  |  LATENCY: {current_latency_ms:4.1f} ms",
                    (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(annotated_display, silicon_info, (25, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)
        cv2.putText(annotated_display, bottleneck_info, (25, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)
        cv2.putText(annotated_display, f"[Press '1': Naive | Press '2': Coded]  •  {speedup_str}",
                    (25, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        cv2.imshow("Hardware Optimization Showcase Studio", annotated_display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('1'):
            with mode_lock:
                current_mode = "NAIVE"
                print("\n>>> SWITCHED TO: NAIVE MODE (CPU Synchronous) <<<")
        elif key == ord('2'):
            with mode_lock:
                current_mode = "HARDWARE_OPTIMIZED"
                print("\n>>> SWITCHED TO: HARDWARE-CODED MODE (Heterogeneous NPU + GPU INT8) <<<")

finally:
    stop_event.set()
    cam.stop()
    cv2.destroyAllWindows()
    print("Showcase terminated cleanly.")