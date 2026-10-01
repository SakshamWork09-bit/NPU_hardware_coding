import os
import glob
import time
import queue
import threading
import cv2
import numpy as np
import openvino as ov
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

# ------------------------------------------------------------
# Locate Quantized INT8 Models Dynamically
# ------------------------------------------------------------
def find_model_xml(folder_candidates):
    for folder in folder_candidates:
        if os.path.exists(folder):
            xmls = glob.glob(os.path.join(folder, "*.xml"))
            if xmls:
                return xmls[0]
    raise FileNotFoundError(f"Could not find model .xml in: {folder_candidates}")

DET_MODEL_XML = find_model_xml([
    r"C:\Users\saksh\HardwareLab\yolov8n_int8_openvino_model",
    r"C:\Users\saksh\HardwareLab\yolov8n_openvino_model"
])

POSE_MODEL_XML = find_model_xml([
    r"C:\Users\saksh\HardwareLab\yolov8n-pose_int8_openvino_model",
    r"C:\Users\saksh\HardwareLab\yolov8n-pose_openvino_model"
])

INPUT_DIM = 640
DET_CONF_THRESH = 0.40
DET_IOU_THRESH = 0.45

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

core = ov.Core()
print("=" * 80)
print("DECOUPLED HIGH-SPEED EDGE STREAMING SERVER (120+ FPS TARGET)")
print("=" * 80)

# Compile models on target silicon
det_model = core.read_model(DET_MODEL_XML)
compiled_det = core.compile_model(det_model, "NPU")
npu_queue = ov.AsyncInferQueue(compiled_det, jobs=2)

pose_model = core.read_model(POSE_MODEL_XML)
compiled_pose = core.compile_model(pose_model, "GPU")
gpu_pose_queue = ov.AsyncInferQueue(compiled_pose, jobs=2)

crop_queue = queue.Queue(maxsize=16)
stop_event = threading.Event()

# Shared state and telemetry
telemetry = {
    "npu_fps": 0.0,
    "gpu_fps": 0.0,
    "subjects_count": 0,
    "status": "Running",
    "devices": "Intel NPU + GPU (INT8)"
}
telemetry_lock = threading.Lock()

latest_boxes = []
latest_kpts = []
state_lock = threading.Lock()

latest_raw_frame = None
latest_frame_lock = threading.Lock()

latest_jpeg_bytes = None
jpeg_lock = threading.Lock()

npu_counter = 0
gpu_counter = 0
count_lock = threading.Lock()

# ------------------------------------------------------------
# Decoding & Callbacks
# ------------------------------------------------------------
def decode_person_detections(raw_out, scale, dx, dy):
    out = raw_out.T
    classes_scores = out[:, 4:]
    max_scores = np.max(classes_scores, axis=1)
    class_ids = np.argmax(classes_scores, axis=1)

    mask = (max_scores >= DET_CONF_THRESH) & (class_ids == 0)
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

    indices = cv2.dnn.NMSBoxes(boxes, confidences, DET_CONF_THRESH, DET_IOU_THRESH)
    detected_people = []
    if len(indices) > 0:
        for idx in indices.flatten():
            detected_people.append((boxes[idx], confidences[idx]))
    return detected_people

def npu_callback(infer_request, user_data):
    global npu_counter
    frame_ref, scale, dx, dy = user_data
    raw_out = infer_request.get_output_tensor(0).data[0]
    detected_people = decode_person_detections(raw_out, scale, dx, dy)

    with state_lock:
        global latest_boxes
        latest_boxes = detected_people

    with count_lock:
        npu_counter += 1

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
    global gpu_counter
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
        gpu_counter += 1

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
# PURE INFERENCE LOOP (NO SLEEPS, NO JPEG COMPRESSION, RUNS FULL SPEED)
# ------------------------------------------------------------
def inference_background_worker():
    global latest_raw_frame
    fps_timer = time.perf_counter()
    prev_npu = 0
    prev_gpu = 0

    while not stop_event.is_set():
        frame = cam.read_latest()
        if frame is None:
            time.sleep(0.001)
            continue

        with latest_frame_lock:
            latest_raw_frame = frame

        tensor, scale, dx, dy = preprocess(frame)
        user_meta = (frame, scale, dx, dy)
        npu_queue.start_async({0: tensor}, userdata=user_meta)

        now = time.perf_counter()
        if now - fps_timer >= 1.0:
            elapsed = now - fps_timer
            with count_lock:
                c_npu, c_gpu = npu_counter, gpu_counter

            current_npu_fps = (c_npu - prev_npu) / elapsed
            current_gpu_fps = (c_gpu - prev_gpu) / elapsed
            prev_npu, prev_gpu = c_npu, c_gpu
            fps_timer = now

            with state_lock:
                n_subjects = len(latest_boxes)

            with telemetry_lock:
                telemetry["npu_fps"] = round(current_npu_fps, 1)
                telemetry["gpu_fps"] = round(current_gpu_fps, 1)
                telemetry["subjects_count"] = n_subjects

            print(f"[EDGE SERVER] NPU: {current_npu_fps:5.1f} FPS | GPU: {current_gpu_fps:5.1f} FPS | Subjects: {n_subjects}")

threading.Thread(target=inference_background_worker, daemon=True, name="AI_Inference_Thread").start()

# ------------------------------------------------------------
# DEDICATED WEB ENCODER THREAD (ISOLATED AT 30 HZ)
# ------------------------------------------------------------
def web_encoder_worker():
    global latest_jpeg_bytes
    while not stop_event.is_set():
        time.sleep(0.033)  # Encodes strictly at 30 FPS for browser stream

        with latest_frame_lock:
            if latest_raw_frame is None:
                continue
            annotated = latest_raw_frame.copy()

        with state_lock:
            boxes = list(latest_boxes)
            kpts = list(latest_kpts)

        # Draw detections
        for (box, score) in boxes:
            bx, by, bw, bh = box
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
            cv2.putText(annotated, f"Person {score:.2f}", (bx, max(20, by - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # Draw skeleton
        if kpts:
            for pt1, pt2 in SKELETON_CONNECTIONS:
                if pt1 < len(kpts) and pt2 < len(kpts):
                    x1, y1, c1 = kpts[pt1]
                    x2, y2, c2 = kpts[pt2]
                    if c1 > 0.4 and c2 > 0.4:
                        cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for (kx, ky, conf) in kpts:
                if conf > 0.4:
                    cv2.circle(annotated, (kx, ky), 4, (0, 0, 255), -1)

        # Compress to JPEG
        ret, buffer = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if ret:
            with jpeg_lock:
                latest_jpeg_bytes = buffer.tobytes()

threading.Thread(target=web_encoder_worker, daemon=True, name="Web_Encoder_Thread").start()

# ------------------------------------------------------------
# FastAPI Web Endpoints
# ------------------------------------------------------------
app = FastAPI(title="Edge Vision Server")

HTML_DASHBOARD = """
<!DOCTYPE html>
<html>
<head>
    <title>Intel Core Ultra Edge Vision Telemetry</title>
    <style>
        body { background: #0f1117; color: #e6edf3; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 20px; }
        .container { max-width: 1100px; margin: 0 auto; }
        .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #30363d; padding-bottom: 15px; margin-bottom: 20px; }
        .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }
        .video-box { background: #161b22; border: 1px solid #30363d; border-radius: 8px; overflow: hidden; }
        .video-box img { width: 100%; height: auto; display: block; }
        .cards { display: flex; flex-direction: column; gap: 12px; }
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
        .card h3 { margin: 0 0 6px 0; font-size: 13px; color: #8b949e; text-transform: uppercase; }
        .card .value { font-size: 26px; font-weight: 700; color: #58a6ff; }
        .badge { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; background: #238636; color: #fff; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1 style="margin:0; font-size:22px;">Intel Core Ultra Heterogeneous Edge Vision</h1>
                <p style="margin:4px 0 0 0; color:#8b949e;">Stage 1 (NPU Detection) + Stage 2 (GPU Pose) - INT8 Native</p>
            </div>
            <div class="badge">SERVER ACTIVE</div>
        </div>
        <div class="grid">
            <div class="video-box">
                <img src="/video_feed" alt="Live Stream">
            </div>
            <div class="cards">
                <div class="card">
                    <h3>NPU Detector Throughput</h3>
                    <div class="value" id="npu-fps">-- FPS</div>
                </div>
                <div class="card">
                    <h3>GPU Pose Throughput</h3>
                    <div class="value" id="gpu-fps">-- FPS</div>
                </div>
                <div class="card">
                    <h3>Active Subjects</h3>
                    <div class="value" id="subjects">0</div>
                </div>
                <div class="card">
                    <h3>Active Hardware Silicon</h3>
                    <div style="font-size: 15px; font-weight: 600; color:#7ee787; margin-top: 4px;">Intel AI Boost NPU + Arc 130T (INT8)</div>
                </div>
            </div>
        </div>
    </div>
    <script>
        setInterval(() => {
            fetch('/api/telemetry')
                .then(res => res.json())
                .then(data => {
                    document.getElementById('npu-fps').innerText = data.npu_fps + ' FPS';
                    document.getElementById('gpu-fps').innerText = data.gpu_fps + ' FPS';
                    document.getElementById('subjects').innerText = data.subjects_count;
                });
        }, 500);
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    return HTML_DASHBOARD

@app.get("/api/telemetry")
def get_telemetry():
    with telemetry_lock:
        return telemetry

def stream_generator():
    while True:
        with jpeg_lock:
            frame_data = latest_jpeg_bytes

        if frame_data is None:
            time.sleep(0.01)
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        time.sleep(0.033)

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        stream_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")