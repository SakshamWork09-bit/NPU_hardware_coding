import os
import time
import queue
import threading
import cv2
import numpy as np
import openvino as ov

MODEL_PATH = r"C:\Users\saksh\HardwareLab\mobilenetv2-7.onnx"
LABELS_PATH = r"C:\Users\saksh\HardwareLab\imagenet_classes.txt"
INPUT_W, INPUT_H = 224, 224

# Worker configuration: 1 GPU worker + 2 NPU workers
WORKER_CONFIG = {
    "GPU": 1,
    "NPU": 2
}

# ------------------------------------------------------------
# Load ImageNet 1,000-Class Labels
# ------------------------------------------------------------
labels = []
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(labels)} ImageNet classification labels.")
else:
    print("[WARNING] imagenet_classes.txt not found. Using numeric class IDs.")
    labels = [f"Class #{i}" for i in range(1000)]

core = ov.Core()

print("=" * 85)
print("HETEROGENEOUS VISION ENGINE WITH REAL-TIME TOP-1 & TOP-5 IMAGENET LABELS")
print("=" * 85)
print("Available Devices:", core.available_devices)
print("Target Workers   :", WORKER_CONFIG)
print("Press 'q' in the video window to quit.")
print("=" * 85)

# Load base model
base_model = core.read_model(MODEL_PATH)
base_model.reshape({base_model.inputs[0].any_name: [1, 3, INPUT_H, INPUT_W]})
input_name = base_model.inputs[0].any_name

# Compile models for GPU and NPU
compiled_models = {}
for dev in WORKER_CONFIG:
    if dev in core.available_devices:
        print(f"Compiling model for {dev}...")
        compiled_models[dev] = core.compile_model(base_model, dev)
    else:
        raise RuntimeError(f"Required device {dev} not found.")

task_queue = queue.Queue(maxsize=16)
stop_event = threading.Event()

# Telemetry and prediction stores
device_counts = {"GPU": 0, "NPU": 0}
counter_lock = threading.Lock()

latest_top5 = []
result_lock = threading.Lock()

# Numerically stable Softmax calculation
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# ------------------------------------------------------------
# Worker Thread (Inference + Top-5 Probability Extraction)
# ------------------------------------------------------------
def engine_worker(device_name, compiled_model):
    infer_request = compiled_model.create_infer_request()
    
    while not stop_event.is_set():
        try:
            tensor = task_queue.get(timeout=0.01)
        except queue.Empty:
            continue

        infer_request.infer({input_name: tensor})
        out = infer_request.get_output_tensor(0).data[0]

        with counter_lock:
            device_counts[device_name] += 1

        # Calculate softmax probabilities and top-5 indices
        probs = softmax(out)
        top5_indices = np.argsort(probs)[-5:][::-1]
        
        top5_data = []
        for idx in top5_indices:
            label_name = labels[idx] if idx < len(labels) else f"Class #{idx}"
            top5_data.append((label_name, probs[idx] * 100.0))

        with result_lock:
            global latest_top5
            latest_top5 = top5_data

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
            print("\n[NOTE] No webcam detected. Generating synthetic moving pattern.")
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
                cv2.circle(f, (int((t * 500) % 1920), 540), 120, (0, 255, 0), -1)
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

# Fast CPU downsampling
def preprocess(frame):
    resized = cv2.resize(frame, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return np.expand_dims(tensor, axis=0)

# ------------------------------------------------------------
# Monitoring & Display Loop
# ------------------------------------------------------------
fps_timer = time.perf_counter()
display_timer = time.perf_counter()
prev_gpu_count = 0
prev_npu_count = 0
total_fps = 0.0
gpu_fps = 0.0
npu_fps = 0.0

print("\nRunning live inference with Top-5 label extraction...\n")

try:
    while True:
        frame = cam.read_latest()
        if frame is None:
            time.sleep(0.001)
            continue

        tensor = preprocess(frame)
        try:
            task_queue.put(tensor, block=False)
        except queue.Full:
            pass

        now = time.perf_counter()

        # Update telemetry every 1.0 second
        if now - fps_timer >= 1.0:
            elapsed = now - fps_timer
            with counter_lock:
                curr_gpu = device_counts["GPU"]
                curr_npu = device_counts["NPU"]

            gpu_fps = (curr_gpu - prev_gpu_count) / elapsed
            npu_fps = (curr_npu - prev_npu_count) / elapsed
            total_fps = gpu_fps + npu_fps

            prev_gpu_count = curr_gpu
            prev_npu_count = curr_npu
            fps_timer = now

            with result_lock:
                top1_str = f"{latest_top5[0][0]} ({latest_top5[0][1]:.1f}%)" if latest_top5 else "N/A"

            print(f"[HETERO] Total: {total_fps:7.1f} FPS | "
                  f"GPU: {gpu_fps:7.1f} | NPU: {npu_fps:7.1f} FPS | "
                  f"Top-1: {top1_str}")

        # Render display at 30 Hz
        if now - display_timer >= 0.033:
            display_frame = cv2.resize(frame, (960, 540))
            
            # Semi-transparent HUD overlay
            hud_w, hud_h = 440, 220
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (15, 15), (15 + hud_w, 15 + hud_h), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.75, display_frame, 0.25, 0, display_frame)

            # Performance stats
            cv2.putText(display_frame, "HETEROGENEOUS ENGINE (GPU + NPU)", (25, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)
            cv2.putText(display_frame, f"THROUGHPUT: {total_fps:6.1f} FPS", (25, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(display_frame, f"GPU: {gpu_fps:5.1f} | NPU: {npu_fps:5.1f} FPS", (25, 92),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1)

            # Render Top-5 Predictions
            cv2.putText(display_frame, "TOP-5 CANDIDATE LABELS:", (25, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 200, 0), 1)

            with result_lock:
                top5_snapshot = list(latest_top5)

            y_offset = 142
            for i, (name, prob) in enumerate(top5_snapshot[:5]):
                # Truncate label if too long for HUD box
                short_name = name[:24] + ".." if len(name) > 26 else name
                color = (0, 255, 0) if i == 0 else (220, 220, 220)
                text = f"{i+1}. {short_name:<26} {prob:4.1f}%"
                cv2.putText(display_frame, text, (30, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                y_offset += 20

            cv2.imshow("Heterogeneous OpenVINO Classifier", display_frame)
            display_timer = now

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    stop_event.set()
    cam.stop()
    cv2.destroyAllWindows()
    print("Application closed.")