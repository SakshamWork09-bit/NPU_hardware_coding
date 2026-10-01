import os
import time
import queue
import threading
import cv2
import numpy as np
import openvino as ov

MODEL_PATH = r"C:\Users\saksh\HardwareLab\mobilenetv2-7.onnx"
INPUT_W, INPUT_H = 224, 224

# Optimal worker allocation determined from our scheduler sweep:
# 1 GPU worker (saturates Arc 130T) + 2 NPU workers (optimal pipeline depth)
WORKER_CONFIG = {
    "GPU": 1,
    "NPU": 2
}

core = ov.Core()

print("=" * 80)
print("HETEROGENEOUS CO-EXECUTION PIPELINE (GPU + NPU CONCURRENT INFERENCE)")
print("=" * 80)
print("Available Devices:", core.available_devices)
print("Target Workers   :", WORKER_CONFIG)
print("Press 'q' in the video window to quit.")
print("=" * 80)

# Load and configure base model
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

# Shared bounded input queue (prevents queue bloat and latency spikes)
task_queue = queue.Queue(maxsize=16)
stop_event = threading.Event()

# Telemetry counters
device_counts = {"GPU": 0, "NPU": 0}
counter_lock = threading.Lock()
latest_result = {"class": "Initializing...", "conf": 0.0}
result_lock = threading.Lock()

# ------------------------------------------------------------
# Worker Thread (Parallel Engine Consumer)
# ------------------------------------------------------------
def engine_worker(device_name, compiled_model):
    infer_request = compiled_model.create_infer_request()
    
    while not stop_event.is_set():
        try:
            tensor = task_queue.get(timeout=0.01)
        except queue.Empty:
            continue

        # Run synchronous inference on this dedicated worker
        infer_request.infer({input_name: tensor})
        out = infer_request.get_output_tensor(0).data

        # Update telemetry
        with counter_lock:
            device_counts[device_name] += 1

        top_class = int(np.argmax(out[0]))
        top_conf = float(np.max(out[0]))
        with result_lock:
            latest_result["class"] = f"Class #{top_class}"
            latest_result["conf"] = top_conf

        task_queue.task_done()

# Start accelerator worker threads
worker_threads = []
for dev, count in WORKER_CONFIG.items():
    for i in range(count):
        t = threading.Thread(
            target=engine_worker,
            args=(dev, compiled_models[dev]),
            daemon=True,
            name=f"{dev}_Worker_{i}"
        )
        t.start()
        worker_threads.append(t)

# ------------------------------------------------------------
# Camera Ingestion Thread
# ------------------------------------------------------------
class CameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.use_synthetic = not self.cap.isOpened()
        self.running = True
        self.frame = None
        self.lock = threading.Lock()

        if self.use_synthetic:
            print("\n[NOTE] No physical webcam detected. Generating synthetic frames.")
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

# Preprocessing: CPU downsamples to 224x224
def preprocess(frame):
    resized = cv2.resize(frame, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return np.expand_dims(tensor, axis=0)

# ------------------------------------------------------------
# Main Pipeline Dispatcher & Display Loop
# ------------------------------------------------------------
fps_timer = time.perf_counter()
display_timer = time.perf_counter()
prev_gpu_count = 0
prev_npu_count = 0

# Properly initialized telemetry variables
total_fps = 0.0
gpu_fps = 0.0
npu_fps = 0.0

print("\nStarting Heterogeneous Co-Execution loop...\n")

try:
    while True:
        frame = cam.read_latest()
        if frame is None:
            time.sleep(0.001)
            continue

        # Downsample on CPU and push to heterogeneous worker pool
        tensor = preprocess(frame)
        try:
            task_queue.put(tensor, block=False)
        except queue.Full:
            # Drop frame if all workers are fully saturated (maintains real-time freshness)
            pass

        now = time.perf_counter()

        # Report aggregate and per-engine throughput every second
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
                pred = latest_result["class"]
                conf = latest_result["conf"]

            print(f"[HETERO] Total: {total_fps:7.1f} FPS | "
                  f"GPU: {gpu_fps:7.1f} FPS | "
                  f"NPU: {npu_fps:7.1f} FPS | {pred} ({conf:.2f})")

        # Capped 30 Hz GUI display window
        if now - display_timer >= 0.033:
            display_frame = cv2.resize(frame, (640, 360))
            cv2.rectangle(display_frame, (10, 10), (420, 130), (0, 0, 0), -1)
            cv2.putText(display_frame, "ENGINE: GPU + NPU (CO-EXECUTION)", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(display_frame, f"TOTAL SPEED: {total_fps:6.1f} FPS", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"GPU: {gpu_fps:5.1f} | NPU: {npu_fps:5.1f} FPS", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            with result_lock:
                res_text = f"{latest_result['class']} ({latest_result['conf']:.2f})"
            cv2.putText(display_frame, f"RESULT: {res_text}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            cv2.imshow("Heterogeneous OpenVINO Stream", display_frame)
            display_timer = now

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    stop_event.set()
    cam.stop()
    cv2.destroyAllWindows()
    print("Heterogeneous pipeline stopped cleanly.")