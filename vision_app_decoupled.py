import os
import time
import threading
import cv2
import numpy as np
import openvino as ov

MODEL_PATH = r"C:\Users\saksh\HardwareLab\mobilenetv2-7.onnx"
INPUT_W, INPUT_H = 224, 224

core = ov.Core()

print("=" * 80)
print("DECOUPLED ASYNCHRONOUS REAL-TIME VISION ENGINE")
print("=" * 80)
print("Available Devices:", core.available_devices)
print("\nControls:")
print("  Press 'g' in display window -> Switch to GPU (Intel Arc 130T)")
print("  Press 'n' in display window -> Switch to NPU (Intel AI Boost)")
print("  Press 'c' in display window -> Switch to CPU (Core Ultra 5 225H)")
print("  Press 'q' in display window -> Quit Application")
print("=" * 80)

# Load and reshape model
base_model = core.read_model(MODEL_PATH)
base_model.reshape({base_model.inputs[0].any_name: [1, 3, INPUT_H, INPUT_W]})
input_name = base_model.inputs[0].any_name

# Pre-compile engines
compiled_engines = {}
for dev in core.available_devices:
    try:
        print(f"Pre-compiling engine for {dev}...")
        compiled = core.compile_model(base_model, dev)
        queue = ov.AsyncInferQueue(compiled, jobs=2)
        compiled_engines[dev] = (compiled, queue)
    except Exception as e:
        print(f"Could not initialize {dev}: {e}")

current_device = "GPU" if "GPU" in compiled_engines else "CPU"
_, current_queue = compiled_engines[current_device]

# ============================================================
# THREADED CAMERA CAPTURE (Prevents 30 FPS Blocking)
# ============================================================
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
                # Mock moving visual frame
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

# Preprocessing: Fast CPU downsample to compact 224x224 tensor
def preprocess(frame):
    resized = cv2.resize(frame, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return np.expand_dims(tensor, axis=0)

inference_count = 0
latest_prediction = "Initializing..."
latest_conf = 0.0

def callback(infer_request, user_data):
    global inference_count, latest_prediction, latest_conf
    inference_count += 1
    out = infer_request.get_output_tensor(0).data
    latest_prediction = f"Class #{np.argmax(out[0])}"
    latest_conf = float(np.max(out[0]))

for dev in compiled_engines:
    compiled_engines[dev][1].set_callback(callback)

# ============================================================
# MAIN INFERENCE & MONITORING LOOP
# ============================================================
fps_timer = time.perf_counter()
display_timer = time.perf_counter()
inferences_last_sec = 0
display_fps = 0.0

print("\nStarting unthrottled inference loop...\n")

try:
    while True:
        # Fetch latest frame from camera thread without blocking
        frame = cam.read_latest()
        if frame is None:
            time.sleep(0.001)
            continue

        # Downsample and launch async inference
        tensor = preprocess(frame)
        current_queue.start_async({input_name: tensor})

        now = time.perf_counter()

        # Update and print real hardware throughput every 1.0 second
        if now - fps_timer >= 1.0:
            elapsed = now - fps_timer
            display_fps = inference_count / elapsed
            print(f"[{current_device}] AI Inference Speed: {display_fps:7.1f} FPS | "
                  f"Latency: {1000.0/display_fps:5.2f} ms | {latest_prediction} ({latest_conf:.2f})")
            inference_count = 0
            fps_timer = now

        # Render display at a capped 30 Hz to eliminate GPU presentation contention
        if now - display_timer >= 0.033:
            display_frame = cv2.resize(frame, (640, 360))
            cv2.rectangle(display_frame, (10, 10), (380, 110), (0, 0, 0), -1)
            cv2.putText(display_frame, f"ENGINE: {current_device}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_frame, f"AI SPEED: {display_fps:6.1f} FPS", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"RESULT  : {latest_prediction}", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Decoupled OpenVINO Stream", display_frame)
            display_timer = now

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('g') and "GPU" in compiled_engines:
                current_queue.wait_all()
                current_device = "GPU"
                current_queue = compiled_engines["GPU"][1]
                print("\n>>> Switched to GPU <<<\n")
            elif key == ord('n') and "NPU" in compiled_engines:
                current_queue.wait_all()
                current_device = "NPU"
                current_queue = compiled_engines["NPU"][1]
                print("\n>>> Switched to NPU <<<\n")
            elif key == ord('c') and "CPU" in compiled_engines:
                current_queue.wait_all()
                current_device = "CPU"
                current_queue = compiled_engines["CPU"][1]
                print("\n>>> Switched to CPU <<<\n")

finally:
    cam.stop()
    cv2.destroyAllWindows()
    print("Application closed.")