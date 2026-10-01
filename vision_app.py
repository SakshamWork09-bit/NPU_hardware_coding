import os
import time
import cv2
import numpy as np
import openvino as ov

# ============================================================
# REAL-TIME MULTI-ACCELERATOR VISION APPLICATION
# ============================================================

MODEL_PATH = r"C:\Users\saksh\HardwareLab\mobilenetv2-7.onnx"
INPUT_W, INPUT_H = 224, 224

# Labels (ImageNet 1000 simplified mock or top indices)
core = ov.Core()

print("=" * 75)
print("REAL-TIME MULTI-ACCELERATOR VISION APPLICATION")
print("=" * 75)
print("Available Devices:", core.available_devices)
print("\nControls:")
print("  Press 'g' -> Switch to GPU (Intel Arc 130T)")
print("  Press 'n' -> Switch to NPU (Intel AI Boost)")
print("  Press 'c' -> Switch to CPU (Core Ultra 5 225H)")
print("  Press 'q' -> Quit Application")
print("=" * 75)

# Load base model
base_model = core.read_model(MODEL_PATH)
base_model.reshape({base_model.inputs[0].any_name: [1, 3, INPUT_H, INPUT_W]})
input_name = base_model.inputs[0].any_name
output_name = base_model.outputs[0].any_name

# Pre-compile engines for instantaneous switching
compiled_engines = {}
for dev in core.available_devices:
    try:
        print(f"Pre-compiling engine for {dev}...")
        compiled = core.compile_model(base_model, dev)
        # 2 jobs for double-buffering latency overlap
        queue = ov.AsyncInferQueue(compiled, jobs=2)
        compiled_engines[dev] = (compiled, queue)
    except Exception as e:
        print(f"Could not initialize {dev}: {e}")

current_device = "GPU" if "GPU" in compiled_engines else "CPU"
_, current_queue = compiled_engines[current_device]

# Video source: Try web camera (0); if unavailable, fallback to synthetic test video
cap = cv2.VideoCapture(0)
use_synthetic = not cap.isOpened()
if use_synthetic:
    print("\n[NOTE] No physical webcam detected. Running on synthetic 1080p video stream.")

latest_prediction = "Initializing..."
latest_confidence = 0.0

def completion_callback(infer_request, user_data):
    global latest_prediction, latest_confidence
    output_tensor = infer_request.get_output_tensor(0).data
    class_idx = np.argmax(output_tensor[0])
    prob = np.max(output_tensor[0])
    latest_prediction = f"Class ID #{class_idx}"
    latest_confidence = float(prob)

# Set callbacks on all queues
for dev in compiled_engines:
    compiled_engines[dev][1].set_callback(completion_callback)

# Preprocessing: Host CPU spatial resize and transposition
def preprocess(frame):
    # Resize to model dimensions
    resized = cv2.resize(frame, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
    # BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # HWC to CHW, batch dimension, FP32 normalize
    tensor = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return np.expand_dims(tensor, axis=0)

frame_times = []
fps_display = 0.0

try:
    while True:
        t_start = time.perf_counter()

        if not use_synthetic:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            # Generate moving pattern for synthetic 1080p frame
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            cv2.circle(frame, (int((time.perf_counter()*400) % 1920), 540), 120, (0, 255, 0), -1)

        # 1. Pipeline Preprocessing on CPU
        preproc_tensor = preprocess(frame)

        # 2. Asynchronous Inference on Selected Accelerator
        current_queue.start_async({input_name: preproc_tensor})

        # Calculate sliding FPS
        t_elapsed = time.perf_counter() - t_start
        frame_times.append(t_elapsed)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps_display = 1.0 / (sum(frame_times) / len(frame_times))

        # 3. Draw HUD Display
        hud_bg = frame[:160, :520].copy()
        cv2.rectangle(frame, (0, 0), (520, 160), (20, 20, 20), -1)
        cv2.addWeighted(hud_bg, 0.2, frame[:160, :520], 0.8, 0, frame[:160, :520])

        cv2.putText(frame, f"ACTIVE ENGINE: {current_device}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, f"THROUGHPUT   : {fps_display:6.1f} FPS ({t_elapsed*1000:4.1f} ms)", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"TOP RESULT   : {latest_prediction} ({latest_confidence:.2f})", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        cv2.putText(frame, "Switch: [G]=GPU | [N]=NPU | [C]=CPU | [Q]=Quit", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        cv2.imshow("OpenVINO Heterogeneous Vision Engine", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('g') and "GPU" in compiled_engines:
            current_queue.wait_all()
            current_device = "GPU"
            current_queue = compiled_engines["GPU"][1]
        elif key == ord('n') and "NPU" in compiled_engines:
            current_queue.wait_all()
            current_device = "NPU"
            current_queue = compiled_engines["NPU"][1]
        elif key == ord('c') and "CPU" in compiled_engines:
            current_queue.wait_all()
            current_device = "CPU"
            current_queue = compiled_engines["CPU"][1]

finally:
    if not use_synthetic:
        cap.release()
    cv2.destroyAllWindows()
    print("\nApplication closed cleanly.")