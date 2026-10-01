import time
import cv2
import numpy as np
import openvino as ov

MODEL_ONNX = r"C:\Users\saksh\HardwareLab\yolov8n.onnx"

print("=" * 70)
print("NAIVE (UNOPTIMIZED) OBJECT DETECTION PIPELINE")
print("=" * 70)
print("Running single-threaded synchronous inference on CPU only...")
print("Press 'q' in the window to quit.")
print("=" * 70)

core = ov.Core()
model = core.read_model(MODEL_ONNX)
# Naive default: compiled for CPU without asynchronous queues or workers
compiled = core.compile_model(model, "CPU")
infer_request = compiled.create_infer_request()

cap = cv2.VideoCapture(0)
use_synthetic = not cap.isOpened()

fps_timer = time.perf_counter()
frames = 0
current_fps = 0.0

try:
    while True:
        t0 = time.perf_counter()

        # 1. Synchronous camera read (blocks on physical sensor)
        if not use_synthetic:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # 2. Slow naive preprocessing (resizing full frame without L3 optimization)
        resized = cv2.resize(frame, (640, 640))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=0)

        # 3. Synchronous blocking inference on CPU
        infer_request.infer({0: tensor})
        out = infer_request.get_output_tensor(0).data[0]

        # 4. Naive scalar Python loop for parsing 8,400 anchors
        out = out.T
        boxes = []
        confidences = []
        for row in out:
            score = float(np.max(row[4:]))
            if score >= 0.40:
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                boxes.append([int(cx - w/2), int(cy - h/2), int(w), int(h)])
                confidences.append(score)

        _ = cv2.dnn.NMSBoxes(boxes, confidences, 0.40, 0.45)

        frames += 1
        t_now = time.perf_counter()
        if t_now - fps_timer >= 1.0:
            current_fps = frames / (t_now - fps_timer)
            print(f"[NAIVE CPU] Speed: {current_fps:5.1f} FPS | Frame Latency: {1000.0/current_fps:5.1f} ms")
            frames = 0
            fps_timer = t_now

        # 5. Synchronous display rendering and OS timer lock
        cv2.putText(frame, f"NAIVE CPU: {current_fps:4.1f} FPS", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imshow("Naive Unoptimized Baseline", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()