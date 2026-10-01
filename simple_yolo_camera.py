import cv2
import numpy as np
import openvino as ov
import time
import os

# Paths to your exported YOLO model
MODEL_XML = r"C:\Users\saksh\HardwareLab\yolov8n_openvino_model\yolov8n.xml"
MODEL_ONNX = r"C:\Users\saksh\HardwareLab\yolov8n.onnx"

if os.path.exists(MODEL_XML):
    MODEL_PATH = MODEL_XML
elif os.path.exists(MODEL_ONNX):
    MODEL_PATH = MODEL_ONNX
else:
    raise FileNotFoundError("YOLO model not found. Ensure yolov8n.onnx or yolov8n_openvino_model exists.")

# Model input parameters
INPUT_W, INPUT_H = 640, 640
CONF_THRESH = 0.40
IOU_THRESH = 0.45

# COCO 80 classes
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

# 1. Initialize OpenVINO on GPU (or change to "NPU" or "CPU")
core = ov.Core()
print(f"Loading {MODEL_PATH} onto GPU...")
model = core.read_model(MODEL_PATH)
compiled_model = core.compile_model(model, "GPU")
infer_request = compiled_model.create_infer_request()

# 2. Open standard webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("\nCamera running! Press 'q' in the window to exit.\n")

fps = 0.0
fps_timer = time.perf_counter()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig_h, orig_w = frame.shape[:2]

    # Preprocessing: Letterbox resize to 640x640
    scale = min(INPUT_W / orig_w, INPUT_H / orig_h)
    nw, nh = int(orig_w * scale), int(orig_h * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((INPUT_H, INPUT_W, 3), dtype=np.uint8)
    dx = (INPUT_W - nw) // 2
    dy = (INPUT_H - nh) // 2
    canvas[dy:dy+nh, dx:dx+nw] = resized

    # Normalization (0-255 -> 0.0-1.0) and transpose to NCHW
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    tensor = np.expand_dims(tensor, axis=0)

    # 3. Synchronous Model Inference
    infer_request.infer({0: tensor})
    out = infer_request.get_output_tensor(0).data[0]  # shape: [84, 8400]

    # 4. Box Decoding and NMS Filtering
    out = out.T  # shape: [8400, 84]
    classes_scores = out[:, 4:]
    max_scores = np.max(classes_scores, axis=1)
    mask = max_scores >= CONF_THRESH

    if np.any(mask):
        filtered_out = out[mask]
        filtered_scores = max_scores[mask]
        class_ids = np.argmax(filtered_out[:, 4:], axis=1)

        boxes_raw = filtered_out[:, :4]
        cx, cy, bw, bh = boxes_raw[:, 0], boxes_raw[:, 1], boxes_raw[:, 2], boxes_raw[:, 3]

        # Rescale boxes back to original camera resolution
        x1 = np.round(((cx - bw / 2.0) - dx) / scale).astype(int)
        y1 = np.round(((cy - bh / 2.0) - dy) / scale).astype(int)
        w_box = np.round(bw / scale).astype(int)
        h_box = np.round(bh / scale).astype(int)

        boxes = np.stack([x1, y1, w_box, h_box], axis=1).tolist()
        confidences = filtered_scores.tolist()
        class_ids = class_ids.tolist()

        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESH, IOU_THRESH)

        # 5. Draw Bounding Boxes on Original Frame
        if len(indices) > 0:
            for idx in indices.flatten():
                bx, by, bw, bh = boxes[idx]
                label = f"{COCO_CLASSES[class_ids[idx]]}: {confidences[idx]:.2f}"
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                cv2.putText(frame, label, (bx, max(25, by - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Calculate real-time FPS
    frame_count += 1
    now = time.perf_counter()
    if now - fps_timer >= 1.0:
        fps = frame_count / (now - fps_timer)
        frame_count = 0
        fps_timer = now

    cv2.putText(frame, f"FPS: {fps:.1f} (GPU)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    # 6. Show the frame
    cv2.imshow("Simple YOLO Camera Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()