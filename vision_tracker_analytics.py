import os
import time
import queue
import threading
from collections import deque
import cv2
import numpy as np
import openvino as ov
from scipy.optimize import linear_sum_assignment

# ============================================================
# HETEROGENEOUS MULTI-OBJECT TRACKING & ZONE ANALYTICS ENGINE
# ============================================================

MODEL_XML = r"C:\Users\saksh\HardwareLab\yolov8n_openvino_model\yolov8n.xml"
MODEL_ONNX = r"C:\Users\saksh\HardwareLab\yolov8n.onnx"

if os.path.exists(MODEL_XML):
    MODEL_PATH = MODEL_XML
    MODEL_TYPE = "INT8 OpenVINO IR"
elif os.path.exists(MODEL_ONNX):
    MODEL_PATH = MODEL_ONNX
    MODEL_TYPE = "FP32/FP16 ONNX"
else:
    raise FileNotFoundError("YOLOv8 model not found. Ensure yolov8n.onnx or yolov8n_openvino_model exists.")

INPUT_W, INPUT_H = 640, 640
CONF_THRESHOLD = 0.40
IOU_THRESHOLD = 0.45

# Heterogeneous worker allocation
WORKER_CONFIG = {
    "GPU": 1,
    "NPU": 2
}

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

core = ov.Core()

print("=" * 85)
print("HETEROGENEOUS MULTI-OBJECT TRACKING & ANALYTICS ENGINE")
print("=" * 85)
print(f"Model       : {MODEL_PATH} ({MODEL_TYPE})")
print(f"Workers     : {WORKER_CONFIG}")
print("Controls    : Press 'q' in display to quit | Press 'r' to reset analytics counters")
print("=" * 85)

compiled_models = {}
for dev in WORKER_CONFIG:
    if dev in core.available_devices:
        print(f"Compiling YOLOv8 for {dev}...")
        m = core.read_model(MODEL_PATH)
        compiled_models[dev] = core.compile_model(m, dev)
    else:
        raise RuntimeError(f"Device {dev} not available.")

# ------------------------------------------------------------
# High-Performance Spatial Centroid & IoU Multi-Object Tracker
# ------------------------------------------------------------
def calculate_iou_matrix(boxes_a, boxes_b):
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))

    b_a = np.array(boxes_a)  # [N, 4]: x, y, w, h
    b_b = np.array(boxes_b)  # [M, 4]: x, y, w, h

    x1_a, y1_a, x2_a, y2_a = b_a[:, 0], b_a[:, 1], b_a[:, 0] + b_a[:, 2], b_a[:, 1] + b_a[:, 3]
    x1_b, y1_b, x2_b, y2_b = b_b[:, 0], b_b[:, 1], b_b[:, 0] + b_b[:, 2], b_b[:, 1] + b_b[:, 3]

    inter_x1 = np.maximum(x1_a[:, None], x1_b[None, :])
    inter_y1 = np.maximum(y1_a[:, None], y1_b[None, :])
    inter_x2 = np.minimum(x2_a[:, None], x2_b[None, :])
    inter_y2 = np.minimum(y2_a[:, None], y2_b[None, :])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)
    union = area_a[:, None] + area_b[None, :] - intersection

    return np.where(union > 0, intersection / union, 0.0)

class TrackedObject:
    def __init__(self, track_id, box, class_id, score):
        self.track_id = track_id
        self.box = box  # [x, y, w, h]
        self.class_id = class_id
        self.score = score
        self.hits = 1
        self.lost = 0
        self.centroid_history = deque(maxlen=25)
        cx = int(box[0] + box[2] / 2)
        cy = int(box[1] + box[3] / 2)
        self.centroid_history.append((cx, cy))
        self.first_seen = time.perf_counter()
        self.zone_entry_time = None

    def update(self, box, score):
        self.box = box
        self.score = score
        self.hits += 1
        self.lost = 0
        cx = int(box[0] + box[2] / 2)
        cy = int(box[1] + box[3] / 2)
        self.centroid_history.append((cx, cy))

class MultiObjectTracker:
    def __init__(self, max_lost=15, iou_thresh=0.25):
        self.next_id = 1
        self.tracks = {}
        self.max_lost = max_lost
        self.iou_thresh = iou_thresh

    def update(self, detections):
        # detections: list of ([x,y,w,h], score, class_id)
        det_boxes = [d[0] for d in detections]
        track_ids = list(self.tracks.keys())
        trk_boxes = [self.tracks[tid].box for tid in track_ids]

        # Cost matrix based on 1.0 - IoU
        iou_mat = calculate_iou_matrix(trk_boxes, det_boxes)
        cost_mat = 1.0 - iou_mat

        matched_tracks = set()
        matched_dets = set()

        if len(trk_boxes) > 0 and len(det_boxes) > 0:
            row_ind, col_ind = linear_sum_assignment(cost_mat)
            for r, c in zip(row_ind, col_ind):
                if iou_mat[r, c] >= self.iou_thresh:
                    tid = track_ids[r]
                    self.tracks[tid].update(det_boxes[c], detections[c][1])
                    matched_tracks.add(tid)
                    matched_dets.add(c)

        # Increment lost count on unmatched tracks
        for tid in track_ids:
            if tid not in matched_tracks:
                self.tracks[tid].lost += 1

        # Remove dead tracks
        dead_ids = [tid for tid, trk in self.tracks.items() if trk.lost > self.max_lost]
        for tid in dead_ids:
            del self.tracks[tid]

        # Register new detections as new tracks
        for i, det in enumerate(detections):
            if i not in matched_dets:
                new_track = TrackedObject(self.next_id, det[0], det[2], det[1])
                self.tracks[self.next_id] = new_track
                self.next_id += 1

        return self.tracks

tracker = MultiObjectTracker()

# ------------------------------------------------------------
# Analytics: Virtual Polygon Zones & Tripwires
# ------------------------------------------------------------
class AnalyticsEngine:
    def __init__(self):
        # Zone A: Top-Right "Priority Alert / Dwell Zone" (Polygon in normalized coords)
        self.zone_polygon = np.array([[480, 50], [900, 50], [900, 360], [480, 360]], np.int32)
        # Tripwire: Horizontal Line across middle of screen (Y = 320)
        self.tripwire_y = 300
        self.tripwire_crossings = 0
        self.zone_dwell_records = {}  # track_id -> total_seconds

    def point_in_zone(self, cx, cy):
        return cv2.pointPolygonTest(self.zone_polygon, (float(cx), float(cy)), False) >= 0

    def update(self, tracks):
        current_zone_occupants = 0
        now = time.perf_counter()

        for tid, trk in tracks.items():
            if len(trk.centroid_history) < 2:
                continue

            prev_cx, prev_cy = trk.centroid_history[-2]
            curr_cx, curr_cy = trk.centroid_history[-1]

            # Tripwire line-crossing check (crossed downwards)
            if prev_cy < self.tripwire_y <= curr_cy:
                self.tripwire_crossings += 1

            # Zone Dwell Analytics
            if self.point_in_zone(curr_cx, curr_cy):
                current_zone_occupants += 1
                if trk.zone_entry_time is None:
                    trk.zone_entry_time = now
                self.zone_dwell_records[tid] = now - trk.zone_entry_time
            else:
                trk.zone_entry_time = None

        return current_zone_occupants, self.tripwire_crossings

analytics = AnalyticsEngine()

# ------------------------------------------------------------
# Inference & Vectorized Postprocessing
# ------------------------------------------------------------
task_queue = queue.Queue(maxsize=16)
stop_event = threading.Event()
device_counts = {"GPU": 0, "NPU": 0}
counter_lock = threading.Lock()
latest_detections = []
det_lock = threading.Lock()

def decode_and_nms(raw_output, scale, dx, dy):
    out = raw_output.T
    classes_scores = out[:, 4:]
    max_scores = np.max(classes_scores, axis=1)
    mask = max_scores >= CONF_THRESHOLD

    if not np.any(mask):
        return []

    filtered_out = out[mask]
    filtered_scores = max_scores[mask]
    class_ids = np.argmax(filtered_out[:, 4:], axis=1)

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
    class_ids = class_ids.tolist()

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, IOU_THRESHOLD)

    detected = []
    if len(indices) > 0:
        for idx in indices.flatten():
            detected.append((boxes[idx], confidences[idx], class_ids[idx]))
    return detected

def engine_worker(device_name, compiled_model):
    infer_request = compiled_model.create_infer_request()
    while not stop_event.is_set():
        try:
            item = task_queue.get(timeout=0.01)
        except queue.Empty:
            continue

        tensor, scale, dx, dy = item
        infer_request.infer({0: tensor})
        raw_output = infer_request.get_output_tensor(0).data[0]
        detected = decode_and_nms(raw_output, scale, dx, dy)

        with counter_lock:
            device_counts[device_name] += 1

        with det_lock:
            global latest_detections
            latest_detections = detected

        task_queue.task_done()

# Start workers
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
# Camera Capture
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
                # Moving simulated object
                cx = int((t * 260) % 1280)
                cy = int(240 + 120 * np.sin(t * 2))
                cv2.circle(f, (cx, cy), 50, (0, 255, 0), -1)
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
    scale = min(INPUT_W / w, INPUT_H / h)
    nw, nh = int(w * scale), int(h * scale)

    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((INPUT_H, INPUT_W, 3), dtype=np.uint8)
    dx = (INPUT_W - nw) // 2
    dy = (INPUT_H - nh) // 2
    canvas[dy:dy+nh, dx:dx+nw] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return np.expand_dims(tensor, axis=0), scale, dx, dy

# ------------------------------------------------------------
# Main Dispatcher & Display Loop
# ------------------------------------------------------------
fps_timer = time.perf_counter()
display_timer = time.perf_counter()
prev_gpu, prev_npu = 0, 0
total_fps = 0.0
gpu_fps, npu_fps = 0.0, 0.0

print("\nStarting Object Tracking & Zone Analytics System...\n")

try:
    while True:
        frame = cam.read_latest()
        if frame is None:
            time.sleep(0.001)
            continue

        tensor, scale, dx, dy = preprocess(frame)
        try:
            task_queue.put((tensor, scale, dx, dy), block=False)
        except queue.Full:
            pass

        now = time.perf_counter()

        # Telemetry throughput output every second
        if now - fps_timer >= 1.0:
            elapsed = now - fps_timer
            with counter_lock:
                c_gpu, c_npu = device_counts["GPU"], device_counts["NPU"]

            gpu_fps = (c_gpu - prev_gpu) / elapsed
            npu_fps = (c_npu - prev_npu) / elapsed
            total_fps = gpu_fps + npu_fps
            prev_gpu, prev_npu = c_gpu, c_npu
            fps_timer = now

            print(f"[TRACKER] Total: {total_fps:6.1f} FPS | "
                  f"GPU: {gpu_fps:5.1f} | NPU: {npu_fps:5.1f} FPS | "
                  f"Active Tracks: {len(tracker.tracks)} | Crossings: {analytics.tripwire_crossings}")

        # Render Analytics Display at 30 Hz
        if now - display_timer >= 0.033:
            display_frame = frame.copy()
            h_disp, w_disp = display_frame.shape[:2]

            # Update tracker with latest detections
            with det_lock:
                dets = list(latest_detections)
            active_tracks = tracker.update(dets)
            zone_occupants, crossings = analytics.update(active_tracks)

            # 1. Draw Virtual Zone Polygon
            zone_overlay = display_frame.copy()
            zone_color = (0, 0, 255) if zone_occupants > 0 else (255, 180, 0)
            cv2.polylines(display_frame, [analytics.zone_polygon], True, zone_color, 2)
            cv2.fillPoly(zone_overlay, [analytics.zone_polygon], zone_color)
            cv2.addWeighted(zone_overlay, 0.20, display_frame, 0.80, 0, display_frame)
            cv2.putText(display_frame, f"ZONE A (Occupants: {zone_occupants})",
                        (analytics.zone_polygon[0][0] + 10, analytics.zone_polygon[0][1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, zone_color, 2)

            # 2. Draw Tripwire Line
            cv2.line(display_frame, (0, analytics.tripwire_y), (w_disp, analytics.tripwire_y), (0, 255, 255), 2)
            cv2.putText(display_frame, f"TRIPWIRE (Crossings: {crossings})", (20, analytics.tripwire_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            # 3. Draw Tracked Objects & Motion Trails
            for tid, trk in active_tracks.items():
                bx, by, bw, bh = trk.box
                cls_name = COCO_CLASSES[trk.class_id] if trk.class_id < len(COCO_CLASSES) else f"ID {trk.class_id}"
                dwell_sec = analytics.zone_dwell_records.get(tid, 0.0)
                dwell_str = f" | Dwell: {dwell_sec:.1f}s" if dwell_sec > 0 else ""
                label = f"#{tid} {cls_name} ({trk.score:.2f}){dwell_str}"

                # Bounding box
                box_color = (0, 255, 0) if trk.lost == 0 else (120, 120, 120)
                cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), box_color, 2)
                cv2.putText(display_frame, label, (bx, max(22, by - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

                # Motion trajectory tail
                pts = list(trk.centroid_history)
                for i in range(1, len(pts)):
                    alpha = i / len(pts)
                    thickness = int(1 + 3 * alpha)
                    cv2.line(display_frame, pts[i - 1], pts[i], (0, 255, 255), thickness)

            # 4. HUD Telemetry Box
            cv2.rectangle(display_frame, (10, 10), (450, 125), (15, 15, 15), -1)
            cv2.putText(display_frame, "ANALYTICS: GPU + NPU PARALLEL", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(display_frame, f"AI SPEED    : {total_fps:5.1f} FPS (GPU: {gpu_fps:.0f} | NPU: {npu_fps:.0f})",
                        (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2)
            cv2.putText(display_frame, f"TRACKS      : {len(active_tracks)} active | Total: {tracker.next_id - 1}",
                        (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
            cv2.putText(display_frame, f"CROSSINGS   : {crossings} | Zone Occupancy: {zone_occupants}",
                        (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

            cv2.imshow("Multi-Object Tracking & Zone Analytics Engine", display_frame)
            display_timer = now

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                analytics.tripwire_crossings = 0
                analytics.zone_dwell_records.clear()
                print("\n>>> Reset Analytics Counters <<<\n")

finally:
    stop_event.set()
    cam.stop()
    cv2.destroyAllWindows()
    print("Tracking and Analytics engine stopped cleanly.")