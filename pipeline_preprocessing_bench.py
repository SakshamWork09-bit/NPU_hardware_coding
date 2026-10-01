import os
import time
import numpy as np
import openvino as ov
from openvino.preprocess import PrePostProcessor, ColorFormat, ResizeAlgorithm

# ============================================================
# REAL-WORLD PREPROCESSING & PIPELINE BOTTLENECK EXPERIMENT
# ============================================================

MODEL_PATH = r"C:\Users\saksh\HardwareLab\mobilenetv2-7.onnx"
DEVICES = ["GPU", "CPU"]
FRAME_COUNT = 3000

# Simulated realistic camera/video input: 1080p BGR frame (HWC, uint8)
RAW_H, RAW_W = 1080, 1920
MODEL_H, MODEL_W = 224, 224

print("=" * 80)
print("REAL-WORLD PREPROCESSING & PIPELINE BOTTLENECK EXPERIMENT")
print("=" * 80)
print(f"Model:                {os.path.basename(MODEL_PATH)}")
print(f"Input Stream Format:  1080p ({RAW_W}x{RAW_H}), BGR, uint8 [Standard Camera]")
print(f"Target Model Format:  224x224, RGB, float32, CHW")
print(f"Evaluation Frames:    {FRAME_COUNT} per configuration")
print("=" * 80)

# Generate a mock 1080p frame buffer once in memory
rng = np.random.default_rng(42)
raw_frame = rng.integers(0, 256, (RAW_H, RAW_W, 3), dtype=np.uint8)

core = ov.Core()

# ------------------------------------------------------------
# Manual Python Preprocessing Simulation
# ------------------------------------------------------------
def manual_preprocess_cpu(frame):
    # 1. Spatial Resize (fast downsampling)
    row_idx = np.linspace(0, RAW_H - 1, MODEL_H).astype(int)
    col_idx = np.linspace(0, RAW_W - 1, MODEL_W).astype(int)
    resized = frame[row_idx[:, None], col_idx]

    # 2. Color Conversion: BGR -> RGB
    rgb = resized[:, :, ::-1]

    # 3. Transpose: HWC -> CHW & Add Batch Dimension
    chw = np.transpose(rgb, (2, 0, 1))
    batched = np.expand_dims(chw, axis=0)

    # 4. Normalize: uint8 [0, 255] -> float32 [0.0, 1.0]
    tensor = batched.astype(np.float32) / 255.0
    return tensor

# ------------------------------------------------------------
# Benchmark Pipeline A: Manual Preprocessing + Plain Model
# ------------------------------------------------------------
def benchmark_pipeline_a(device):
    print(f"\n--- [Pipeline A: Manual Python CPU Preprocessing + {device} Model] ---")
    
    # Measure pure preprocessing cost
    t_pre_start = time.perf_counter()
    for _ in range(200):
        _ = manual_preprocess_cpu(raw_frame)
    avg_pre_time_ms = ((time.perf_counter() - t_pre_start) / 200.0) * 1000.0
    max_pre_fps = 1000.0 / avg_pre_time_ms

    print(f"Pure Preprocessing Latency (CPU) : {avg_pre_time_ms:6.3f} ms/frame")
    print(f"Maximum Possible Pipeline Ceiling: {max_pre_fps:6.1f} FPS (preproc limited)")

    # Compile vanilla model
    model = core.read_model(MODEL_PATH)
    compiled = core.compile_model(model, device)
    queue = ov.AsyncInferQueue(compiled, jobs=2)

    # Benchmark full end-to-end pipeline
    start = time.perf_counter()
    for _ in range(FRAME_COUNT):
        processed_input = manual_preprocess_cpu(raw_frame)
        queue.start_async({compiled.input(0).any_name: processed_input})
    queue.wait_all()
    total_elapsed = time.perf_counter() - start

    fps = FRAME_COUNT / total_elapsed
    latency_ms = (total_elapsed / FRAME_COUNT) * 1000.0

    print(f"End-to-End Pipeline Latency      : {latency_ms:6.3f} ms/frame")
    print(f"End-to-End Effective Throughput  : {fps:6.1f} FPS")
    return {"pre_ms": avg_pre_time_ms, "e2e_fps": fps, "e2e_ms": latency_ms}

# ------------------------------------------------------------
# Benchmark Pipeline B: OpenVINO Integrated Hardware Preprocessing
# ------------------------------------------------------------
def benchmark_pipeline_b(device):
    print(f"\n--- [Pipeline B: OpenVINO Hardware Preprocessing + {device} Model] ---")
    
    model = core.read_model(MODEL_PATH)
    ppp = PrePostProcessor(model)

    # 1. Set static 1080p input tensor shape and properties:
    # Source: [1, 1080, 1920, 3], uint8, BGR
    ppp.input().tensor() \
        .set_shape([1, RAW_H, RAW_W, 3]) \
        .set_element_type(ov.Type.u8) \
        .set_color_format(ColorFormat.BGR) \
        .set_layout(ov.Layout("NHWC"))

    # 2. Tell OpenVINO the original neural network expects NCHW
    ppp.input().model().set_layout(ov.Layout("NCHW"))

    # 3. Embed pre-processing steps into the accelerator execution graph:
    ppp.input().preprocess() \
        .convert_element_type(ov.Type.f32) \
        .convert_color(ColorFormat.RGB) \
        .resize(ResizeAlgorithm.RESIZE_LINEAR, MODEL_H, MODEL_W) \
        .scale([255.0, 255.0, 255.0])

    model = ppp.build()

    compiled = core.compile_model(model, device)
    queue = ov.AsyncInferQueue(compiled, jobs=2)

    # Prepare raw frame with batch dimension: [1, 1080, 1920, 3] uint8
    raw_batched = np.expand_dims(raw_frame, axis=0)

    # Warmup
    for _ in range(50):
        queue.start_async({compiled.input(0).any_name: raw_batched})
    queue.wait_all()

    # Benchmark full end-to-end pipeline (passing raw 1080p frame directly)
    start = time.perf_counter()
    for _ in range(FRAME_COUNT):
        queue.start_async({compiled.input(0).any_name: raw_batched})
    queue.wait_all()
    total_elapsed = time.perf_counter() - start

    fps = FRAME_COUNT / total_elapsed
    latency_ms = (total_elapsed / FRAME_COUNT) * 1000.0

    print(f"End-to-End Pipeline Latency      : {latency_ms:6.3f} ms/frame")
    print(f"End-to-End Effective Throughput  : {fps:6.1f} FPS")
    return {"e2e_fps": fps, "e2e_ms": latency_ms}

# ------------------------------------------------------------
# Execution & Summary Comparison
# ------------------------------------------------------------
results = {}

for dev in DEVICES:
    print(f"\n{'#' * 80}")
    print(f"TESTING ACCELERATOR: {dev}")
    print(f"{'#' * 80}")
    res_a = benchmark_pipeline_a(dev)
    res_b = benchmark_pipeline_b(dev)
    results[dev] = {"Pipeline_A": res_a, "Pipeline_B": res_b}

print("\n" + "=" * 80)
print("FINAL PIPELINE ARCHITECTURE COMPARISON (1080p Stream -> MobileNetV2)")
print("=" * 80)
print(f"{'Device':<8} | {'Pipeline A (Manual CPU)':<24} | {'Pipeline B (OV Integrated)':<26} | {'Speedup':<10}")
print("-" * 80)

for dev, data in results.items():
    fps_a = data["Pipeline_A"]["e2e_fps"]
    fps_b = data["Pipeline_B"]["e2e_fps"]
    speedup = fps_b / fps_a if fps_a > 0 else 0
    print(f"{dev:<8} | {fps_a:7.1f} FPS ({data['Pipeline_A']['e2e_ms']:5.2f} ms)   | "
          f"{fps_b:7.1f} FPS ({data['Pipeline_B']['e2e_ms']:5.2f} ms)     | "
          f"{speedup:5.2f}x")

print("=" * 80)
print("EXPERIMENT COMPLETE")