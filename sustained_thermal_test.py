import os
import time
import statistics
import numpy as np
import openvino as ov

# ============================================================
# SUSTAINED HARDWARE / THERMAL SCALING TEST
# ============================================================

MODEL_PATH = r"C:\Users\saksh\HardwareLab\mobilenetv2-7.onnx"

DURATION_SEC = 180       # 3 minutes per device
WINDOW_SEC = 10          # report every 10 seconds
WARMUP_RUNS = 100
BATCH = 1

DEVICES = ["CPU", "GPU", "NPU"]

# ------------------------------------------------------------
# Environment information
# ------------------------------------------------------------

print("=" * 70)
print("SUSTAINED HARDWARE / THERMAL TEST")
print("=" * 70)
print(f"Model:       {os.path.basename(MODEL_PATH)}")
print(f"Batch:       {BATCH}")
print(f"Duration:    {DURATION_SEC} seconds/device")
print(f"Window:      {WINDOW_SEC} seconds")
print(f"Warmup:      {WARMUP_RUNS} runs")
print()
print("IMPORTANT:")
print("  AC POWER")
print("  Windows: BEST PERFORMANCE")
print("  Laptop on hard flat surface")
print("  Vents unobstructed")
print("=" * 70)


core = ov.Core()

print("\nAvailable devices:")
print(core.available_devices)


# ------------------------------------------------------------
# Prepare model
# ------------------------------------------------------------

model_original = core.read_model(MODEL_PATH)

input_name = model_original.inputs[0].get_any_name()

# Force static batch=1
try:
    model_original.reshape({
        input_name: [BATCH, 3, 224, 224]
    })
except Exception as e:
    print("\nCould not reshape model:", e)
    raise

# Fixed input
rng = np.random.default_rng(42)

input_data = rng.random(
    (BATCH, 3, 224, 224),
    dtype=np.float32
)

# ------------------------------------------------------------
# Run one device
# ------------------------------------------------------------

all_results = {}


def run_device(device):

    print("\n")
    print("=" * 70)
    print(f"DEVICE: {device}")
    print("=" * 70)

    # Fresh model compilation for every device
    model = core.read_model(MODEL_PATH)

    model.reshape({
        model.inputs[0].get_any_name(): [BATCH, 3, 224, 224]
    })

    print("Compiling model...")

    compiled = core.compile_model(
        model,
        device
    )

    request = compiled.create_infer_request()

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    print(f"Warmup: {WARMUP_RUNS} inference(s)...")

    for _ in range(WARMUP_RUNS):
        request.infer({
            input_name: input_data
        })

    print("Warmup complete.")
    print("Starting sustained workload...")
    print()

    # --------------------------------------------------------
    # Sustained test
    # --------------------------------------------------------

    test_start = time.perf_counter()
    window_start = test_start

    total_images = 0
    total_inference_time = 0.0

    window_images = 0
    window_times = []

    window_results = []

    last_print = 0

    while True:

        now = time.perf_counter()

        if now - test_start >= DURATION_SEC:
            break

        # One inference
        t0 = time.perf_counter()

        request.infer({
            input_name: input_data
        })

        elapsed = time.perf_counter() - t0

        total_images += BATCH
        total_inference_time += elapsed

        window_images += BATCH
        window_times.append(elapsed)

        # ----------------------------------------------------
        # Print 10-second window
        # ----------------------------------------------------

        now = time.perf_counter()

        if now - window_start >= WINDOW_SEC:

            window_elapsed = now - window_start

            throughput = window_images / window_elapsed

            median_ms = (
                statistics.median(window_times) * 1000
                if window_times else 0
            )

            p95_ms = (
                np.percentile(window_times, 95) * 1000
                if window_times else 0
            )

            elapsed_total = now - test_start

            print(
                f"[{elapsed_total:6.1f}s] "
                f"{throughput:8.2f} img/s | "
                f"median {median_ms:7.3f} ms | "
                f"P95 {p95_ms:7.3f} ms | "
                f"window {window_images:5d}"
            )

            window_results.append({
                "time": elapsed_total,
                "throughput": throughput,
                "median_ms": median_ms,
                "p95_ms": p95_ms
            })

            window_start = now
            window_images = 0
            window_times = []

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    total_elapsed = time.perf_counter() - test_start

    overall_throughput = total_images / total_elapsed

    all_window_throughputs = [
        x["throughput"] for x in window_results
    ]

    print()
    print("-" * 70)
    print(f"{device} FINAL")
    print("-" * 70)

    print(f"Total images:       {total_images}")
    print(f"Total time:         {total_elapsed:.3f} s")
    print(f"Average throughput: {overall_throughput:.2f} img/s")

    if all_window_throughputs:

        first_window = all_window_throughputs[0]
        last_window = all_window_throughputs[-1]
        peak_window = max(all_window_throughputs)
        min_window = min(all_window_throughputs)

        degradation = (
            (first_window - last_window)
            / first_window
            * 100
        )

        print(f"First window:       {first_window:.2f} img/s")
        print(f"Last window:        {last_window:.2f} img/s")
        print(f"Peak window:        {peak_window:.2f} img/s")
        print(f"Minimum window:     {min_window:.2f} img/s")
        print(f"First→last change:  {degradation:+.2f}%")

    all_results[device] = {
        "total_images": total_images,
        "total_time": total_elapsed,
        "throughput": overall_throughput,
        "windows": window_results
    }


# ============================================================
# MAIN TEST
# ============================================================

for device in DEVICES:

    try:
        run_device(device)

    except Exception as e:

        print("\n" + "!" * 70)
        print(f"{device} FAILED")
        print("!" * 70)
        print(str(e))


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n")
print("=" * 70)
print("FINAL SUSTAINED THROUGHPUT COMPARISON")
print("=" * 70)

for device, result in all_results.items():

    windows = result["windows"]

    if windows:

        first = windows[0]["throughput"]
        last = windows[-1]["throughput"]

        change = (last - first) / first * 100

        print(
            f"{device:5s} | "
            f"Average: {result['throughput']:8.2f} img/s | "
            f"First: {first:8.2f} | "
            f"Last: {last:8.2f} | "
            f"Change: {change:+7.2f}%"
        )

print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)