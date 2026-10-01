import openvino as ov
import numpy as np
import time
import statistics


MODEL_PATH = r"C:\Users\saksh\HardwareLab\mobilenetv2-7.onnx"

JOBS = 5000

INPUT = np.random.rand(
    1, 3, 224, 224
).astype(np.float32)


def run_device(core, model, device, jobs):

    print(f"\n{'=' * 60}")
    print(f"DEVICE: {device}")
    print(f"{'=' * 60}")

    compiled = core.compile_model(
        model,
        device
    )

    request = compiled.create_infer_request()

    # Warmup
    for _ in range(20):
        request.infer({0: INPUT})

    latencies = []

    start = time.perf_counter()

    for _ in range(jobs):

        t0 = time.perf_counter()

        request.infer({0: INPUT})

        t1 = time.perf_counter()

        latencies.append(
            (t1 - t0) * 1000
        )

    elapsed = time.perf_counter() - start

    throughput = jobs / elapsed

    print(f"Jobs:       {jobs}")
    print(f"Time:       {elapsed:.3f} s")
    print(f"Throughput: {throughput:.2f} img/s")

    print(
        f"Median:     "
        f"{statistics.median(latencies):.3f} ms"
    )

    print(
        f"P95:        "
        f"{sorted(latencies)[int(len(latencies) * .95)]:.3f} ms"
    )

    return throughput


def main():

    core = ov.Core()

    print("Available devices:")
    print(core.available_devices)

    model = core.read_model(MODEL_PATH)

    print("\nRunning isolated tests...")

    gpu = run_device(
        core,
        model,
        "GPU",
        JOBS
    )

    npu = run_device(
        core,
        model,
        "NPU",
        JOBS
    )

    print("\n")
    print("=" * 60)
    print("ISOLATED DEVICE RESULT")
    print("=" * 60)

    print(
        f"\nGPU: {gpu:.2f} img/s"
    )

    print(
        f"NPU: {npu:.2f} img/s"
    )


if __name__ == "__main__":
    main()