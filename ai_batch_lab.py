import openvino as ov
import numpy as np
import time

MODEL = "mobilenetv2-7.onnx"

BATCHES = [1, 2, 4, 8]
WARMUP = 10
ITERATIONS = 30

core = ov.Core()

print("=" * 75)
print("OPENVINO CPU / GPU / NPU BATCH SCALING LAB")
print("=" * 75)
print("OpenVINO:", ov.__version__)
print("Devices:", core.available_devices)
print()

for batch in BATCHES:

    print()
    print("#" * 75)
    print(f"BATCH SIZE: {batch}")
    print("#" * 75)

    # Read a fresh model for each batch size.
    model = core.read_model(MODEL)

    input_layer = model.inputs[0]
    input_name = input_layer.any_name

    # Change only the batch dimension.
    model.reshape({
        input_name: [batch, 3, 224, 224]
    })

    # Identical input generation for all devices.
    rng = np.random.default_rng(42)
    input_data = rng.random(
        (batch, 3, 224, 224),
        dtype=np.float32
    )

    for device in ["CPU", "GPU", "NPU"]:

        print()
        print("-" * 75)
        print("DEVICE:", device)

        try:
            # Compile
            t0 = time.perf_counter()
            compiled = core.compile_model(model, device)
            compile_time = time.perf_counter() - t0

            request = compiled.create_infer_request()

            # Warm-up
            for _ in range(WARMUP):
                request.infer({
                    input_name: input_data
                })

            # Benchmark
            times = []

            for _ in range(ITERATIONS):

                t0 = time.perf_counter()

                result = request.infer({
                    input_name: input_data
                })

                t1 = time.perf_counter()

                times.append(t1 - t0)

            avg = float(np.mean(times))
            median = float(np.median(times))
            best = float(np.min(times))

            images_per_second = batch / avg
            ms_per_image = (avg * 1000.0) / batch

            # Verify output exists.
            output = next(iter(result.values()))

            checksum = float(np.sum(output))

            print(f"Compile time       : {compile_time:.4f} s")
            print(f"Batch latency      : {avg * 1000:.4f} ms")
            print(f"Median latency     : {median * 1000:.4f} ms")
            print(f"Best latency       : {best * 1000:.4f} ms")
            print(f"Per-image latency  : {ms_per_image:.4f} ms")
            print(f"Throughput         : {images_per_second:.2f} images/sec")
            print(f"Output shape       : {output.shape}")
            print(f"Checksum           : {checksum:.6f}")

        except Exception as e:

            print()
            print("DEVICE FAILED")
            print(type(e).__name__ + ":", e)

print()
print("=" * 75)
print("BATCH SCALING TEST COMPLETE")
print("=" * 75)