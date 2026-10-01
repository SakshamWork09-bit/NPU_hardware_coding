import openvino as ov
import numpy as np
import time

MODEL = "mobilenetv2-7.onnx"
WARMUP = 10
ITERATIONS = 50

core = ov.Core()

print("=" * 60)
print("OPENVINO AI HARDWARE LAB")
print("=" * 60)
print("OpenVINO:", ov.__version__)
print("Model:", MODEL)
print("Devices:", core.available_devices)
print()

model = core.read_model(MODEL)

input_layer = model.inputs[0]
output_layer = model.outputs[0]

input_name = input_layer.any_name
output_name = output_layer.any_name

print("Input :", input_name, input_layer.shape, input_layer.element_type)
print("Output:", output_name, output_layer.shape, output_layer.element_type)
print()

# Identical input for CPU, GPU and NPU
rng = np.random.default_rng(42)
input_data = rng.random(
    tuple(input_layer.shape),
    dtype=np.float32
)


def benchmark(device):
    print("-" * 60)
    print("DEVICE:", device)

    try:
        # Compile model for selected hardware
        t0 = time.perf_counter()
        compiled = core.compile_model(model, device)
        compile_time = time.perf_counter() - t0

        request = compiled.create_infer_request()

        # Warm-up
        for _ in range(WARMUP):
            request.infer({
                input_name: input_data
            })

        # Timed inference
        times = []
        result = None

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

        throughput = 1.0 / avg

        # IMPORTANT:
        # Use the output NAME rather than the OpenVINO Output object.
        output = result[output_name]

        checksum = float(np.sum(output))

        print(f"Compile time : {compile_time:.4f} s")
        print(f"Average      : {avg * 1000:.4f} ms")
        print(f"Median       : {median * 1000:.4f} ms")
        print(f"Best         : {best * 1000:.4f} ms")
        print(f"Throughput   : {throughput:.2f} inferences/sec")
        print(f"Output shape : {output.shape}")
        print(f"Output sum   : {checksum:.6f}")

        return {
            "device": device,
            "compile": compile_time,
            "avg": avg,
            "median": median,
            "best": best,
            "throughput": throughput,
            "checksum": checksum
        }

    except Exception as e:
        print()
        print("DEVICE FAILED:", device)
        print(type(e).__name__ + ":", e)
        print()
        return None


results = []

for device in ["CPU", "GPU", "NPU"]:
    result = benchmark(device)

    if result is not None:
        results.append(result)


print()
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)

for r in results:
    print(
        f"{r['device']:>4} | "
        f"{r['avg'] * 1000:9.3f} ms | "
        f"{r['throughput']:8.2f} inf/s | "
        f"compile {r['compile']:.3f} s"
    )

print("=" * 60)
print()
print("Same model + same input used on every device.")