import time
import threading
import numpy as np
from openvino import Core

MODEL = "mobilenetv2-7.onnx"
DURATION = 15

core = Core()

print("OpenVINO:", core.get_property("CPU", "FULL_DEVICE_NAME"))
print("Available:", core.available_devices)
print()

# Same batch-1 input for every accelerator.
rng = np.random.default_rng(42)
INPUT = rng.random((1, 3, 224, 224), dtype=np.float32)


def prepare(device):
    print(f"Compiling {device}...")
    model = core.read_model(MODEL)
    compiled = core.compile_model(model, device)

    input_port = compiled.input(0)

    # One independent InferRequest per worker.
    request = compiled.create_infer_request()
    request.infer({input_port.any_name: INPUT})

    return compiled, input_port


def worker(compiled, input_port, stop_event, result, index):
    request = compiled.create_infer_request()

    count = 0
    start = time.perf_counter()

    while not stop_event.is_set():
        request.infer({input_port.any_name: INPUT})
        count += 1

    elapsed = time.perf_counter() - start
    result[index] = (count, elapsed)


def run_case(name, configs, compiled_devices):
    """
    configs example:
        {"CPU": 4, "GPU": 1, "NPU": 2}
    """

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    stop_event = threading.Event()
    threads = []
    results = {}

    for device, workers in configs.items():
        compiled, input_port = compiled_devices[device]

        for i in range(workers):
            index = f"{device}-{i}"

            t = threading.Thread(
                target=worker,
                args=(compiled, input_port, stop_event, results, index)
            )

            threads.append(t)

    start = time.perf_counter()

    for t in threads:
        t.start()

    time.sleep(DURATION)

    stop_event.set()

    for t in threads:
        t.join()

    total_elapsed = time.perf_counter() - start

    device_totals = {}

    for device in configs:
        device_totals[device] = 0

        for i in range(configs[device]):
            count, elapsed = results[f"{device}-{i}"]
            device_totals[device] += count

    total_images = sum(device_totals.values())
    total_throughput = total_images / total_elapsed

    for device in configs:
        count = device_totals[device]
        throughput = count / total_elapsed

        print(
            f"{device:4s} | "
            f"workers: {configs[device]:2d} | "
            f"images: {count:7d} | "
            f"throughput: {throughput:8.2f} images/s"
        )

    print("-" * 70)
    print(f"TOTAL | images: {total_images}")
    print(f"TOTAL | throughput: {total_throughput:.2f} images/s")
    print(f"Elapsed: {total_elapsed:.3f} s")

    return total_throughput


def main():

    # Based on our previous saturation experiment:
    #
    # CPU  -> around 4 concurrent requests
    # GPU  -> already nearly saturated at 1
    # NPU  -> around 2 concurrent requests
    #
    # These are starting points for THIS workload, not hardware limits.

    compiled_devices = {}

    for device in ["CPU", "GPU", "NPU"]:
        compiled_devices[device] = prepare(device)

    print("\nAll three devices compiled successfully.")

    results = {}

    # Individual baselines
    results["CPU"] = run_case(
        "CPU ONLY",
        {"CPU": 4},
        compiled_devices
    )

    results["GPU"] = run_case(
        "GPU ONLY",
        {"GPU": 1},
        compiled_devices
    )

    results["NPU"] = run_case(
        "NPU ONLY",
        {"NPU": 2},
        compiled_devices
    )

    # Pair combinations
    results["CPU+GPU"] = run_case(
        "CPU + GPU",
        {"CPU": 4, "GPU": 1},
        compiled_devices
    )

    results["CPU+NPU"] = run_case(
        "CPU + NPU",
        {"CPU": 4, "NPU": 2},
        compiled_devices
    )

    results["GPU+NPU"] = run_case(
        "GPU + NPU",
        {"GPU": 1, "NPU": 2},
        compiled_devices
    )

    # Everything simultaneously
    results["ALL"] = run_case(
        "CPU + GPU + NPU",
        {"CPU": 4, "GPU": 1, "NPU": 2},
        compiled_devices
    )

    print("\n\n" + "=" * 70)
    print("FINAL HETEROGENEOUS RESULTS")
    print("=" * 70)

    for name, throughput in results.items():
        print(f"{name:15s} : {throughput:10.2f} images/s")

    print("=" * 70)


if __name__ == "__main__":
    main()