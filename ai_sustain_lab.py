import time
import threading
import numpy as np
from openvino import Core

MODEL = "mobilenetv2-7.onnx"

DURATION = 30.0
WARMUP = 10

CONCURRENCY_LEVELS = [1, 2, 4, 8]

core = Core()

print("=" * 80)
print("OPENVINO CONCURRENT INFERENCE LAB")
print("=" * 80)
print("Model:", MODEL)
print("Devices:", core.available_devices)
print(f"Duration per test: {DURATION:.0f} seconds")
print("Concurrency levels:", CONCURRENCY_LEVELS)
print()


def worker(compiled_model, input_data, start_event, stop_event, counter):
    request = compiled_model.create_infer_request()
    input_port = compiled_model.input(0)

    # Wait until all workers are ready.
    start_event.wait()

    local_count = 0

    while not stop_event.is_set():
        request.infer({
            input_port: input_data
        })
        local_count += 1

    counter.append(local_count)


def run_test(device, concurrency):

    print("-" * 80)
    print(f"DEVICE: {device} | CONCURRENCY: {concurrency}")

    model = core.read_model(MODEL)

    compile_start = time.perf_counter()
    compiled = core.compile_model(model, device)
    compile_time = time.perf_counter() - compile_start

    input_data = np.random.rand(
        1, 3, 224, 224
    ).astype(np.float32)

    input_port = compiled.input(0)

    # Warm up using one request.
    warmup_request = compiled.create_infer_request()

    for _ in range(WARMUP):
        warmup_request.infer({
            input_port: input_data
        })

    del warmup_request

    workers = []
    counters = []

    start_event = threading.Event()
    stop_event = threading.Event()

    for _ in range(concurrency):

        counter = []
        counters.append(counter)

        thread = threading.Thread(
            target=worker,
            args=(
                compiled,
                input_data,
                start_event,
                stop_event,
                counter
            )
        )

        workers.append(thread)

    for thread in workers:
        thread.start()

    # Start everyone simultaneously.
    start_event.set()

    start_time = time.perf_counter()

    time.sleep(DURATION)

    stop_event.set()

    for thread in workers:
        thread.join()

    elapsed = time.perf_counter() - start_time

    total_calls = sum(
        counter[0]
        for counter in counters
        if counter
    )

    throughput = total_calls / elapsed

    print(f"Compile time:       {compile_time:.4f} s")
    print(f"Elapsed time:       {elapsed:.3f} s")
    print(f"Total inferences:   {total_calls}")
    print(f"Throughput:         {throughput:.2f} inferences/sec")
    print(f"Per-thread average: {throughput / concurrency:.2f} inf/sec")

    return {
        "device": device,
        "concurrency": concurrency,
        "throughput": throughput,
        "total": total_calls,
        "compile": compile_time,
    }


results = []

for device in ["CPU", "GPU", "NPU"]:

    for concurrency in CONCURRENCY_LEVELS:

        try:
            result = run_test(
                device,
                concurrency
            )

            results.append(result)

        except Exception as e:

            print()
            print(
                f"FAILED: {device} | "
                f"Concurrency {concurrency}"
            )

            print(
                type(e).__name__,
                ":",
                e
            )

print()
print("=" * 95)
print("FINAL CONCURRENCY RESULTS")
print("=" * 95)

print(
    f"{'Device':<8}"
    f"{'Conc.':>8}"
    f"{'Throughput':>18}"
    f"{'Per-thread':>18}"
    f"{'Total calls':>16}"
)

print("-" * 95)

for r in results:

    print(
        f"{r['device']:<8}"
        f"{r['concurrency']:>8}"
        f"{r['throughput']:>18.2f}"
        f"{r['throughput'] / r['concurrency']:>18.2f}"
        f"{r['total']:>16}"
    )

print("=" * 95)
print("TEST COMPLETE")