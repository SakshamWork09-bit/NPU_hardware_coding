import openvino as ov
import numpy as np
import time
import statistics
import os

BASE = r"C:\Users\saksh\HardwareLab"

MODELS = [
    ("MobileNetV2", os.path.join(BASE, "mobilenetv2-7.onnx")),
    ("SqueezeNet1.1", os.path.join(BASE, "squeezenet1.1-7.onnx")),
    ("ResNet50", os.path.join(BASE, "resnet50-v1-7.onnx")),
]

DEVICES = ["CPU", "GPU", "NPU"]

BATCHES = [1, 2, 4, 8]

WARMUP = 50
RUNS = 300

IMAGE_SIZE = [3, 224, 224]


print("=" * 76)
print("CPU / GPU / NPU BATCH-SIZE SCALING LAB")
print("=" * 76)
print(f"Batches: {BATCHES}")
print(f"Warmup per configuration: {WARMUP}")
print(f"Timed runs per configuration: {RUNS}")

core = ov.Core()

print("\nAvailable devices:")
print(core.available_devices)

results = {}


def prepare_model(model, batch):

    input_port = model.input(0)

    target_shape = [batch] + IMAGE_SIZE

    if not input_port.partial_shape.is_static:

        model.reshape({
            input_port.any_name: target_shape
        })

    else:

        current_shape = list(input_port.shape)

        if current_shape != target_shape:

            model.reshape({
                input_port.any_name: target_shape
            })

    return model


for model_name, model_path in MODELS:

    print("\n" + "#" * 76)
    print(f"MODEL: {model_name}")
    print("#" * 76)

    if not os.path.exists(model_path):
        print("SKIPPED: model file not found")
        continue

    results[model_name] = {}

    for batch in BATCHES:

        print("\n" + "=" * 76)
        print(f"BATCH SIZE: {batch}")
        print("=" * 76)

        results[model_name][batch] = {}

        for device in DEVICES:

            print("\n" + "-" * 76)
            print(f"DEVICE: {device} | BATCH: {batch}")
            print("-" * 76)

            try:

                # Reload model for every batch so each
                # configuration starts from a clean graph.
                model = core.read_model(model_path)

                model = prepare_model(
                    model,
                    batch
                )

                input_port = model.input(0)

                input_shape = list(
                    input_port.shape
                )

                input_name = input_port.any_name

                print(
                    f"Input shape: {input_shape}"
                )

                INPUT = np.random.rand(
                    *input_shape
                ).astype(np.float32)

                compiled = core.compile_model(
                    model,
                    device
                )

                request = compiled.create_infer_request()

                # --------------------------------------------------
                # WARMUP
                # --------------------------------------------------

                for _ in range(WARMUP):

                    request.infer({
                        input_name: INPUT
                    })

                # --------------------------------------------------
                # BENCHMARK
                # --------------------------------------------------

                latencies = []

                start = time.perf_counter()

                for _ in range(RUNS):

                    t0 = time.perf_counter()

                    request.infer({
                        input_name: INPUT
                    })

                    t1 = time.perf_counter()

                    latencies.append(
                        (t1 - t0) * 1000.0
                    )

                total_time = (
                    time.perf_counter() - start
                )

                total_images = (
                    RUNS * batch
                )

                images_per_second = (
                    total_images / total_time
                )

                batches_per_second = (
                    RUNS / total_time
                )

                median_latency = statistics.median(
                    latencies
                )

                p95_latency = float(
                    np.percentile(
                        latencies,
                        95
                    )
                )

                p99_latency = float(
                    np.percentile(
                        latencies,
                        99
                    )
                )

                best = min(latencies)
                worst = max(latencies)

                results[
                    model_name
                ][batch][device] = {
                    "throughput": images_per_second,
                    "batch_rate": batches_per_second,
                    "median": median_latency,
                    "p95": p95_latency,
                    "p99": p99_latency,
                }

                print(
                    f"Batch throughput: {batches_per_second:.2f} batches/s"
                )

                print(
                    f"Image throughput: {images_per_second:.2f} img/s"
                )

                print(
                    f"Median latency:   {median_latency:.4f} ms"
                )

                print(
                    f"P95 latency:      {p95_latency:.4f} ms"
                )

                print(
                    f"P99 latency:      {p99_latency:.4f} ms"
                )

                print(
                    f"Best:              {best:.4f} ms"
                )

                print(
                    f"Worst:             {worst:.4f} ms"
                )

            except Exception as e:

                print(
                    f"FAILED on {device}, batch {batch}: {e}"
                )


# ================================================================
# THROUGHPUT TABLES
# ================================================================

print("\n\n" + "=" * 76)
print("IMAGE THROUGHPUT — img/s")
print("=" * 76)

for model_name in results:

    print("\n" + model_name)

    print(
        f"{'Batch':<10}"
        f"{'CPU':>14}"
        f"{'GPU':>14}"
        f"{'NPU':>14}"
    )

    for batch in BATCHES:

        row = results[model_name].get(
            batch,
            {}
        )

        cpu = row.get("CPU", {}).get(
            "throughput", 0
        )

        gpu = row.get("GPU", {}).get(
            "throughput", 0
        )

        npu = row.get("NPU", {}).get(
            "throughput", 0
        )

        print(
            f"{batch:<10}"
            f"{cpu:>14.2f}"
            f"{gpu:>14.2f}"
            f"{npu:>14.2f}"
        )


# ================================================================
# PER-IMAGE MEDIAN LATENCY
# ================================================================

print("\n\n" + "=" * 76)
print("MEDIAN LATENCY PER IMAGE — ms")
print("=" * 76)

for model_name in results:

    print("\n" + model_name)

    print(
        f"{'Batch':<10}"
        f"{'CPU':>14}"
        f"{'GPU':>14}"
        f"{'NPU':>14}"
    )

    for batch in BATCHES:

        row = results[model_name].get(
            batch,
            {}
        )

        values = []

        for device in DEVICES:

            latency = row.get(
                device,
                {}
            ).get(
                "median",
                0
            )

            # Convert batch latency into
            # approximate per-image latency.
            per_image = (
                latency / batch
                if batch > 0
                else 0
            )

            values.append(per_image)

        print(
            f"{batch:<10}"
            f"{values[0]:>14.4f}"
            f"{values[1]:>14.4f}"
            f"{values[2]:>14.4f}"
        )


# ================================================================
# GPU SCALING
# ================================================================

print("\n\n" + "=" * 76)
print("GPU BATCH SCALING")
print("=" * 76)

for model_name in results:

    batch1 = results[model_name].get(
        1,
        {}
    ).get(
        "GPU",
        {}
    ).get(
        "throughput",
        0
    )

    print(f"\n{model_name}")

    for batch in BATCHES:

        throughput = results[model_name].get(
            batch,
            {}
        ).get(
            "GPU",
            {}
        ).get(
            "throughput",
            0
        )

        if batch1 > 0:

            speedup = throughput / batch1

            print(
                f"Batch {batch:<2}: "
                f"{throughput:>8.2f} img/s   "
                f"{speedup:.2f}x vs batch 1"
            )


print("\n" + "=" * 76)
print("BATCH-SIZE LAB COMPLETE")
print("=" * 76)