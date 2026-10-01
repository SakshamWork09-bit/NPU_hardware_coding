import os
import csv
import time
import statistics
import numpy as np
import openvino as ov

from hwinfo_telemetry import HWiNFO, extract


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r"C:\Users\saksh\HardwareLab\mobilenetv2-7.onnx"

DEVICES = [
    "CPU",
    "GPU",
    "NPU",
]

DURATION_SECONDS = 180
WARMUP_RUNS = 100

TELEMETRY_INTERVAL = 0.5
WINDOW_SECONDS = 10.0

OUTPUT_DIR = r"C:\Users\saksh\HardwareLab\thermal_results"


# ============================================================
# HELPERS
# ============================================================

def fmt(value, digits=2):

    if value is None:
        return "N/A"

    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def safe_float(value):

    try:
        return float(value)
    except Exception:
        return None


def bool_flag(value):

    if value is None:
        return 0

    try:
        return 1 if float(value) != 0 else 0
    except Exception:
        return 0


# ============================================================
# BENCHMARK
# ============================================================

def run_device(core, model, device):

    print()
    print("=" * 90)
    print(f" DEVICE: {device}")
    print("=" * 90)

    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    print(f"Compiling model for {device}...")

    compiled = core.compile_model(
        model,
        device
    )

    request = compiled.create_infer_request()

    input_port = compiled.input(0)

    shape = list(input_port.shape)

    print(f"Input shape: {shape}")

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    rng = np.random.default_rng(12345)

    input_data = rng.random(
        shape,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    print(
        f"Warming up ({WARMUP_RUNS} inference calls)..."
    )

    for _ in range(WARMUP_RUNS):

        request.infer({
            input_port.any_name:
                input_data
        })

    print("Warmup complete.")

    # --------------------------------------------------------
    # HWiNFO
    # --------------------------------------------------------

    hw = HWiNFO()
    hw.connect()

    print("HWiNFO telemetry connected.")

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    csv_path = os.path.join(
        OUTPUT_DIR,
        f"{device.lower()}_3min.csv"
    )

    fields = [
        "Timestamp",
        "Elapsed_s",

        "Window_s",
        "Window_Inferences",
        "Window_Throughput_img_s",

        "CPU_Temp_C",
        "CPU_Package_Power_W",
        "CPU_Usage_%",
        "CPU_Effective_Clock_MHz",

        "CPU_Thermal_Throttle",
        "CPU_Package_Power_Limit",
        "CPU_PL1_Limit",
        "CPU_PL2_Limit",
        "CPU_Max_Turbo_Limit",
        "CPU_Thermal_Event",
        "CPU_PROCHOT",
        "CPU_VR_Thermal",

        "GPU_Temp_C",
        "GPU_Power_W",
        "GPU_Clock_MHz",
        "GPU_Util_%",

        "GPU_PL1_Limit",
        "GPU_PL2_Limit",
        "GPU_PL4_Limit",
        "GPU_Thermal_Limit",
        "GPU_Power_Limit",
        "GPU_Software_Limit",
        "GPU_Hardware_Limit",

        "NPU_Clock_MHz",
        "NPU_Util_%",

        "NPU_Power_Limit",
        "NPU_Voltage_Limit",
        "NPU_PL4_Limit",
        "NPU_Thermal_Limit",
        "NPU_Utilization_Limit",
    ]

    # --------------------------------------------------------
    # Runtime
    # --------------------------------------------------------

    start = time.perf_counter()

    next_sample = start

    window_start = start

    window_count = 0

    total_count = 0

    next_progress = 10.0

    telemetry = None

    rows = []

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        try:

            while True:

                now = time.perf_counter()

                elapsed = now - start

                if elapsed >= DURATION_SECONDS:
                    break

                # ------------------------------------------------
                # INFERENCE
                # ------------------------------------------------

                request.infer({
                    input_port.any_name:
                        input_data
                })

                window_count += 1
                total_count += 1

                # ------------------------------------------------
                # HWiNFO SAMPLE
                # ------------------------------------------------

                if now >= next_sample:

                    readings = hw.snapshot()

                    telemetry = extract(
                        readings
                    )

                    next_sample = (
                        now
                        + TELEMETRY_INTERVAL
                    )

                # ------------------------------------------------
                # 10-SECOND WINDOW
                # ------------------------------------------------

                window_elapsed = (
                    now
                    - window_start
                )

                if window_elapsed >= WINDOW_SECONDS:

                    throughput = (
                        window_count
                        / window_elapsed
                    )

                    elapsed_actual = (
                        now
                        - start
                    )

                    values = telemetry or {}

                    row = {
                        "Timestamp":
                            time.strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),

                        "Elapsed_s":
                            round(
                                elapsed_actual,
                                3
                            ),

                        "Window_s":
                            round(
                                window_elapsed,
                                3
                            ),

                        "Window_Inferences":
                            window_count,

                        "Window_Throughput_img_s":
                            round(
                                throughput,
                                3
                            ),

                        **values
                    }

                    writer.writerow(row)
                    f.flush()

                    rows.append(row)

                    # ------------------------------------------------
                    # LIVE OUTPUT
                    # ------------------------------------------------

                    print(
                        f"[{elapsed_actual:6.1f}s] "
                        f"{throughput:8.2f} img/s | "
                        f"CPU "
                        f"{fmt(values.get('CPU_Temp_C'))}°C "
                        f"{fmt(values.get('CPU_Package_Power_W'))}W | "
                        f"GPU "
                        f"{fmt(values.get('GPU_Temp_C'))}°C "
                        f"{fmt(values.get('GPU_Power_W'))}W "
                        f"{fmt(values.get('GPU_Clock_MHz'),0)}MHz | "
                        f"NPU "
                        f"{fmt(values.get('NPU_Clock_MHz'),0)}MHz"
                    )

                    window_start = now
                    window_count = 0

        except KeyboardInterrupt:

            print()
            print("Benchmark interrupted.")

        finally:

            hw.close()

    # ========================================================
    # SUMMARY
    # ========================================================

    if not rows:

        print("No complete telemetry windows recorded.")

        return

    throughputs = [
        safe_float(
            r["Window_Throughput_img_s"]
        )
        for r in rows
    ]

    throughputs = [
        x for x in throughputs
        if x is not None
    ]

    first = throughputs[0]
    last = throughputs[-1]
    peak = max(throughputs)
    minimum = min(throughputs)
    average = statistics.mean(
        throughputs
    )

    print()
    print("-" * 90)
    print(f"{device} SUMMARY")
    print("-" * 90)

    print(
        f"Total inferences : {total_count:,}"
    )

    print(
        f"Average throughput: "
        f"{average:.2f} img/s"
    )

    print(
        f"First window     : "
        f"{first:.2f} img/s"
    )

    print(
        f"Last window      : "
        f"{last:.2f} img/s"
    )

    print(
        f"Peak             : "
        f"{peak:.2f} img/s"
    )

    print(
        f"Minimum          : "
        f"{minimum:.2f} img/s"
    )

    print(
        f"First → Last     : "
        f"{((last / first) - 1) * 100:+.2f}%"
    )

    print()
    print(f"CSV saved to:")
    print(csv_path)

    # --------------------------------------------------------
    # LIMIT EVENTS
    # --------------------------------------------------------

    print()
    print("Detected limit events:")

    event_found = False

    for r in rows:

        events = []

        if bool_flag(
            r.get("CPU_Thermal_Event")
        ):
            events.append("CPU thermal")

        if bool_flag(
            r.get("CPU_PROCHOT")
        ):
            events.append("CPU PROCHOT")

        if bool_flag(
            r.get("CPU_VR_Thermal")
        ):
            events.append("CPU VR thermal")

        if bool_flag(
            r.get("CPU_PL1_Limit")
        ):
            events.append("CPU PL1")

        if bool_flag(
            r.get("CPU_PL2_Limit")
        ):
            events.append("CPU PL2")

        if bool_flag(
            r.get("GPU_Thermal_Limit")
        ):
            events.append("GPU thermal")

        if bool_flag(
            r.get("GPU_Power_Limit")
        ):
            events.append("GPU power")

        if bool_flag(
            r.get("GPU_Hardware_Limit")
        ):
            events.append("GPU hardware")

        if bool_flag(
            r.get("GPU_Software_Limit")
        ):
            events.append("GPU software")

        if events:

            event_found = True

            print(
                f"  {r['Elapsed_s']:6.1f}s -> "
                + ", ".join(events)
            )

    if not event_found:

        print("  No recorded thermal/power limit events.")

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 90)
    print(" HWiNFO + OPENVINO SUSTAINED HARDWARE BENCHMARK")
    print("=" * 90)
    print()

    print(f"Model: {MODEL_PATH}")
    print(f"Duration/device: {DURATION_SECONDS}s")
    print(f"Telemetry interval: {TELEMETRY_INTERVAL}s")
    print()

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    core = ov.Core()

    print(
        "Available OpenVINO devices:",
        core.available_devices
    )

    model = core.read_model(
        MODEL_PATH
    )

    print()

    for device in DEVICES:

        if device not in core.available_devices:

            print(
                f"Skipping {device}: "
                f"device unavailable."
            )

            continue

        run_device(
            core,
            model,
            device
        )

    print()
    print("=" * 90)
    print(" ALL TESTS COMPLETE")
    print("=" * 90)
    print()
    print(
        f"Results folder:\n{OUTPUT_DIR}"
    )
    print()


if __name__ == "__main__":
    main()