import os
import csv
import time
import threading
import numpy as np
import openvino as ov
from hwinfo_telemetry import HWiNFO, extract

# ============================================================
# HIGH-FREQUENCY SYNCHRONIZED MICRO-BENCHMARK (200ms)
# ============================================================

MODEL_PATH = r"C:\Users\saksh\HardwareLab\mobilenetv2-7.onnx"
OUTPUT_DIR = r"C:\Users\saksh\HardwareLab\high_freq_results"
DEVICES = ["GPU", "CPU"]       # Focus on GPU and CPU first
DURATION_SEC = 120            # 2 minutes per device is sufficient
TELEMETRY_INTERVAL = 0.200    # 200 ms high-rate sampling (5 Hz)
WINDOW_SEC = 1.0              # Fine-grained 1-second throughput windows

os.makedirs(OUTPUT_DIR, exist_ok=True)

core = ov.Core()
model = core.read_model(MODEL_PATH)
input_port = model.input(0)
input_shape = list(input_port.shape)
input_name = input_port.any_name

rng = np.random.default_rng(42)
dummy_tensor = rng.random(input_shape, dtype=np.float32)

def run_test(device):
    print("\n" + "=" * 80)
    print(f"STARTING HIGH-FREQUENCY RUN: {device}")
    print("=" * 80)

    compiled = core.compile_model(model, device)
    
    # 2 infer requests in flight prevents host-submission queue starvation
    infer_queue = ov.AsyncInferQueue(compiled, jobs=2)

    hw = HWiNFO()
    hw.connect()

    csv_path = os.path.join(OUTPUT_DIR, f"{device.lower()}_highfreq.csv")

    fields = [
        "Timestamp", "Elapsed_s", "Window_Throughput_img_s",
        "CPU_Temp_C", "CPU_Package_Power_W", "CPU_Effective_Clock_MHz",
        "CPU_PROCHOT", "CPU_PL1_Limit", "CPU_PL2_Limit",
        "GPU_Temp_C", "GPU_Power_W", "GPU_Clock_MHz", "GPU_Util_%",
        "GPU_Thermal_Limit", "GPU_Power_Limit"
    ]

    stop_event = threading.Event()
    telemetry_records = []
    
    # Background high-rate telemetry sampler (every 200ms)
    def telemetry_loop():
        start_t = time.perf_counter()
        while not stop_event.is_set():
            t_now = time.perf_counter()
            readings = hw.snapshot()
            data = extract(readings)
            data["Timestamp"] = time.strftime("%H:%M:%S")
            data["Elapsed_s"] = round(t_now - start_t, 3)
            telemetry_records.append(data)
            time.sleep(TELEMETRY_INTERVAL)

    t_thread = threading.Thread(target=telemetry_loop, daemon=True)

    # Warmup
    print("Warming up 50 iterations...")
    for _ in range(50):
        infer_queue.start_async({input_name: dummy_tensor})
    infer_queue.wait_all()

    print("Benchmarking with 1.0s window reporting and 200ms telemetry...")
    t_thread.start()

    benchmark_start = time.perf_counter()
    window_start = benchmark_start
    window_inferences = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        def callback(request, user_data):
            nonlocal window_inferences
            window_inferences += 1

        infer_queue.set_callback(callback)

        while True:
            now = time.perf_counter()
            if now - benchmark_start >= DURATION_SEC:
                break

            infer_queue.start_async({input_name: dummy_tensor})

            # Check 1-second window boundary
            if now - window_start >= WINDOW_SEC:
                infer_queue.wait_all()
                w_elapsed = now - window_start
                throughput = window_inferences / w_elapsed
                
                # Fetch latest 200ms telemetry sample
                latest = telemetry_records[-1] if telemetry_records else {}
                
                row = {
                    "Timestamp": latest.get("Timestamp", time.strftime("%H:%M:%S")),
                    "Elapsed_s": round(now - benchmark_start, 2),
                    "Window_Throughput_img_s": round(throughput, 2),
                    "CPU_Temp_C": latest.get("CPU_Temp_C"),
                    "CPU_Package_Power_W": latest.get("CPU_Package_Power_W"),
                    "CPU_Effective_Clock_MHz": latest.get("CPU_Effective_Clock_MHz"),
                    "CPU_PROCHOT": latest.get("CPU_PROCHOT", 0),
                    "CPU_PL1_Limit": latest.get("CPU_PL1_Limit", 0),
                    "CPU_PL2_Limit": latest.get("CPU_PL2_Limit", 0),
                    "GPU_Temp_C": latest.get("GPU_Temp_C"),
                    "GPU_Power_W": latest.get("GPU_Power_W"),
                    "GPU_Clock_MHz": latest.get("GPU_Clock_MHz"),
                    "GPU_Util_%": latest.get("GPU_Util_%"),
                    "GPU_Thermal_Limit": latest.get("GPU_Thermal_Limit", 0),
                    "GPU_Power_Limit": latest.get("GPU_Power_Limit", 0)
                }
                writer.writerow(row)
                f.flush()

                print(f"[{row['Elapsed_s']:5.1f}s] {throughput:7.1f} img/s | "
                      f"CPU: {row['CPU_Temp_C']}°C {row['CPU_Package_Power_W']}W | "
                      f"GPU: {row['GPU_Clock_MHz']}MHz {row['GPU_Temp_C']}°C")

                window_inferences = 0
                window_start = time.perf_counter()

    stop_event.set()
    t_thread.join(timeout=1.0)
    hw.close()
    print(f"Saved: {csv_path}")

for dev in DEVICES:
    run_test(dev)