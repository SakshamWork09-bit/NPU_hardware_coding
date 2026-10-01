import openvino as ov
import numpy as np
import threading
import queue
import time
import statistics
from dataclasses import dataclass


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = r"C:\Users\saksh\HardwareLab\mobilenetv2-7.onnx"

NUM_JOBS = 10000

GPU_WORKERS = 1
NPU_WORKERS = 2

# Small bounded queue per worker.
QUEUE_SIZE = 8

# Calibration:
# enough samples to obtain a stable estimate without
# making startup excessively long.
CALIBRATION_WARMUP = 20
CALIBRATION_RUNS = 50

INPUT = np.random.rand(
    1, 3, 224, 224
).astype(np.float32)


# ============================================================
# JOB
# ============================================================

@dataclass
class Job:

    job_id: int

    created: float = 0.0
    dispatched: float = 0.0
    started: float = 0.0
    finished: float = 0.0

    device: str = ""
    worker: str = ""

    error: str = ""

    @property
    def queue_latency_ms(self):

        return (
            self.started
            - self.created
        ) * 1000.0

    @property
    def inference_latency_ms(self):

        return (
            self.finished
            - self.started
        ) * 1000.0

    @property
    def end_to_end_ms(self):

        return (
            self.finished
            - self.created
        ) * 1000.0


# ============================================================
# WORKER
# ============================================================

class Worker:

    def __init__(
        self,
        name,
        device,
        compiled_model
    ):

        self.name = name
        self.device = device
        self.compiled_model = compiled_model

        self.jobs = queue.Queue(
            maxsize=QUEUE_SIZE
        )

        self.thread = threading.Thread(
            target=self.run,
            name=self.name,
            daemon=True
        )

        self.completed = 0
        self.failed = 0

        self.latencies = []

        # Learned latency.
        self.estimated_latency_ms = None

        # Number of jobs assigned but not yet completed.
        self.pending = 0

        self.pending_lock = threading.Lock()

    # --------------------------------------------------------

    def start(self):

        self.thread.start()

    # --------------------------------------------------------

    def get_pending(self):

        with self.pending_lock:

            return self.pending

    # --------------------------------------------------------

    def increment_pending(self):

        with self.pending_lock:

            self.pending += 1

    # --------------------------------------------------------

    def decrement_pending(self):

        with self.pending_lock:

            self.pending -= 1

    # --------------------------------------------------------

    def set_latency(self, latency):

        self.estimated_latency_ms = latency

    # --------------------------------------------------------

    def update_latency(self, latency):

        # Smooth online adaptation.

        if self.estimated_latency_ms is None:

            self.estimated_latency_ms = latency

        else:

            self.estimated_latency_ms = (
                0.95 * self.estimated_latency_ms
                + 0.05 * latency
            )

    # --------------------------------------------------------

    def predicted_finish_ms(self):

        latency = self.estimated_latency_ms

        if latency is None:

            latency = 1.0

        pending = self.get_pending()

        return pending * latency

    # --------------------------------------------------------

    def submit(self, job):

        self.increment_pending()

        self.jobs.put(job)

    # --------------------------------------------------------

    def run(self):

        request = (
            self.compiled_model
            .create_infer_request()
        )

        # ====================================================
        # WARMUP
        # ====================================================

        for _ in range(CALIBRATION_WARMUP):

            request.infer({
                0: INPUT
            })

        # ====================================================
        # CALIBRATION
        # ====================================================

        calibration = []

        for _ in range(CALIBRATION_RUNS):

            t0 = time.perf_counter()

            request.infer({
                0: INPUT
            })

            t1 = time.perf_counter()

            calibration.append(
                (t1 - t0) * 1000.0
            )

        self.estimated_latency_ms = (
            statistics.median(
                calibration
            )
        )

        print(
            f"[CALIBRATED] "
            f"{self.name}: "
            f"{self.estimated_latency_ms:.3f} ms "
            f"â‰ˆ "
            f"{1000.0 / self.estimated_latency_ms:.1f} img/s"
        )

        # ====================================================
        # WORK
        # ====================================================

        while True:

            job = self.jobs.get()

            if job is None:

                self.jobs.task_done()

                break

            try:

                job.device = self.device
                job.worker = self.name

                job.dispatched = (
                    time.perf_counter()
                )

                job.started = (
                    time.perf_counter()
                )

                request.infer({
                    0: INPUT
                })

                job.finished = (
                    time.perf_counter()
                )

                latency = (
                    job.inference_latency_ms
                )

                self.latencies.append(
                    latency
                )

                self.update_latency(
                    latency
                )

                self.completed += 1

            except Exception as e:

                job.finished = (
                    time.perf_counter()
                )

                job.error = str(e)

                self.failed += 1

            finally:

                self.decrement_pending()

                self.jobs.task_done()


# ============================================================
# PREDICTIVE ROUTER
# ============================================================

class PredictiveRouter:

    def __init__(self, workers):

        self.workers = workers

    def choose_worker(self):

        # ----------------------------------------------------
        # Choose the worker with the earliest predicted
        # completion time.
        #
        # Example:
        #
        # GPU:
        #   2 pending Ã— 0.8ms = 1.6ms
        #
        # NPU:
        #   1 pending Ã— 1.6ms = 1.6ms
        #
        # Both are approximately equally loaded.
        # ----------------------------------------------------

        while True:

            candidates = [
                worker
                for worker in self.workers
                if not worker.jobs.full()
            ]

            if candidates:

                return min(
                    candidates,
                    key=lambda worker:
                        worker.predicted_finish_ms()
                )

            # Every queue is temporarily full.
            # Wait briefly rather than spinning.
            time.sleep(0.00005)


# ============================================================
# STATISTICS
# ============================================================

def percentile(values, p):

    if not values:

        return 0.0

    values = sorted(values)

    index = (
        (len(values) - 1)
        * p
        / 100.0
    )

    low = int(index)

    high = min(
        low + 1,
        len(values) - 1
    )

    fraction = index - low

    return (
        values[low]
        + (
            values[high]
            - values[low]
        ) * fraction
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "HETEROGENEOUS AI RUNTIME V3"
    )
    print(
        "PREDICTIVE ADAPTIVE SCHEDULER"
    )
    print("=" * 70)

    # ========================================================
    # OPENVINO
    # ========================================================

    core = ov.Core()

    print("\nAvailable devices:")
    print(core.available_devices)

    print("\nLoading model:")
    print(MODEL_PATH)

    model = core.read_model(
        MODEL_PATH
    )

    # ========================================================
    # COMPILE
    # ========================================================

    print("\nCompiling GPU...")

    gpu_model = core.compile_model(
        model,
        "GPU"
    )

    print("GPU ready.")

    print("\nCompiling NPU...")

    npu_model = core.compile_model(
        model,
        "NPU"
    )

    print("NPU ready.")

    # ========================================================
    # WORKERS
    # ========================================================

    workers = []

    for i in range(GPU_WORKERS):

        workers.append(
            Worker(
                f"GPU-{i}",
                "GPU",
                gpu_model
            )
        )

    for i in range(NPU_WORKERS):

        workers.append(
            Worker(
                f"NPU-{i}",
                "NPU",
                npu_model
            )
        )

    print("\nWorkers:")

    for worker in workers:

        print(
            f"  {worker.name}: "
            f"{worker.device}"
        )

    # ========================================================
    # START WORKERS
    # ========================================================

    print("\nStarting workers...")

    for worker in workers:

        worker.start()

    # ========================================================
    # WAIT FOR CALIBRATION
    # ========================================================

    print(
        "\nCalibrating hardware..."
    )

    while any(
        worker.estimated_latency_ms
        is None
        for worker in workers
    ):

        time.sleep(0.01)

    print(
        "\nLearned hardware rates:"
    )

    for worker in workers:

        latency = (
            worker.estimated_latency_ms
        )

        rate = 1000.0 / latency

        print(
            f"  {worker.name}: "
            f"{latency:.3f} ms "
            f"â‰ˆ {rate:.1f} img/s"
        )

    # ========================================================
    # ROUTER
    # ========================================================

    router = PredictiveRouter(
        workers
    )

    results = []

    print(
        "\nStarting predictive scheduler..."
    )

    print(
        f"Jobs: {NUM_JOBS}"
    )

    print(
        f"Queue size per worker: "
        f"{QUEUE_SIZE}"
    )

    # ========================================================
    # DISPATCH
    # ========================================================

    total_start = time.perf_counter()

    for job_id in range(NUM_JOBS):

        job = Job(
            job_id=job_id,
            created=time.perf_counter()
        )

        worker = (
            router.choose_worker()
        )

        worker.submit(job)

        results.append(job)

    dispatch_end = time.perf_counter()

    print(
        f"\nAll jobs dispatched in "
        f"{dispatch_end - total_start:.3f} s"
    )

    print(
        "Waiting for workers..."
    )

    # ========================================================
    # WAIT FOR COMPLETION
    # ========================================================

    for worker in workers:

        worker.jobs.join()

    total_end = time.perf_counter()

    # ========================================================
    # RESULTS
    # ========================================================

    successful = [
        job
        for job in results
        if not job.error
    ]

    failed = [
        job
        for job in results
        if job.error
    ]

    elapsed = (
        total_end
        - total_start
    )

    throughput = (
        len(successful)
        / elapsed
    )

    queue_latency = [
        job.queue_latency_ms
        for job in successful
    ]

    inference_latency = [
        job.inference_latency_ms
        for job in successful
    ]

    end_to_end = [
        job.end_to_end_ms
        for job in successful
    ]

    # ========================================================
    # DISTRIBUTION
    # ========================================================

    distribution = {}

    for job in successful:

        distribution[job.worker] = (
            distribution.get(
                job.worker,
                0
            ) + 1
        )

    # ========================================================
    # REPORT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("V3 RESULT")
    print("=" * 70)

    print(
        f"\nCompleted:       "
        f"{len(successful)}"
    )

    print(
        f"Failed:          "
        f"{len(failed)}"
    )

    print(
        f"Total time:      "
        f"{elapsed:.3f} s"
    )

    print(
        f"Throughput:      "
        f"{throughput:.2f} images/s"
    )

    # --------------------------------------------------------
    # Dispatch overhead
    # --------------------------------------------------------

    print(
        f"Dispatch time:   "
        f"{dispatch_end - total_start:.3f} s"
    )

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    print(
        "\nAdaptive distribution:"
    )

    for worker, count in sorted(
        distribution.items()
    ):

        percentage = (
            count
            / len(successful)
            * 100.0
        )

        print(
            f"  {worker}: "
            f"{count} "
            f"({percentage:.2f}%)"
        )

    # --------------------------------------------------------
    # Learned rates
    # --------------------------------------------------------

    print(
        "\nFinal learned rates:"
    )

    for worker in workers:

        latency = (
            worker.estimated_latency_ms
        )

        rate = (
            1000.0
            / latency
        )

        print(
            f"  {worker.name}: "
            f"{latency:.3f} ms "
            f"â‰ˆ {rate:.1f} img/s"
        )

    # --------------------------------------------------------
    # Queue
    # --------------------------------------------------------

    print(
        "\nQueue latency:"
    )

    print(
        f"  Median: "
        f"{statistics.median(queue_latency):.3f} ms"
    )

    print(
        f"  P95:    "
        f"{percentile(queue_latency, 95):.3f} ms"
    )

    print(
        f"  P99:    "
        f"{percentile(queue_latency, 99):.3f} ms"
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    print(
        "\nInference latency:"
    )

    print(
        f"  Median: "
        f"{statistics.median(inference_latency):.3f} ms"
    )

    print(
        f"  P95:    "
        f"{percentile(inference_latency, 95):.3f} ms"
    )

    print(
        f"  P99:    "
        f"{percentile(inference_latency, 99):.3f} ms"
    )

    # --------------------------------------------------------
    # End-to-end
    # --------------------------------------------------------

    print(
        "\nEnd-to-end latency:"
    )

    print(
        f"  Median: "
        f"{statistics.median(end_to_end):.3f} ms"
    )

    print(
        f"  P95:    "
        f"{percentile(end_to_end, 95):.3f} ms"
    )

    print(
        f"  P99:    "
        f"{percentile(end_to_end, 99):.3f} ms"
    )

    # ========================================================
    # WORKERS
    # ========================================================

    print(
        "\nWorker statistics:"
    )

    for worker in workers:

        print(
            f"\n  {worker.name}"
        )

        print(
            f"    Device:    "
            f"{worker.device}"
        )

        print(
            f"    Completed: "
            f"{worker.completed}"
        )

        print(
            f"    Failed:    "
            f"{worker.failed}"
        )

        if worker.latencies:

            print(
                f"    Median:    "
                f"{statistics.median(worker.latencies):.3f} ms"
            )

            print(
                f"    P95:       "
                f"{percentile(worker.latencies, 95):.3f} ms"
            )

    # ========================================================
    # COMPARISON
    # ========================================================

    print("\n")
    print("=" * 70)
    print("BASELINE COMPARISON")
    print("=" * 70)

    baseline = 2287.54

    difference = (
        throughput
        / baseline
        - 1.0
    ) * 100.0

    print(
        f"\nFixed scheduler:   "
        f"{baseline:.2f} img/s"
    )

    print(
        f"V3 adaptive:       "
        f"{throughput:.2f} img/s"
    )

    print(
        f"Difference:        "
        f"{difference:+.2f}%"
    )

    if throughput > baseline:

        print(
            "\nRESULT: "
            "ADAPTIVE SCHEDULER BEATS BASELINE"
        )

    else:

        print(
            "\nRESULT: "
            "FIXED SCHEDULER STILL WINS"
        )

    print(
        "\n" + "=" * 70
    )


if __name__ == "__main__":

    main()
