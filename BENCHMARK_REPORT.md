# Intel Core Ultra Heterogeneous Architecture Benchmark Report

- **Processor**: Intel Core Ultra 5 225H (AI Boost NPU + Arc 130T iGPU)
- **Quantization**: INT8 NNCF Post-Training Quantization
- **Naive CPU Execution**: 22.4 FPS (~44.7 ms)
- **Hardware-Coded Execution**: 82.9 FPS (~12.1 ms)
- **Measured Speedup**: **3.70x faster**

| Architecture Tier | Target Silicon | Precision | Throughput | Latency |
| :--- | :--- | :--- | :--- | :--- |
| Naive Baseline | Host CPU | FP32 | 22.4 FPS | 44.7 ms |
| Hardware Coded | NPU + GPU | INT8 | 82.9 FPS | 12.1 ms |
