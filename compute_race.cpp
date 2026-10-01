#include <sycl/sycl.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <thread>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <algorithm>

constexpr size_t N = 20'000'000;
constexpr int REPEATS = 50;
constexpr int CPU_THREADS = 14;
constexpr size_t GPU_GROUP = 32;

// ============================================================
// COMPUTATION
// ============================================================

inline float heavy_math(float x)
{
    for (int r = 0; r < REPEATS; ++r)
    {
        x = std::sqrt(x * x + 1.234567f);
        x = std::sqrt(x + 2.345678f);
        x = x * x + 3.456789f;
        x = std::sqrt(x + 4.567891f);
        x = x * 0.999999f + 0.123456f;
    }

    return x;
}

// ============================================================
// CPU WORKER
// ============================================================

void cpu_worker(
    const float* input,
    float* output,
    size_t start,
    size_t end)
{
    for (size_t i = start; i < end; ++i)
        output[i] = heavy_math(input[i]);
}

// ============================================================
// CPU PORTION
// ============================================================

void run_cpu_part(
    const float* input,
    float* output,
    size_t start,
    size_t end)
{
    if (start >= end)
        return;

    std::vector<std::thread> workers;

    size_t count = end - start;

    size_t chunk =
        count / CPU_THREADS;

    for (int t = 0; t < CPU_THREADS; ++t)
    {
        size_t local_start =
            start + t * chunk;

        size_t local_end =
            (t == CPU_THREADS - 1)
            ? end
            : local_start + chunk;

        workers.emplace_back(
            cpu_worker,
            input,
            output,
            local_start,
            local_end
        );
    }

    for (auto& worker : workers)
        worker.join();
}

// ============================================================
// HYBRID / GPU BENCHMARK
// ============================================================

double run_split(
    sycl::queue& q,
    const std::vector<float>& input,
    std::vector<float>& output,
    double cpu_fraction)
{
    const size_t cpu_count =
        static_cast<size_t>(
            N * cpu_fraction
        );

    const size_t gpu_start =
        cpu_count;

    const size_t gpu_count =
        N - gpu_start;

    float* device_input = nullptr;
    float* device_output = nullptr;

    // --------------------------------------------------------
    // Allocate GPU memory
    // --------------------------------------------------------

    if (gpu_count > 0)
    {
        device_input =
            sycl::malloc_device<float>(
                gpu_count,
                q
            );

        device_output =
            sycl::malloc_device<float>(
                gpu_count,
                q
            );

        if (!device_input ||
            !device_output)
        {
            throw std::runtime_error(
                "GPU allocation failed."
            );
        }
    }

    // --------------------------------------------------------
    // EVERYTHING BELOW IS TIMED
    // --------------------------------------------------------

    auto total_start =
        std::chrono::steady_clock::now();

    // --------------------------------------------------------
    // GPU input transfer
    // --------------------------------------------------------

    if (gpu_count > 0)
    {
        q.memcpy(
            device_input,
            input.data() + gpu_start,
            gpu_count * sizeof(float)
        ).wait();
    }

    // --------------------------------------------------------
    // Start GPU
    // --------------------------------------------------------

    sycl::event gpu_event;

    if (gpu_count > 0)
    {
        gpu_event =
            q.parallel_for(
                sycl::nd_range<1>(
                    sycl::range<1>(gpu_count),
                    sycl::range<1>(GPU_GROUP)
                ),
                [=](sycl::nd_item<1> item)
                {
                    size_t i =
                        item.get_global_id(0);

                    float x =
                        device_input[i];

                    for (int r = 0;
                         r < REPEATS;
                         ++r)
                    {
                        x =
                            sycl::sqrt(
                                x * x + 1.234567f
                            );

                        x =
                            sycl::sqrt(
                                x + 2.345678f
                            );

                        x =
                            x * x + 3.456789f;

                        x =
                            sycl::sqrt(
                                x + 4.567891f
                            );

                        x =
                            x * 0.999999f
                            + 0.123456f;
                    }

                    device_output[i] = x;
                }
            );
    }

    // --------------------------------------------------------
    // CPU runs simultaneously
    // --------------------------------------------------------

    if (cpu_count > 0)
    {
        run_cpu_part(
            input.data(),
            output.data(),
            0,
            cpu_count
        );
    }

    // --------------------------------------------------------
    // GPU must finish
    // --------------------------------------------------------

    if (gpu_count > 0)
    {
        gpu_event.wait();

        q.memcpy(
            output.data() + gpu_start,
            device_output,
            gpu_count * sizeof(float)
        ).wait();
    }

    auto total_end =
        std::chrono::steady_clock::now();

    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

    if (device_input)
        sycl::free(device_input, q);

    if (device_output)
        sycl::free(device_output, q);

    return std::chrono::duration<double>(
        total_end - total_start
    ).count();
}

// ============================================================
// MAIN
// ============================================================

int main()
{
    try
    {
        std::cout << std::fixed
                  << std::setprecision(6);

        std::cout
            << "============================================================\n"
            << "       FAIR CPU + GPU WORKLOAD SPLIT TEST\n"
            << "============================================================\n\n";

        std::cout
            << "Elements:       "
            << N << "\n";

        std::cout
            << "Math repeats:   "
            << REPEATS << "\n";

        std::cout
            << "CPU threads:    "
            << CPU_THREADS << "\n";

        std::cout
            << "GPU work-group: "
            << GPU_GROUP << "\n\n";

        sycl::queue q(
            sycl::gpu_selector_v
        );

        std::cout
            << "GPU: "
            << q.get_device().get_info<
                sycl::info::device::name>()
            << "\n\n";

        // ----------------------------------------------------
        // Prepare input ONCE
        // ----------------------------------------------------

        std::vector<float> input(
            N,
            1.0f
        );

        std::vector<float> output(
            N,
            0.0f
        );

        // ----------------------------------------------------
        // Warm GPU once
        // ----------------------------------------------------

        {
            float* warm_input =
                sycl::malloc_device<float>(
                    1,
                    q
                );

            float* warm_output =
                sycl::malloc_device<float>(
                    1,
                    q
                );

            q.memcpy(
                warm_input,
                input.data(),
                sizeof(float)
            ).wait();

            q.parallel_for(
                sycl::range<1>(1),
                [=](sycl::id<1>)
                {
                    float x =
                        warm_input[0];

                    for (int r = 0;
                         r < REPEATS;
                         ++r)
                    {
                        x =
                            sycl::sqrt(
                                x * x + 1.234567f
                            );

                        x =
                            sycl::sqrt(
                                x + 2.345678f
                            );

                        x =
                            x * x + 3.456789f;

                        x =
                            sycl::sqrt(
                                x + 4.567891f
                            );

                        x =
                            x * 0.999999f
                            + 0.123456f;
                    }

                    warm_output[0] = x;
                }
            ).wait();

            sycl::free(
                warm_input,
                q
            );

            sycl::free(
                warm_output,
                q
            );
        }

        // ----------------------------------------------------
        // SPLITS
        // ----------------------------------------------------

        const double splits[] =
        {
            1.00,   // CPU only
            0.75,   // 75 CPU / 25 GPU
            0.50,   // 50 CPU / 50 GPU
            0.25,   // 25 CPU / 75 GPU
            0.00    // GPU only
        };

        std::cout
            << std::left
            << std::setw(18)
            << "CPU/GPU Split"
            << std::setw(18)
            << "Time (s)"
            << std::setw(18)
            << "Speedup"
            << "Sample\n";

        std::cout
            << "------------------------------------------------------------\n";

        double baseline = 0.0;

        for (double cpu_fraction : splits)
        {
            std::fill(
                output.begin(),
                output.end(),
                0.0f
            );

            double gpu_fraction =
                1.0 - cpu_fraction;

            double time =
                run_split(
                    q,
                    input,
                    output,
                    cpu_fraction
                );

            if (cpu_fraction == 1.0)
                baseline = time;

            double speedup =
                baseline / time;

            size_t sample_index =
                N / 2;

            std::cout
                << std::left
                << std::setw(18);

            if (cpu_fraction == 1.0)
            {
                std::cout
                    << "100 / 0";
            }
            else if (cpu_fraction == 0.75)
            {
                std::cout
                    << "75 / 25";
            }
            else if (cpu_fraction == 0.50)
            {
                std::cout
                    << "50 / 50";
            }
            else if (cpu_fraction == 0.25)
            {
                std::cout
                    << "25 / 75";
            }
            else
            {
                std::cout
                    << "0 / 100";
            }

            std::cout
                << std::setw(18)
                << time

                << std::setw(18)
                << speedup

                << output[sample_index]
                << "\n";
        }

        std::cout
            << "\n============================================================\n";

        std::cout
            << "CPU/GPU split = CPU work percentage / GPU work percentage\n";

        std::cout
            << "Speedup is relative to the 100% CPU baseline.\n";

        std::cout
            << "All timings include GPU transfers and final synchronization.\n";

        std::cout
            << "============================================================\n";
    }
    catch (const sycl::exception& e)
    {
        std::cerr
            << "\nSYCL ERROR:\n"
            << e.what()
            << "\n";

        return 1;
    }
    catch (const std::exception& e)
    {
        std::cerr
            << "\nERROR:\n"
            << e.what()
            << "\n";

        return 1;
    }

    return 0;
}