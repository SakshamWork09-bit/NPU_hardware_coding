#include <sycl/sycl.hpp>
#include <iostream>
#include <chrono>
#include <cmath>

int main()
{
    try
    {
        sycl::queue q(
            sycl::gpu_selector_v,
            sycl::property::queue::enable_profiling{}
        );

        std::cout << "GPU: "
                  << q.get_device().get_info<sycl::info::device::name>()
                  << "\n\n";

        constexpr size_t N = 100'000'000;

        float* data = sycl::malloc_shared<float>(N, q);

        // Initialize
        for (size_t i = 0; i < N; ++i)
            data[i] = static_cast<float>(i) * 0.000001f;

        std::cout << "Elements: " << N << "\n";

        // Warm-up
        q.parallel_for(
            sycl::range<1>(N),
            [=](sycl::id<1> i)
            {
                float x = data[i];
                data[i] = sycl::sqrt(x * x + 1.234567f);
            }
        ).wait();

        // Timed GPU kernel
        auto event = q.parallel_for(
            sycl::range<1>(N),
            [=](sycl::id<1> i)
            {
                float x = data[i];

                // Several operations per element
                x = sycl::sqrt(x * x + 1.234567f);
                x = sycl::sqrt(x + 2.345678f);
                x = x * x + 3.456789f;

                data[i] = x;
            }
        );

        event.wait();

        uint64_t start =
            event.get_profiling_info<
                sycl::info::event_profiling::command_start>();

        uint64_t end =
            event.get_profiling_info<
                sycl::info::event_profiling::command_end>();

        double gpu_ms = (end - start) / 1e6;

        std::cout << "GPU kernel time: "
                  << gpu_ms << " ms\n";

        std::cout << "GPU throughput: "
                  << (N / (gpu_ms / 1000.0)) / 1e9
                  << " billion elements/sec\n";

        std::cout << "Sample result: "
                  << data[N / 2] << "\n";

        sycl::free(data, q);
    }
    catch (const sycl::exception& e)
    {
        std::cerr << "\nSYCL ERROR:\n"
                  << e.what() << "\n";

        return 1;
    }

    return 0;
}