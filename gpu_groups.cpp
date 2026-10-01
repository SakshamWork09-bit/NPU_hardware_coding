#include <sycl/sycl.hpp>
#include <iostream>
#include <iomanip>
#include <vector>

int main()
{
    try
    {
        sycl::queue q(
            sycl::gpu_selector_v,
            sycl::property::queue::enable_profiling{}
        );

        auto device = q.get_device();

        std::cout << "GPU: "
                  << device.get_info<sycl::info::device::name>()
                  << "\n\n";

        constexpr size_t N = 100'000'000;
        constexpr size_t GROUP = 32;

        const double GB = 1e9;

        std::vector<float> host(N, 1.0f);

        float* device_data =
            sycl::malloc_device<float>(N, q);

        if (!device_data)
        {
            std::cerr << "Device allocation failed.\n";
            return 1;
        }

        // -----------------------------
        // HOST -> GPU
        // -----------------------------
        auto h2d_event =
            q.memcpy(device_data,
                     host.data(),
                     N * sizeof(float));

        h2d_event.wait();

        double h2d_ms =
            (h2d_event.get_profiling_info<
                sycl::info::event_profiling::command_end>() -
             h2d_event.get_profiling_info<
                sycl::info::event_profiling::command_start>())
            / 1e6;

        // -----------------------------
        // GPU MEMORY BANDWIDTH
        // -----------------------------
        auto kernel_event =
            q.parallel_for(
                sycl::nd_range<1>(
                    sycl::range<1>(N),
                    sycl::range<1>(GROUP)
                ),
                [=](sycl::nd_item<1> item)
                {
                    size_t i = item.get_global_id(0);

                    float x = device_data[i];

                    // One read + one write
                    device_data[i] =
                        x * 1.000001f + 1.0f;
                }
            );

        kernel_event.wait();

        double kernel_ms =
            (kernel_event.get_profiling_info<
                sycl::info::event_profiling::command_end>() -
             kernel_event.get_profiling_info<
                sycl::info::event_profiling::command_start>())
            / 1e6;

        // -----------------------------
        // GPU -> HOST
        // -----------------------------
        auto d2h_event =
            q.memcpy(host.data(),
                     device_data,
                     N * sizeof(float));

        d2h_event.wait();

        double d2h_ms =
            (d2h_event.get_profiling_info<
                sycl::info::event_profiling::command_end>() -
             d2h_event.get_profiling_info<
                sycl::info::event_profiling::command_start>())
            / 1e6;

        // --------------------------------
        // BANDWIDTH CALCULATION
        // --------------------------------

        // Kernel:
        // 1 float read + 1 float write = 8 bytes
        double kernel_bytes =
            static_cast<double>(N) * 8.0;

        double kernel_bandwidth =
            kernel_bytes / (kernel_ms / 1000.0) / GB;

        double transfer_bytes =
            static_cast<double>(N) * sizeof(float);

        double h2d_bandwidth =
            transfer_bytes / (h2d_ms / 1000.0) / GB;

        double d2h_bandwidth =
            transfer_bytes / (d2h_ms / 1000.0) / GB;

        std::cout << std::fixed << std::setprecision(3);

        std::cout << "Elements:              " << N << "\n";
        std::cout << "Work-group:            " << GROUP << "\n\n";

        std::cout << "Host -> GPU:           "
                  << h2d_ms << " ms\n";

        std::cout << "H2D bandwidth:         "
                  << h2d_bandwidth << " GB/s\n\n";

        std::cout << "GPU kernel:            "
                  << kernel_ms << " ms\n";

        std::cout << "GPU memory bandwidth:  "
                  << kernel_bandwidth << " GB/s\n\n";

        std::cout << "GPU -> Host:           "
                  << d2h_ms << " ms\n";

        std::cout << "D2H bandwidth:         "
                  << d2h_bandwidth << " GB/s\n\n";

        std::cout << "Sample result:         "
                  << host[0] << "\n";

        sycl::free(device_data, q);
    }
    catch (const sycl::exception& e)
    {
        std::cerr << "\nSYCL ERROR:\n"
                  << e.what() << "\n";

        return 1;
    }

    return 0;
}