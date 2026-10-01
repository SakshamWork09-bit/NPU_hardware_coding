#include <sycl/sycl.hpp>
#include <iostream>

int main()
{
    try
    {
        sycl::queue q(sycl::gpu_selector_v);

        std::cout << "Selected device:\n";
        std::cout << "  "
                  << q.get_device().get_info<sycl::info::device::name>()
                  << "\n\n";

        constexpr int N = 1024;

        int* data = sycl::malloc_shared<int>(N, q);

        q.parallel_for(
            sycl::range<1>(N),
            [=](sycl::id<1> i)
            {
                data[i] = i[0] * 2;
            }
        ).wait();

        std::cout << "GPU kernel completed.\n";
        std::cout << "data[0]    = " << data[0] << "\n";
        std::cout << "data[100]  = " << data[100] << "\n";
        std::cout << "data[1023] = " << data[1023] << "\n";

        sycl::free(data, q);
    }
    catch (const sycl::exception& e)
    {
        std::cerr << "SYCL error:\n"
                  << e.what() << "\n";
        return 1;
    }

    return 0;
}