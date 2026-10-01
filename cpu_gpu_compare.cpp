#include <iostream>
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>
#include <iomanip>

void worker(float* data, size_t start, size_t end)
{
    for (size_t i = start; i < end; ++i)
    {
        float x = data[i];

        x = std::sqrt(x * x + 1.234567f);
        x = std::sqrt(x + 2.345678f);
        x = x * x + 3.456789f;

        data[i] = x;
    }
}

double run_cpu(size_t N, int threads)
{
    std::vector<float> data(N);

    for (size_t i = 0; i < N; ++i)
        data[i] = static_cast<float>(i) * 0.000001f;

    // Warm-up
    worker(data.data(), 0, N);

    size_t chunk = N / threads;

    std::vector<std::thread> pool;

    auto start = std::chrono::steady_clock::now();

    for (int t = 0; t < threads; ++t)
    {
        size_t begin = t * chunk;
        size_t end = (t == threads - 1) ? N : begin + chunk;

        pool.emplace_back(
            worker,
            data.data(),
            begin,
            end
        );
    }

    for (auto& thread : pool)
        thread.join();

    auto finish = std::chrono::steady_clock::now();

    volatile float check = data[N / 2];

    (void)check;

    return std::chrono::duration<double>(finish - start).count();
}

int main()
{
    constexpr size_t N = 100'000'000;

    std::cout << std::fixed << std::setprecision(3);

    std::cout << "Workload: "
              << N
              << " elements\n\n";

    std::vector<int> thread_counts = {1, 2, 4, 8, 14};

    double single_time = 0.0;

    for (int threads : thread_counts)
    {
        double seconds = run_cpu(N, threads);

        if (threads == 1)
            single_time = seconds;

        double throughput =
            (N / seconds) / 1e9;

        double speedup =
            single_time / seconds;

        std::cout
            << threads
            << " thread"
            << (threads == 1 ? " " : "s ")
            << "| Time: "
            << seconds * 1000.0
            << " ms"
            << " | Speedup: "
            << speedup
            << "x"
            << " | Throughput: "
            << throughput
            << " B elements/s\n";
    }

    return 0;
}