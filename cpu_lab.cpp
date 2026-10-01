#include <iostream>
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>
#include <iomanip>

double calculate(long long start, long long end)
{
    double result = 0.0;

    for (long long i = start; i <= end; ++i)
    {
        result += std::sqrt(static_cast<double>(i));
    }

    return result;
}

double run_test(long long N, int thread_count)
{
    std::vector<std::thread> workers;
    std::vector<double> results(thread_count);

    long long chunk = N / thread_count;

    auto start_time = std::chrono::steady_clock::now();

    for (int t = 0; t < thread_count; ++t)
    {
        long long start = t * chunk + 1;

        long long end =
            (t == thread_count - 1)
            ? N
            : (t + 1) * chunk;

        workers.emplace_back(
            [&, t, start, end]()
            {
                results[t] = calculate(start, end);
            }
        );
    }

    for (auto& worker : workers)
        worker.join();

    auto end_time = std::chrono::steady_clock::now();

    double total = 0.0;

    for (double r : results)
        total += r;

    volatile double keep = total;
    (void)keep;

    return std::chrono::duration<double>(
        end_time - start_time
    ).count();
}

int main()
{
    const long long N = 5000000000LL;

    std::cout << "\n=== MOTOBOOK CPU SCALING LAB ===\n";
    std::cout << "Workload: " << N << " sqrt operations per test\n\n";

    double baseline = 0.0;

    int thread_counts[] = {1, 2, 4, 8, 14};

    std::cout << std::fixed << std::setprecision(3);

    for (int threads : thread_counts)
    {
        std::cout << "Running " << threads << " threads...\n";

        double seconds = run_test(N, threads);

        if (threads == 1)
            baseline = seconds;

        double speedup = baseline / seconds;

        double million_ops =
            static_cast<double>(N) / seconds / 1000000.0;

        std::cout
            << "Threads: " << threads
            << " | Time: " << seconds << " s"
            << " | Speedup: " << speedup << "x"
            << " | Rate: " << million_ops << " M ops/s\n\n";
    }

    std::cout << "=== TEST COMPLETE ===\n";

    return 0;
}