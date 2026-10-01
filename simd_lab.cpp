#include <immintrin.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <thread>
#include <vector>
#include <cmath>

volatile double final_sink = 0.0;

// ============================================================
// AVX2 WORKER
// Each thread processes its own portion of the workload.
// ============================================================

__declspec(noinline)
double avx2_worker(long long start, long long end)
{
    __m256d vsum = _mm256_setzero_pd();

    const __m256d vstep =
        _mm256_set1_pd(0.000001);

    const __m256d vone =
        _mm256_set1_pd(1.234567);

    const __m256d four =
        _mm256_set1_pd(4.0);

    __m256d vi =
        _mm256_set_pd(
            (double)start + 3.0,
            (double)start + 2.0,
            (double)start + 1.0,
            (double)start
        );

    long long i = start;

    for (; i + 4 <= end; i += 4)
    {
        // x = i * 0.000001
        __m256d x =
            _mm256_mul_pd(vi, vstep);

        // x²
        __m256d x2 =
            _mm256_mul_pd(x, x);

        // x² + 1.234567
        __m256d result =
            _mm256_add_pd(x2, vone);

        // Accumulate
        vsum =
            _mm256_add_pd(vsum, result);

        vi =
            _mm256_add_pd(vi, four);
    }

    // Extract the four SIMD values
    alignas(32) double values[4];

    _mm256_store_pd(values, vsum);

    double sum =
        values[0] +
        values[1] +
        values[2] +
        values[3];

    // Handle any remaining elements
    for (; i < end; ++i)
    {
        double x =
            (double)i * 0.000001;

        sum += x * x + 1.234567;
    }

    return sum;
}


// ============================================================
// MULTITHREADED AVX2 TEST
// ============================================================

double run_avx2(long long N, int thread_count)
{
    std::vector<std::thread> threads;
    std::vector<double> results(thread_count);

    long long chunk =
        N / thread_count;

    for (int t = 0; t < thread_count; ++t)
    {
        long long start =
            t * chunk;

        long long end =
            (t == thread_count - 1)
                ? N
                : start + chunk;

        threads.emplace_back(
            [&, t, start, end]()
            {
                results[t] =
                    avx2_worker(start, end);
            }
        );
    }

    for (auto& thread : threads)
    {
        thread.join();
    }

    double total = 0.0;

    for (double value : results)
    {
        total += value;
    }

    final_sink = total;

    return total;
}


// ============================================================
// MAIN
// ============================================================

int main()
{
    const long long N =
        5'000'000'000LL;

    const int thread_counts[] =
    {
        1, 2, 4, 8, 14
    };

    std::cout
        << "========================================\n"
        << "       CPU + AVX2 HARDWARE LAB\n"
        << "========================================\n\n";

    std::cout
        << "Workload: "
        << N
        << " calculations\n\n";

    double baseline = 0.0;

    for (int thread_count : thread_counts)
    {
        auto start =
            std::chrono::steady_clock::now();

        double result =
            run_avx2(N, thread_count);

        auto end =
            std::chrono::steady_clock::now();

        double seconds =
            std::chrono::duration<double>(
                end - start
            ).count();

        if (thread_count == 1)
        {
            baseline = seconds;
        }

        double speedup =
            baseline / seconds;

        double throughput =
            (double)N /
            seconds /
            1e9;

        std::cout
            << std::fixed
            << std::setprecision(3);

        std::cout
            << thread_count
            << " thread";

        if (thread_count != 1)
            std::cout << "s";

        std::cout
            << " | Time: "
            << seconds
            << " s"
            << " | Speedup: "
            << speedup
            << "x"
            << " | Throughput: "
            << throughput
            << " billion calc/s\n";
    }

    std::cout
        << "\n========================================\n";

    std::cout
        << "Final result: "
        << final_sink
        << "\n";

    std::cout
        << "========================================\n";

    return 0;
}