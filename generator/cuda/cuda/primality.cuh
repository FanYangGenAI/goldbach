#pragma once

#include <cstddef>
#include <cstdint>

namespace goldbach {

void launch_batch_primality_test(
    const std::uint64_t* device_candidates,
    std::uint8_t* device_results,
    int count);

class GpuPrimalityTester {
public:
    explicit GpuPrimalityTester(std::size_t max_batch);
    ~GpuPrimalityTester();

    GpuPrimalityTester(const GpuPrimalityTester&) = delete;
    GpuPrimalityTester& operator=(const GpuPrimalityTester&) = delete;

    void test(const std::uint64_t* candidates, std::uint8_t* results, int count);

private:
    std::size_t max_batch_;
    std::uint64_t* device_candidates_ = nullptr;
    std::uint8_t* device_results_ = nullptr;
};

}  // namespace goldbach
