#include "cuda/primality.cuh"

#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>

namespace goldbach {

namespace {

void check_cuda(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
    }
}

}  // namespace

GpuPrimalityTester::GpuPrimalityTester(std::size_t max_batch) : max_batch_(max_batch) {
    check_cuda(cudaSetDevice(0), "cudaSetDevice failed");
    check_cuda(cudaMalloc(&device_candidates_, max_batch_ * sizeof(std::uint64_t)),
               "cudaMalloc candidates failed");
    check_cuda(cudaMalloc(&device_results_, max_batch_ * sizeof(std::uint8_t)),
               "cudaMalloc results failed");
}

GpuPrimalityTester::~GpuPrimalityTester() {
    if (device_candidates_ != nullptr) {
        cudaFree(device_candidates_);
    }
    if (device_results_ != nullptr) {
        cudaFree(device_results_);
    }
}

void GpuPrimalityTester::test(const std::uint64_t* candidates, std::uint8_t* results, int count) {
    if (count <= 0) {
        return;
    }
    if (static_cast<std::size_t>(count) > max_batch_) {
        throw std::runtime_error("GPU batch exceeds allocated capacity");
    }

    check_cuda(cudaMemcpy(device_candidates_,
                          candidates,
                          static_cast<std::size_t>(count) * sizeof(std::uint64_t),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy H2D failed");
    launch_batch_primality_test(device_candidates_, device_results_, count);
    check_cuda(cudaMemcpy(results,
                          device_results_,
                          static_cast<std::size_t>(count) * sizeof(std::uint8_t),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy D2H failed");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
}

void batch_is_prime_gpu(const std::uint64_t* candidates, std::uint8_t* results, int count) {
    static thread_local GpuPrimalityTester tester(131'072);
    tester.test(candidates, results, count);
}

}  // namespace goldbach
