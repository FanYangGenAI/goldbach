#include "cuda/primality.cuh"

#include <cstdint>

namespace {

__device__ __forceinline__ std::uint64_t mul_mod(std::uint64_t a, std::uint64_t b, std::uint64_t mod) {
#if defined(__CUDA_ARCH__)
    std::uint64_t high = __umul64hi(a, b);
    std::uint64_t low = a * b;
#else
    unsigned __int128 product = static_cast<unsigned __int128>(a) * b;
    std::uint64_t low = static_cast<std::uint64_t>(product);
    std::uint64_t high = static_cast<std::uint64_t>(product >> 64);
#endif
    if (high == 0) {
        return low % mod;
    }

    std::uint64_t remainder = 0;
    for (int bit = 63; bit >= 0; --bit) {
        remainder <<= 1;
        remainder |= (high >> bit) & 1ull;
        if (remainder >= mod) {
            remainder -= mod;
        }
    }
    for (int bit = 63; bit >= 0; --bit) {
        remainder <<= 1;
        remainder |= (low >> bit) & 1ull;
        if (remainder >= mod) {
            remainder -= mod;
        }
    }
    return remainder;
}

__device__ __forceinline__ std::uint64_t pow_mod(std::uint64_t base, std::uint64_t exp, std::uint64_t mod) {
    std::uint64_t result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1ull) {
            result = mul_mod(result, base, mod);
        }
        base = mul_mod(base, base, mod);
        exp >>= 1ull;
    }
    return result;
}

__device__ bool is_prime_gpu(std::uint64_t n) {
    if (n < 2) {
        return false;
    }
    if (n == 2 || n == 3) {
        return true;
    }
    if ((n & 1ull) == 0) {
        return false;
    }

    std::uint64_t d = n - 1;
    int s = 0;
    while ((d & 1ull) == 0) {
        d >>= 1ull;
        ++s;
    }

    const std::uint64_t bases[7] = {2ull, 325ull, 9375ull, 28178ull, 450775ull, 9780504ull, 1795265022ull};
    for (int index = 0; index < 7; ++index) {
        const std::uint64_t a = bases[index] % n;
        if (a == 0) {
            continue;
        }

        std::uint64_t x = pow_mod(a, d, n);
        if (x == 1 || x == n - 1) {
            continue;
        }

        bool composite = true;
        for (int round = 1; round < s; ++round) {
            x = mul_mod(x, x, n);
            if (x == n - 1) {
                composite = false;
                break;
            }
        }
        if (composite) {
            return false;
        }
    }
    return true;
}

__global__ void batch_primality_test_kernel(
    const std::uint64_t* candidates,
    std::uint8_t* results,
    int num_candidates) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_candidates) {
        results[idx] = is_prime_gpu(candidates[idx]) ? 1 : 0;
    }
}

}  // namespace

namespace goldbach {

void launch_batch_primality_test(
    const std::uint64_t* device_candidates,
    std::uint8_t* device_results,
    int count) {
    constexpr int kBlockSize = 256;
    const int blocks = (count + kBlockSize - 1) / kBlockSize;
    batch_primality_test_kernel<<<blocks, kBlockSize>>>(device_candidates, device_results, count);
}

}  // namespace goldbach
