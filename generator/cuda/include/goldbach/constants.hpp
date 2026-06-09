#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace goldbach {

inline constexpr std::uint32_t kMagic = 0x47424B48u;
inline constexpr std::uint32_t kVersion = 1u;
inline constexpr std::size_t kHeaderSize = 64u;
inline constexpr std::size_t kSampleSize = 26u;
inline constexpr std::uint64_t kMaxEven = 4'000'000'000'000'000'000ull;
inline constexpr std::uint32_t kMaxP1 = 5000u;
inline constexpr std::uint64_t kTotalSamples = 100'000'000ull;

inline constexpr std::uint32_t kSmallPrimes[10] = {
    3u, 5u, 7u, 11u, 13u, 17u, 19u, 23u, 29u, 31u};

inline constexpr std::uint64_t kMrBases[7] = {
    2ull,
    325ull,
    9375ull,
    28178ull,
    450775ull,
    9780504ull,
    1795265022ull,
};

struct BucketSpec {
    int id;
    std::uint64_t low;
    std::uint64_t high;
    std::uint64_t target_count;
    bool enumerate;
};

inline const std::vector<BucketSpec>& bucket_specs() {
    static const std::vector<BucketSpec> specs = {
        {1, 4ull, 100ull, 48ull, true},
        {2, 100ull, 1'000ull, 450ull, true},
        {3, 1'000ull, 10'000ull, 4'500ull, true},
        {4, 10'000ull, 100'000ull, 45'000ull, true},
        {5, 100'000ull, 1'000'000ull, 450'000ull, true},
        {6, 1'000'000ull, 10'000'000ull, 4'500'000ull, true},
        {7, 10'000'000ull, 100'000'000ull, 7'916'667ull, false},
        {8, 100'000'000ull, 1'000'000'000ull, 7'916'667ull, false},
        {9, 1'000'000'000ull, 10'000'000'000ull, 7'916'667ull, false},
        {10, 10'000'000'000ull, 100'000'000'000ull, 7'916'667ull, false},
        {11, 100'000'000'000ull, 1'000'000'000'000ull, 7'916'667ull, false},
        {12, 1'000'000'000'000ull, 10'000'000'000'000ull, 7'916'667ull, false},
        {13, 10'000'000'000'000ull, 100'000'000'000'000ull, 7'916'667ull, false},
        {14, 100'000'000'000'000ull, 1'000'000'000'000'000ull, 7'916'667ull, false},
        {15, 1'000'000'000'000'000ull, 10'000'000'000'000'000ull, 7'916'667ull, false},
        {16, 10'000'000'000'000'000ull, 100'000'000'000'000'000ull, 7'916'666ull, false},
        {17, 100'000'000'000'000'000ull, 1'000'000'000'000'000'000ull, 7'916'666ull, false},
        {18, 1'000'000'000'000'000'000ull, kMaxEven + 1ull, 7'916'667ull, false},
    };
    return specs;
}

inline constexpr std::uint64_t kHardCaseCount = 100'000ull;

}  // namespace goldbach
