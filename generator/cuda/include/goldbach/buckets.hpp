#pragma once

#include <cstdint>
#include <vector>

namespace goldbach {

std::vector<std::uint64_t> collect_bucket_evens(
    int bucket_id,
    std::uint64_t low,
    std::uint64_t high,
    std::uint64_t target_count,
    bool enumerate,
    std::uint64_t seed);

std::vector<std::uint64_t> generate_hard_cases(
    std::uint64_t low,
    std::uint64_t high,
    std::uint64_t target_count,
    std::uint64_t seed);

}  // namespace goldbach
