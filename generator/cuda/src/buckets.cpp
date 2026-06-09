#include "goldbach/buckets.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace goldbach {

namespace {

std::uint64_t round_to_even_in_range(double value, std::uint64_t low, std::uint64_t high) {
    if (high <= low) {
        throw std::runtime_error("Invalid bucket range");
    }

    std::uint64_t candidate = static_cast<std::uint64_t>(std::llround(value));
    if (candidate < low) {
        candidate = (low % 2 == 0) ? low : low + 1;
    }
    if (candidate >= high) {
        candidate = (high % 2 == 0) ? high - 2 : high - 1;
    }
    if (candidate % 2 != 0) {
        if (candidate + 1 < high) {
            ++candidate;
        } else if (candidate > low) {
            --candidate;
        }
    }
    if (candidate < low || candidate >= high || candidate % 2 != 0) {
        throw std::runtime_error("Failed to round candidate into bucket range");
    }
    return candidate;
}

std::vector<std::uint64_t> enumerate_evens(std::uint64_t low, std::uint64_t high) {
    std::vector<std::uint64_t> values;
    const std::uint64_t start = (low % 2 == 0) ? low : low + 1;
    for (std::uint64_t value = start; value < high; value += 2) {
        values.push_back(value);
    }
    return values;
}

std::vector<std::uint64_t> sample_log_uniform(
    std::uint64_t low,
    std::uint64_t high,
    std::uint64_t target_count,
    std::mt19937_64& rng) {
    std::unordered_set<std::uint64_t> seen;
    seen.reserve(static_cast<std::size_t>(target_count * 2));
    std::vector<std::uint64_t> values;
    values.reserve(static_cast<std::size_t>(target_count));

    std::uniform_real_distribution<double> dist(std::log(static_cast<double>(low)),
                                                std::log(static_cast<double>(high)));
    const std::uint64_t max_attempts = target_count * 200 + 10'000;
    std::uint64_t attempts = 0;

    while (values.size() < target_count && attempts < max_attempts) {
        ++attempts;
        const double draw = dist(rng);
        std::uint64_t candidate = 0;
        try {
            candidate = round_to_even_in_range(std::exp(draw), low, high);
        } catch (const std::runtime_error&) {
            continue;
        }
        if (!seen.insert(candidate).second) {
            continue;
        }
        values.push_back(candidate);
    }

    if (values.size() != target_count) {
        throw std::runtime_error("Failed to sample enough unique evens for bucket");
    }
    return values;
}

}  // namespace

std::vector<std::uint64_t> collect_bucket_evens(
    int bucket_id,
    std::uint64_t low,
    std::uint64_t high,
    std::uint64_t target_count,
    bool enumerate,
    std::uint64_t seed) {
    if (enumerate) {
        const auto values = enumerate_evens(low, high);
        if (values.size() != target_count) {
            throw std::runtime_error(
                "Bucket " + std::to_string(bucket_id) + " enumerate count mismatch");
        }
        return values;
    }

    std::mt19937_64 rng(seed + static_cast<std::uint64_t>(bucket_id) * 1'000'003ull);
    return sample_log_uniform(low, high, target_count, rng);
}

std::vector<std::uint64_t> generate_hard_cases(
    std::uint64_t low,
    std::uint64_t high,
    std::uint64_t target_count,
    std::uint64_t seed) {
    std::mt19937_64 rng(seed ^ 0x9E3779B97F4A7C15ull);
    std::unordered_set<std::uint64_t> seen;
    seen.reserve(static_cast<std::size_t>(target_count * 2));
    std::vector<std::uint64_t> values;
    values.reserve(static_cast<std::size_t>(target_count));

    constexpr std::uint64_t kStep = 2ull * 3ull * 5ull * 7ull * 11ull;  // 2310
    const std::uint64_t start = ((low + kStep - 1) / kStep) * kStep;
    std::uniform_int_distribution<std::uint64_t> offset_dist(0, kStep - 2);

    std::uint64_t attempts = 0;
    const std::uint64_t max_attempts = target_count * 500 + 10'000;

    while (values.size() < target_count && attempts < max_attempts) {
        ++attempts;
        std::uint64_t candidate = start + ((attempts * kStep) % ((high - start) / kStep + 1)) * kStep;
        if (candidate >= high) {
            candidate = start + offset_dist(rng);
            if (candidate % 2 != 0) {
                ++candidate;
            }
        }
        if (candidate < low || candidate >= high || candidate % 2 != 0) {
            continue;
        }
        if ((candidate / 2) % 1155 != 0 && (attempts % 3) != 0) {
            continue;
        }
        if (!seen.insert(candidate).second) {
            continue;
        }
        values.push_back(candidate);
    }

    if (values.size() < target_count) {
        const auto fallback = sample_log_uniform(low, high, target_count - values.size(), rng);
        for (const std::uint64_t value : fallback) {
            if (seen.insert(value).second) {
                values.push_back(value);
            }
            if (values.size() == target_count) {
                break;
            }
        }
    }

    if (values.size() != target_count) {
        throw std::runtime_error("Failed to generate enough hard cases");
    }
    return values;
}

}  // namespace goldbach
