#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace goldbach {

#pragma pack(push, 1)
struct GoldbachSample {
    std::uint64_t even_number;
    std::uint8_t modulo_features[10];
    float prime_density;
    std::uint32_t p1_label;
};
#pragma pack(pop)

static_assert(sizeof(GoldbachSample) == 26, "GoldbachSample must be 26 bytes");

std::vector<std::uint8_t> build_header(std::uint64_t sample_count);
GoldbachSample make_sample(std::uint64_t even_number, std::uint32_t p1_label);

}  // namespace goldbach
