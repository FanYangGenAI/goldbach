#include "goldbach/sample.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "goldbach/constants.hpp"

namespace goldbach {

namespace {

std::uint32_t crc32_update(std::uint32_t crc, const std::uint8_t* data, std::size_t size) {
    crc = ~crc;
    for (std::size_t index = 0; index < size; ++index) {
        crc ^= data[index];
        for (int bit = 0; bit < 8; ++bit) {
            const std::uint32_t mask = static_cast<std::uint32_t>(-(crc & 1u));
            crc = (crc >> 1u) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}

}  // namespace

std::vector<std::uint8_t> build_header(std::uint64_t sample_count) {
    std::vector<std::uint8_t> header(kHeaderSize, 0);
    std::memcpy(header.data() + 0, &kMagic, sizeof(kMagic));
    std::memcpy(header.data() + 4, &kVersion, sizeof(kVersion));
    std::memcpy(header.data() + 8, &sample_count, sizeof(sample_count));
    const std::uint64_t feature_flags = 0;
    std::memcpy(header.data() + 16, &feature_flags, sizeof(feature_flags));
    const std::uint32_t crc = crc32_update(0, header.data(), 24);
    std::memcpy(header.data() + 24, &crc, sizeof(crc));
    return header;
}

GoldbachSample make_sample(std::uint64_t even_number, std::uint32_t p1_label) {
    GoldbachSample sample{};
    sample.even_number = even_number;
    for (std::size_t index = 0; index < 10; ++index) {
        sample.modulo_features[index] =
            static_cast<std::uint8_t>(even_number % kSmallPrimes[index]);
    }
    sample.prime_density = static_cast<float>(1.0 / std::log(static_cast<double>(even_number)));
    sample.p1_label = p1_label;
    return sample;
}

}  // namespace goldbach
