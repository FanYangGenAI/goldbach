#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "goldbach/sample.hpp"

namespace goldbach {

struct BucketStats {
    int id;
    std::uint64_t low;
    std::uint64_t high;
    std::uint64_t count;
    std::string strategy;
};

struct GenerationOptions {
    std::string output_path;
    std::string manifest_path;
    std::uint64_t seed = 42;
    std::uint64_t sample_limit = 0;
    std::size_t chunk_size = 100'000;
    std::size_t gpu_batch_size = 65'536;
};

struct GenerationResult {
    std::uint64_t samples_written = 0;
    double elapsed_seconds = 0.0;
    std::string sha256;
    std::vector<BucketStats> buckets;
};

class GpuPrimalityTester;

class Pipeline {
public:
    explicit Pipeline(std::size_t gpu_batch_size);
    ~Pipeline();

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    std::vector<GoldbachSample> label_evens(const std::vector<std::uint64_t>& evens);

private:
    std::size_t gpu_batch_size_;
    GpuPrimalityTester* tester_;
};

GenerationResult run_generation(const GenerationOptions& options);

}  // namespace goldbach
