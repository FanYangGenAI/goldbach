#include "goldbach/pipeline.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "cuda/primality.cuh"
#include "goldbach/buckets.hpp"
#include "goldbach/constants.hpp"
#include "goldbach/primes.hpp"
#include "goldbach/sample.hpp"

namespace goldbach {

namespace {

std::string sha256_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Failed to open file for sha256: " + path);
    }

    static const std::uint32_t kInit[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    static const std::uint32_t kRound[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

    auto rotr = [](std::uint32_t value, std::uint32_t bits) {
        return (value >> bits) | (value << (32 - bits));
    };

    auto process_block = [&](const std::uint8_t block[64], std::uint32_t state[8]) {
        std::uint32_t words[64];
        for (int index = 0; index < 16; ++index) {
            words[index] = (block[index * 4] << 24) | (block[index * 4 + 1] << 16) |
                           (block[index * 4 + 2] << 8) | block[index * 4 + 3];
        }
        for (int index = 16; index < 64; ++index) {
            const std::uint32_t s0 = rotr(words[index - 15], 7) ^ rotr(words[index - 15], 18) ^ (words[index - 15] >> 3);
            const std::uint32_t s1 = rotr(words[index - 2], 17) ^ rotr(words[index - 2], 19) ^ (words[index - 2] >> 10);
            words[index] = words[index - 16] + s0 + words[index - 7] + s1;
        }

        std::uint32_t a = state[0];
        std::uint32_t b = state[1];
        std::uint32_t c = state[2];
        std::uint32_t d = state[3];
        std::uint32_t e = state[4];
        std::uint32_t f = state[5];
        std::uint32_t g = state[6];
        std::uint32_t h = state[7];

        for (int index = 0; index < 64; ++index) {
            const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const std::uint32_t ch = (e & f) ^ ((~e) & g);
            const std::uint32_t temp1 = h + s1 + ch + kRound[index] + words[index];
            const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temp2 = s0 + maj;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }

        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;
    };

    std::uint32_t state[8];
    std::memcpy(state, kInit, sizeof(state));

    std::uint64_t total_bytes = 0;
    std::array<std::uint8_t, 64> block{};
    while (true) {
        input.read(reinterpret_cast<char*>(block.data()), 64);
        const std::streamsize read_bytes = input.gcount();
        if (read_bytes <= 0) {
            break;
        }
        total_bytes += static_cast<std::uint64_t>(read_bytes);
        if (read_bytes == 64) {
            process_block(block.data(), state);
            continue;
        }

        std::vector<std::uint8_t> tail(block.begin(), block.begin() + read_bytes);
        tail.push_back(0x80);
        while ((tail.size() % 64) != 56) {
            tail.push_back(0);
        }
        const std::uint64_t total_bits = total_bytes * 8ull;
        for (int index = 7; index >= 0; --index) {
            tail.push_back(static_cast<std::uint8_t>((total_bits >> (8 * index)) & 0xFF));
        }
        for (std::size_t offset = 0; offset < tail.size(); offset += 64) {
            process_block(tail.data() + offset, state);
        }
        break;
    }

    if (total_bytes == 0) {
        std::array<std::uint8_t, 64> empty{};
        empty[0] = 0x80;
        process_block(empty.data(), state);
    }

    std::ostringstream output;
    for (int index = 0; index < 8; ++index) {
        output << std::hex << std::setfill('0') << std::setw(8) << state[index];
    }
    return output.str();
}

void write_manifest(
    const std::string& path,
    const GenerationResult& result,
    std::uint64_t seed,
    const std::string& file_name) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Failed to write manifest: " + path);
    }
    output << "{\n";
    output << "  \"phase\": \"phase2\",\n";
    output << "  \"version\": 1,\n";
    output << "  \"sample_count\": " << result.samples_written << ",\n";
    output << "  \"file\": \"" << file_name << "\",\n";
    output << "  \"sha256\": \"" << result.sha256 << "\",\n";
    output << "  \"seed\": " << seed << ",\n";
    output << "  \"hard_case_count\": " << kHardCaseCount << ",\n";
    output << "  \"buckets\": [\n";
    for (std::size_t index = 0; index < result.buckets.size(); ++index) {
        const auto& bucket = result.buckets[index];
        output << "    {\"id\": " << bucket.id
               << ", \"range\": [" << bucket.low << ", " << bucket.high << "]"
               << ", \"count\": " << bucket.count
               << ", \"strategy\": \"" << bucket.strategy << "\"}";
        output << (index + 1 < result.buckets.size() ? ",\n" : "\n");
    }
    output << "  ]\n";
    output << "}\n";
}

std::vector<std::uint64_t> build_bucket18_evens(
    std::uint64_t low,
    std::uint64_t high,
    std::uint64_t bucket_target,
    std::uint64_t seed) {
    const std::uint64_t hard_count = std::min<std::uint64_t>(kHardCaseCount, bucket_target);
    auto evens = generate_hard_cases(low, high, hard_count, seed);
    std::unordered_set<std::uint64_t> seen(evens.begin(), evens.end());

    if (bucket_target > hard_count) {
        const std::uint64_t need = bucket_target - hard_count;
        auto random_part = collect_bucket_evens(18, low, high, need * 2, false, seed + 180017);
        for (const std::uint64_t value : random_part) {
            if (seen.insert(value).second) {
                evens.push_back(value);
            }
            if (evens.size() >= bucket_target) {
                break;
            }
        }
    }

    if (evens.size() < bucket_target) {
        auto filler = collect_bucket_evens(
            18, low, high, bucket_target - evens.size() + 1000, false, seed + 990011);
        for (const std::uint64_t value : filler) {
            if (seen.insert(value).second) {
                evens.push_back(value);
            }
            if (evens.size() >= bucket_target) {
                break;
            }
        }
    }

    if (evens.size() != bucket_target) {
        throw std::runtime_error("Failed to assemble bucket 18 evens");
    }
    return evens;
}

}  // namespace

Pipeline::Pipeline(std::size_t gpu_batch_size)
    : gpu_batch_size_(gpu_batch_size), tester_(new GpuPrimalityTester(gpu_batch_size)) {}

Pipeline::~Pipeline() {
    delete tester_;
}

std::vector<GoldbachSample> Pipeline::label_evens(const std::vector<std::uint64_t>& evens) {
    const auto& primes = primes_up_to_max_p1();
    std::vector<std::uint32_t> labels(evens.size(), 0);
    std::vector<std::uint8_t> done(evens.size(), 0);

    for (const std::uint32_t p1 : primes) {
        std::vector<std::uint64_t> candidates;
        std::vector<std::size_t> mapping;
        candidates.reserve(evens.size());
        mapping.reserve(evens.size());

        for (std::size_t index = 0; index < evens.size(); ++index) {
            if (done[index]) {
                continue;
            }
            const std::uint64_t even = evens[index];
            const std::uint64_t p2 = even - p1;
            if (p2 < p1) {
                throw std::runtime_error("No Goldbach pair found for even=" + std::to_string(even));
            }
            candidates.push_back(p2);
            mapping.push_back(index);
        }

        if (candidates.empty()) {
            break;
        }

        std::vector<std::uint8_t> results(candidates.size(), 0);
        for (std::size_t offset = 0; offset < candidates.size(); offset += gpu_batch_size_) {
            const int batch = static_cast<int>(std::min(gpu_batch_size_, candidates.size() - offset));
            tester_->test(candidates.data() + offset, results.data() + offset, batch);
        }

        for (std::size_t result_index = 0; result_index < results.size(); ++result_index) {
            if (!results[result_index]) {
                continue;
            }
            const std::size_t even_index = mapping[result_index];
            if (!done[even_index]) {
                labels[even_index] = p1;
                done[even_index] = 1;
            }
        }

        bool all_done = true;
        for (const std::uint8_t flag : done) {
            if (!flag) {
                all_done = false;
                break;
            }
        }
        if (all_done) {
            break;
        }
    }

    std::vector<GoldbachSample> samples;
    samples.reserve(evens.size());
    for (std::size_t index = 0; index < evens.size(); ++index) {
        if (!done[index]) {
            throw std::runtime_error("Failed to label even=" + std::to_string(evens[index]));
        }
        samples.push_back(make_sample(evens[index], labels[index]));
    }
    return samples;
}

GenerationResult run_generation(const GenerationOptions& options) {
    Pipeline pipeline(options.gpu_batch_size);
    GenerationResult result;
    const auto started = std::chrono::steady_clock::now();

    std::ofstream output(options.output_path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("Failed to open output file: " + options.output_path);
    }

    const auto header = build_header(0);
    output.write(reinterpret_cast<const char*>(header.data()), static_cast<std::streamsize>(header.size()));

    const std::uint64_t total_target = options.sample_limit == 0 ? kTotalSamples : options.sample_limit;

    for (const auto& spec : bucket_specs()) {
        if (result.samples_written >= total_target) {
            break;
        }

        std::uint64_t bucket_target = spec.target_count;
        if (result.samples_written + bucket_target > total_target) {
            bucket_target = total_target - result.samples_written;
        }

        std::cout << "Bucket " << spec.id << ": preparing " << bucket_target << " evens\n";

        std::vector<std::uint64_t> evens;
        if (spec.id == 18) {
            evens = build_bucket18_evens(spec.low, spec.high, bucket_target, options.seed);
        } else {
            evens = collect_bucket_evens(
                spec.id, spec.low, spec.high, bucket_target, spec.enumerate, options.seed);
        }

        for (std::size_t offset = 0; offset < evens.size(); offset += options.chunk_size) {
            const std::size_t end = std::min(offset + options.chunk_size, evens.size());
            std::vector<std::uint64_t> chunk(evens.begin() + static_cast<std::ptrdiff_t>(offset),
                                             evens.begin() + static_cast<std::ptrdiff_t>(end));
            const auto samples = pipeline.label_evens(chunk);
            for (const auto& sample : samples) {
                output.write(reinterpret_cast<const char*>(&sample), sizeof(sample));
            }
            result.samples_written += samples.size();
            std::cout << "  written " << result.samples_written << " / " << total_target << "\n";
        }

        result.buckets.push_back(
            BucketStats{spec.id, spec.low, spec.high, evens.size(), spec.enumerate ? "enumerate" : "log_uniform"});
    }

    output.seekp(0, std::ios::beg);
    const auto final_header = build_header(result.samples_written);
    output.write(reinterpret_cast<const char*>(final_header.data()),
                 static_cast<std::streamsize>(final_header.size()));
    output.close();

    result.elapsed_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    result.sha256 = sha256_file(options.output_path);

    const std::size_t slash = options.output_path.find_last_of("/\\");
    const std::string file_name =
        slash == std::string::npos ? options.output_path : options.output_path.substr(slash + 1);
    write_manifest(options.manifest_path, result, options.seed, file_name);
    return result;
}

}  // namespace goldbach
