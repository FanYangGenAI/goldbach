#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "goldbach/constants.hpp"
#include "goldbach/pipeline.hpp"

namespace {

void print_usage() {
    std::cout
        << "Usage: goldbach_generate [options]\n"
        << "  --output <path>      Output .bin path (default: ../../data/goldbach_train_v1.bin)\n"
        << "  --manifest <path>    Manifest JSON path\n"
        << "  --seed <n>           RNG seed (default: 42)\n"
        << "  --limit <n>          Generate at most n samples (0 = full 100M)\n"
        << "  --chunk-size <n>     Evens per labeling chunk (default: 100000)\n"
        << "  --gpu-batch <n>      GPU primality batch size (default: 65536)\n";
}

std::string arg_value(int argc, char** argv, int index, const std::string& fallback) {
    if (index + 1 >= argc) {
        throw std::runtime_error("Missing value for " + std::string(argv[index]));
    }
    return argv[index + 1];
}

}  // namespace

int main(int argc, char** argv) {
    goldbach::GenerationOptions options;
    options.output_path = "../../data/goldbach_train_v1.bin";
    options.manifest_path = "../../data/goldbach_manifest_v1.json";

    try {
        for (int index = 1; index < argc; ++index) {
            const std::string arg = argv[index];
            if (arg == "--help" || arg == "-h") {
                print_usage();
                return 0;
            }
            if (arg == "--output") {
                options.output_path = arg_value(argc, argv, index, options.output_path);
                ++index;
            } else if (arg == "--manifest") {
                options.manifest_path = arg_value(argc, argv, index, options.manifest_path);
                ++index;
            } else if (arg == "--seed") {
                options.seed = std::stoull(arg_value(argc, argv, index, "42"));
                ++index;
            } else if (arg == "--limit") {
                options.sample_limit = std::stoull(arg_value(argc, argv, index, "0"));
                ++index;
            } else if (arg == "--chunk-size") {
                options.chunk_size = static_cast<std::size_t>(
                    std::stoull(arg_value(argc, argv, index, "100000")));
                ++index;
            } else if (arg == "--gpu-batch") {
                options.gpu_batch_size = static_cast<std::size_t>(
                    std::stoull(arg_value(argc, argv, index, "65536")));
                ++index;
            } else {
                throw std::runtime_error("Unknown argument: " + arg);
            }
        }

        std::cout << "Goldbach CUDA generator (phase2)\n";
        std::cout << "  output:     " << options.output_path << "\n";
        std::cout << "  manifest:   " << options.manifest_path << "\n";
        std::cout << "  target:     "
                  << (options.sample_limit == 0 ? goldbach::kTotalSamples : options.sample_limit)
                  << " samples\n";

        const auto result = goldbach::run_generation(options);
        std::cout << "\nGeneration complete.\n";
        std::cout << "  samples: " << result.samples_written << "\n";
        std::cout << "  sha256:  " << result.sha256 << "\n";
        std::cout << "  elapsed: " << result.elapsed_seconds << "s\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << "\n";
        return 1;
    }
}
