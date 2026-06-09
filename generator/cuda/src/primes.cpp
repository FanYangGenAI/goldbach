#include "goldbach/primes.hpp"

#include <vector>

#include "goldbach/constants.hpp"

namespace goldbach {

namespace {

std::vector<std::uint32_t> sieve(std::uint32_t limit) {
    std::vector<bool> is_prime(limit + 1, true);
    if (limit >= 0) {
        is_prime[0] = false;
    }
    if (limit >= 1) {
        is_prime[1] = false;
    }
    for (std::uint32_t number = 2; number * number <= limit; ++number) {
        if (!is_prime[number]) {
            continue;
        }
        for (std::uint32_t multiple = number * number; multiple <= limit; multiple += number) {
            is_prime[multiple] = false;
        }
    }
    std::vector<std::uint32_t> primes;
    for (std::uint32_t number = 2; number <= limit; ++number) {
        if (is_prime[number]) {
            primes.push_back(number);
        }
    }
    return primes;
}

}  // namespace

const std::vector<std::uint32_t>& primes_up_to_max_p1() {
    static const std::vector<std::uint32_t> primes = sieve(kMaxP1);
    return primes;
}

}  // namespace goldbach
