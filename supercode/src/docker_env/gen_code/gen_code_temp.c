#include <assert.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cassert>

bool has_even_digit(int num) {
    while (num > 0) {
        int digit = num % 10;
        if (digit % 2 == 0) {
            return true;
        }
        num /= 10;
    }
    return false;
}

std::vector<int> unique_digits(std::vector<int> x) {
    std::vector<int> result;
    for (int num : x) {
        if (!has_even_digit(num)) {
            result.push_back(num);
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

void check_examples() {
    std::vector<std::pair<std::vector<int>, std::vector<int>>> test_cases = {
        { {15, 33, 1422, 1}, {1, 15, 33} },
        { {152, 323, 1422, 10}, {} }
    };

    int n_incorrect = 0;

    for (const auto& [input, expected] : test_cases) {
        std::vector<int> result = unique_digits(input);
        if (result != expected) {
            std::cout << "Warning: Input: ";
            for (int num : input) {
                std::cout << num << " ";
            }
            std::cout << "\nExpected Output: ";
            for (int num : expected) {
                std::cout << num << " ";
            }
            std::cout << "\nActual Output: ";
            for (int num : result) {
                std::cout << num << " ";
            }
            std::cout << "\n\n";
            ++n_incorrect;
        }
    }

    if (n_incorrect == 0) {
        std::cout << "Code passed all the tests.\n";
    } else {
        std::cout << n_incorrect << " examples returned the incorrect output.\n";
    }

    // Assert statement to check if n_incorrect == 0
    assert(n_incorrect == 0);
}

int main() {
    check_examples();
    return 0;
}