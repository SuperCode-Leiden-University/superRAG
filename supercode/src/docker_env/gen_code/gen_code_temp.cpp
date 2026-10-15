#include<stdio.h>
#include<vector>
#include<math.h>
using namespace std;

bool has_close_elements(vector<float> numbers, float threshold) {
    for (size_t i = 0; i < numbers.size(); ++i) {
        for (size_t j = i + 1; j < numbers.size(); ++j) {
            if (abs(numbers[i] - numbers[j]) < threshold) {
                return true;
            }
        }
    }
    return false;
}

// Test cases
assert(has_close_elements({1.0, 2.0, 3.0}, 0.5) == false, f'the correct result is false, but the function returned {has_close_elements({1.0, 2.0, 3.0}, 0.5)} instead');
assert(has_close_elements({1.0, 2.8, 3.0, 4.0, 5.0, 2.0}, 0.3) == true, f'the correct result is true, but the function returned {has_close_elements({1.0, 2.8, 3.0, 4.0, 5.0, 2.0}, 0.3)} instead');