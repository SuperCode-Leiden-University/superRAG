#undef NDEBUG
#include<assert.h>
#include <stdio.h>
#include <vector>
#include <cassert> // Include this header to use assert
using namespace std;

// Function to check if the balance falls below zero at any point
bool below_zero(vector<int> operations) {
    int balance = 0;
    for (int op : operations) {
        balance += op;
        if (balance < 0) {
            return true;
        }
    }
    return false;
}

// Function to check the examples and verify correctness
void check_examples() {
    vector<vector<int>> testCases = {
        {1, 2, 3}, // Expected: false
        {1, 2, -4, 5} // Expected: true
    };
    vector<bool> expectedResults = {
        false,
        true
    };
    int n_incorrect = 0;

    for (size_t i = 0; i < testCases.size(); ++i) {
        bool result = below_zero(testCases[i]);
        if (result != expectedResults[i]) {
            printf("Warning: Test case %zu failed. Expected %s, got %s\n",
                    i, result ? "true" : "false", expectedResults[i] ? "true" : "false");
            n_incorrect++;
        }
    }

    if (n_incorrect == 0) {
        printf("code passed all the tests\n");
    } else {
        printf("Number of incorrect tests: %d\n", n_incorrect);
    }

    // Assert to check if all tests passed
    assert(n_incorrect == 0);
}

int main() {
    check_examples();
    assert (below_zero({}) == false);
    assert (below_zero({1, 2, -3, 1, 2, -3}) == false);
    assert (below_zero({1, 2, -4, 5, 6}) == true);
    assert (below_zero({1, -1, 2, -2, 5, -5, 4, -4}) == false);
    assert (below_zero({1, -1, 2, -2, 5, -5, 4, -5}) == true);
    assert (below_zero({1, -2, 2, -2, 5, -5, 4, -4}) == true);

    return 0;
}