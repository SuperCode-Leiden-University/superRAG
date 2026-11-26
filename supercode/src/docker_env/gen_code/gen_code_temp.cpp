#include <assert.h>
#include <stdio.h>
#include <vector>
#include <algorithm> // For std::sort
#include <cassert>   // For assert

using namespace std;

vector<int> generate_integers(int a, int b) {
    vector<int> result;
    int start = min(a, b);
    int end = max(a, b);

    for (int i = start + 1; i < end; ++i) {
        if (i % 2 == 0) {
            result.push_back(i);
        }
    }

    return result;
}

void check_examples() {
    struct TestCase {
        int a;
        int b;
        vector<int> expected;
    };

    vector<TestCase> test_cases = {
        {2, 8, {2, 4, 6, 8}},
        {8, 2, {2, 4, 6, 8}},
        {10, 14, {}}
    };

    int n_incorrect = 0;

    for (const auto& tc : test_cases) {
        vector<int> result = generate_integers(tc.a, tc.b);
        if (result != tc.expected) {
            printf("Test failed for inputs (%d, %d): expected {", tc.a, tc.b);
            for (size_t i = 0; i < tc.expected.size(); ++i) {
                printf("%d%s", tc.expected[i], i < tc.expected.size() - 1 ? ", " : "");
            }
            printf("}, got {");
            for (size_t i = 0; i < result.size(); ++i) {
                printf("%d%s", result[i], i < result.size() - 1 ? ", " : "");
            }
            printf("}\n");
            ++n_incorrect;
        }
    }

    if (n_incorrect == 0) {
        printf("code passed all the tests\n");
    } else {
        printf("%d incorrect tests\n", n_incorrect);
    }

    assert(n_incorrect == 0);
}

int main() {
    check_examples();
    return 0;
}