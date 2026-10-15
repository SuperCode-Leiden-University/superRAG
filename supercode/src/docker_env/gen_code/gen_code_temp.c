#include<stdio.h>
#include<vector.h>
#include<assert.h>

bool below_zero(vector<int> operations){
    int balance = 0;
    for (int operation : operations) {
        balance += operation;
        if (balance < 0) {
            printf("true");
            return true;
        }
    }
    printf("false")
    return false;
}

int main() {
    assert(below_zero({1, 2, 3}) == false , f'the correct result is false, but the function returned { below_zero({1, 2, 3}) } instead');
    assert(below_zero({1, 2, -4, 5}) == true , f'the correct result is true, but the function returned { below_zero({1, 2, -4, 5}) } instead');
    assert(below_zero({-1, 2, -3, 4}) == true , f'the correct result is true, but the function returned { below_zero({-1, 2, -3, 4}) } instead');
    assert(below_zero({1, 2, 3, 4, 5, 6}) == false , f'the correct result is false, but the function returned { below_zero({1, 2, 3, 4, 5, 6}) } instead');
    assert(below_zero({0}) == false , f'the correct result is false, but the function returned { below_zero({0}) } instead');
    printf("end of the code");
    return 0;
}
