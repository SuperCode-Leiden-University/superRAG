#include<stdio.h>
#include<vector>
using namespace std;

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

// assert statements for the examples in the description of the function
assert(below_zero({1, 2, 3}) == false, "the correct result is false, but the function returned true instead");
assert(below_zero({1, 2, -4, 5}) == true, "the correct result is true, but the function returned false instead");

// finally print 'end of the code'
cout << "end of the code";