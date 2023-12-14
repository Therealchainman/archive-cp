#include <bits/stdc++.h>
using namespace std;


int main() {
    int cycles = 1'000'000'000;
    int n = 14;
    int i = 0;
    for (int j = 0; j < cycles - 183; j++) {
        i = (i + 1) % n;
    }
    cout << i << endl;
}