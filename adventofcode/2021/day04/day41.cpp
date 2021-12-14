#include <bits/stdc++.h>
using namespace std;
/*
part 1
*/
int convBinaryToDecimal(string& s) {
    int n = s.size(), dec = 0;
    for (int i = 0; i < n; i++) {
        dec += ((s[i] - '0')*(1<<(n-i-1)));
    }
    return dec;
}
int main() {
    freopen("inputs/input.txt", "r", stdin);
}
