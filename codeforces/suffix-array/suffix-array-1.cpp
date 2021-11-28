#include <bits/stdc++.h>
using namespace std;
/*
O(n^2) solution for suffix array construction.
*/
int main() {
    string s;
    cin >> s;
    int n = s.size();
    vector<int> suffixArray(n+1);
    for (int i = 0; i <= n; ++i) {
        suffixArray[i] = i;
    }
    sort(suffixArray.begin(), suffixArray.end(), [&](int i, int j) {
        return s.substr(i) < s.substr(j);
    });
    for (int i = 0; i <= n; ++i) {
        cout << suffixArray[i] << " ";
    }
}