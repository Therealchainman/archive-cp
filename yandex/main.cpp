#include <bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'

string S, T;

void solve() {
    cin >> S >> T;
    cout << S << " " << T << endl;
    int ans = 0;
    // for (char ch : T) {
    //     if (S.find(ch) != string::npos) ans++;
    // }
    cout << ans << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    freopen("./input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
    solve();
    return 0;
}


/*
problem solve

You can output floats with using cout << fixed << setprecision(12) << p << endl;

This is to avoid stack overflow error

on linux:
g++ -O2 main.cpp -o main
ulimit -s unlimited in each shell, then run the ./main file
on windows this works. 
g++ "-Wl,--stack=26843546" main.cpp -o main
*/