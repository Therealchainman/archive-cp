#include <bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'
#define x first
#define y second

string base = "substantial_losses";
string name = base + "_sample_input.txt";
// string name = base + "_validation_input.txt";
// string name = base + "_input.txt";

const int M = 998244353;
int W, G, L;

void solve() {
    cin >> W >> G >> L;
    int v = (2LL * L + 1) % M;
    int ans = ((W - G) * v) % M;
    cout << ans << endl;
}

signed main() {
	ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    string in = "inputs/" + name;
    string out = "outputs/" + name;
    freopen(in.c_str(), "r", stdin);
    freopen(out.c_str(), "w", stdout);
    int T;
    cin >> T;
    for (int i = 1; i <= T ; i++) {
        cout << "Case #" << i << ": ";
        solve();
    }
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
