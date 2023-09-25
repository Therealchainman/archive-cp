#include <bits/stdc++.h>
using namespace std;
#define int long long

inline int read() {
	int x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}

// string name = "cheeseburger_corollary_2_sample_input.txt";
// string name = "cheeseburger_corollary_1_validation_input.txt";
string name = "cheeseburger_corollary_1_input.txt";


void solve(int t) {
    int S = read(), D = read(), K = read();
    int buns = 2 * (S + D);
    int patties = S + 2 * D;
    int decker = min(buns - 1, patties);
    string res = decker >= K ? "YES" : "NO";
    cout << "Case #" << t << ": " << res << endl;
}

int32_t main() {
	ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    string in = "inputs/" + name;
    string out = "outputs/" + name;
    freopen(in.c_str(), "r", stdin);
    freopen(out.c_str(), "w", stdout);
    int T = read();
    for (int i = 1; i <= T ; i++) {
        solve(i);
    }
    return 0;
}
