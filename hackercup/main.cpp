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
// string name = "cheeseburger_corollary_2_validation_input.txt";
string name = "cheeseburger_corollary_2_input.txt";


int solve() {
    int res = 0;
    return res;
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
        cout << "Case #" << i << ": " << solve() << endl;
    }
    // int T;
    // cin >> T;
    // for (int i = 1; i <= T ; i++) {
    //     printf("Case #%d", i);
    //     solve();
    // }
    return 0;
}

/*
problem solve

This doesn't work that great if you are reading in a float type, You need to remove the read function and just read inputs using cin >> 
and printf to output it and specify the desired number of decimal places. printf("%.12f\n", p);

g++ "-Wl,--stack,1078749825" nutella.cpp -o main
*/
