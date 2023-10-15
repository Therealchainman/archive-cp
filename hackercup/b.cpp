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

string name = "carnival_coins_input.txt";

const double SMALL = numeric_limits<double>::min();
vector<vector<double>> prob;
vector<double> dp, expected;

void solve() {
    int N, K;
    double p;
    cin >> N >> K >> p;
    prob.assign(N + 1, vector<double>(N + 1, 0.0));
    dp.assign(N + 1, 0.0);
    expected.assign(N + 1, SMALL);
    prob[0][0] = 1.0;
    for (int i = 1; i <= N; i++ ) { // flip i coins
        for (int j = 0; j <= i; j++) { // exactly j heads
            prob[i][j] = (1 - p) * prob[i - 1][j];
            if (j > 0) prob[i][j] += p * prob[i - 1][j - 1];
        }
    }
    // dp[i] calculates the probability when flipping i coins to get at least K heads
    for (int i = 1; i <= N; i++) {
        for (int j = K; j <= i; j++) {
            dp[i] += prob[i][j];
        }
    }
    expected[0] = 0.0;
    // maximum expected value
    for (int i = 1; i <= N; i++) { // up to i coins flippped
        for (int j = 0; j < i; j++ ) { 
            expected[i] = max(expected[i], expected[j] + dp[i - j]);
        }
    }
    printf("%.12f\n", expected.end()[-1]);
}

int32_t main() {
	ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    string in = "inputs/" + name;
    string out = "outputs/" + name;
    freopen(in.c_str(), "r", stdin);
    freopen(out.c_str(), "w", stdout);
    int T;
    cin >> T;
    for (int i = 1; i <= T ; i++) {
        printf("Case #%d: ", i);
        solve();
    }
    return 0;
}

/*
problem solve

g++ "-Wl,--stack,1078749825" b.cpp -o main
*/
