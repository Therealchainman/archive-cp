#include <bits/stdc++.h>


using namespace std;

inline int read()
{
	int x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}

inline long long readll() {
	long long x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}

int main() {
    int n = read();
    vector<long long> numbers(n);
    for (int i = 0; i < n; i++) {
        numbers[i] = readll();
    }
    vector<vector<long long>> dp(n + 1, vector<long long>(n + 1, LONG_LONG_MIN));
    for (int i = 0; i <= n; i++) {
        dp[i][i] = 0;
    }
    for (int len = 1; len <= n; len++) {
        for (int i = 0; i + len <= n; i++) {
            int j = i + len;
            dp[i][j] = max(dp[i][j], numbers[i] - dp[i + 1][j]);
            dp[i][j] = max(dp[i][j], numbers[j - 1] - dp[i][j - 1]);
        }
    }
    long long res = (dp[0][n] + accumulate(numbers.begin(), numbers.end(), 0LL)) / 2;
    cout << res << endl;
}
