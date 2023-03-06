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
    int n = read(), x = read();
    int mod = 1e9 + 7;
    vector<int> dp(x + 1, 0);
    dp[0] = 1;
    vector<int> coins;
    for (int i = 0; i < n; i++) {
        int c = read();
        coins.push_back(c);
    }
    for (int i = 0; i < n; i++) {
        for (int coin_sum = coins[i]; coin_sum <= x; coin_sum++) {
            if (coins[i] > coin_sum) continue;
            dp[coin_sum] = (dp[coin_sum] + dp[coin_sum - coins[i]]) % mod;
        }
    }
    cout << dp[x] << endl;
    return 0;
}