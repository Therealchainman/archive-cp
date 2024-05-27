


##

### Solution 1:  pascal's identity

```cpp
signed main() {
    int n;
    cin >> n;
    vector<vector<int>> dp(n + 1, vector<int>(n + 1, 0));
    dp[0][0] = 1;
    for (int i = 1; i <= n; i++) {
        dp[i][0] = 1;
        dp[i][i] = 1;
        for (int j = 1; j < i; j++) {
            dp[i][j] = dp[i - 1][j] + dp[i - 1][j - 1];
        }
    }
    int ans = accumulate(dp[n].begin(), dp[n].end(), 0LL);
    cout << ans << endl;
    return 0;
}
```

##

### Solution 1: 

```py
signed main() {
    int n, k;
    cin >> k >> n;
    vector<vector<int>> dp(n + 1, vector<int>(n + 1, 0));
    dp[0][0] = 1;
    for (int i = 1; i <= k; i++) {
        for (int j = 0; j <= n; j++) {
            for (int v = 0; v <= j; v++) {
                // for the ith term, pick v
                dp[i][j] += dp[i - 1][j - v];
            }
        }
    }
    cout << dp[k][n] << endl;
    return 0;
}
```

##

### Solution 1: 

hack, but how to solve in C++ without integer overflow? 

```py
T = int(input())
import math
for _ in range(T):
	n, k = map(int, input().split())
	print(math.comb(n, k))
```

##

### Solution 1: 

grid problem

```py

```


##

### Solution 1: 

This one is hard, reconstruct all possible (n, k) from given m

```py

```