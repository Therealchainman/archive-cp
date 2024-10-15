# Codeforces Educational Rounds

# Codeforces Round 170

## C. New Game

### Solution 1:  sliding window, counts, coordinate compression

```cpp
int N, K;
vector<int> arr;
 
void solve() {
    cin >> N >> K;
    arr.resize(N);
    vector<int> val, cnt;
    map<int, int> freq;
    for (int i = 0; i < N; i++) {
        cin >> arr[i];
        freq[arr[i]]++;
    }
    for (auto [key, value] : freq) {
        val.emplace_back(key);
        cnt.emplace_back(value);
    }
    int ans = 0, wcnt = 0;
    for (int l = 0, r = 0; r < val.size(); r++) {
        if (r > 0 && val[r] > val[r - 1] + 1) {
            wcnt = 0;
            l = r;
        }
        wcnt += cnt[r];
        if (r - l + 1 > K) {
            wcnt -= cnt[l];
            l++;
        }
        ans = max(ans, wcnt);
    }
    cout << ans << endl;
}
 
signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    int T;
    cin >> T;
    while (T--) {
        solve();
    }
    return 0;
}
```

## D. Attribute Checks

### Solution 1:  dynamic programming, suffix frequency

```cpp
const int MAXN = 5e3 + 5;
int N, M;
vector<int> arr, dp, ndp;
int str[MAXN], intel[MAXN];
 
void solve() {
    cin >> N >> M;
    arr.resize(N);
    memset(str, 0, sizeof(str));
    memset(intel, 0, sizeof(intel));
    for (int i = 0; i < N; i++) {
        cin >> arr[i];
        if (arr[i] < 0) {
            str[-arr[i]]++;
        } else if (arr[i] > 0) {
            intel[arr[i]]++;
        }
    }
    int ans = 0, cnt = 0;
    dp.assign(M + 1, 0);
    for (int i = 0; i < N; i++) {
        if (arr[i] < 0) {
            str[-arr[i]]--;
        } else if (arr[i] > 0) {
            intel[arr[i]]--;
        } else {
            cnt++;
            ndp.assign(M + 1, 0);
            for (int j = 0; j <= cnt; j++) {
                if (j > 0) {
                    ndp[j] = max(ndp[j], dp[j - 1] + str[j]);
                }
                ndp[j] = max(ndp[j], dp[j] + intel[cnt - j]);
                ans = max(ans, ndp[j]);
            }
            swap(dp, ndp);
        }
    }
 
    cout << ans << endl;
}
 
signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## E. Card Game

### Solution 1:  iterative dynamic programming

```cpp
const int MOD = 998244353;
int N, M;
vector<int> dp1, ndp1, dp, ndp;
 
void solve() {
    cin >> N >> M;
    dp1.assign(M + 1, 0);
    dp1[0] = 1;
    for (int i = 0; i < M; i++) {
        ndp1.resize(M + 1);
        for (int j = 0; j <= M; j++) {
            ndp1[j] = 0;
            if (j > 0) ndp1[j] += dp1[j - 1];
            if (j < M) ndp1[j] += dp1[j + 1];
            ndp1[j] %= MOD;
        }
        swap(dp1, ndp1);
    }
    dp.resize(M + 1);
    for (int i = 0; i <= M; i++) {
        dp[i] = dp1[i];
    }
    for (int i = 1; i < N; i++) {
        ndp.resize(M + 1);
        for (int j = 0; j <= M; j++) {
            ndp[j] = 0;
            for (int k = 0; k <= j; k++) {
                ndp[j - k] += dp[j] * dp1[k];
                ndp[j - k] %= MOD;
            }
        }
        swap(dp, ndp);
    }
    cout << dp[0] << endl;
}
 
signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

### Solution 2:  recursive dynamic programming

```cpp
const int MOD = 998244353;
int N, M;
vector<int> dp1, ndp1;
vector<vector<int>> dp;
 
int dfs(int i, int j) {
    if (i == 0) return dp1[j];
    if (dp[i][j] != -1) return dp[i][j];
    int ans = 0;
    for (int k = 0; j + k <= M; k++) {
        ans += dfs(i - 1, j + k) * dp1[k];
        ans %= MOD;
    }
    return dp[i][j] = ans;
}
 
void solve() {
    cin >> N >> M;
    dp1.assign(M + 1, 0);
    dp1[0] = 1;
    for (int i = 0; i < M; i++) {
        ndp1.resize(M + 1);
        for (int j = 0; j <= M; j++) {
            ndp1[j] = 0;
            if (j > 0) ndp1[j] += dp1[j - 1];
            if (j < M) ndp1[j] += dp1[j + 1];
            ndp1[j] %= MOD;
        }
        swap(dp1, ndp1);
    }
    dp.assign(N, vector<int>(M + 1, -1));
    cout << dfs(N - 1, 0) << endl;
}
 
signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## F. Choose Your Queries

You are given an array of n pairs of integers, where each pair is of the form [(x1, y1), (x2, y2), ... (xn, yn)], with the integers xi and yi in each pair satisfying 1 ≤ xi, yi ≤ N, and N ≤ 300,000. From each pair, you must select exactly one integer. Your goal is to select integers in such a way that minimizes the number of distinct integers that are chosen an odd number of times across all pairs.

You can also think of it as a directed graph, and analysis of indegree parity.

### Solution 1: 

```cpp

```

# Codeforces Round 171

## 

### Solution 1: 

```cpp

```

## 

### Solution 1: 

```cpp

```

## 

### Solution 1: 

```cpp

```