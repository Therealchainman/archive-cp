# Bay Area Programming Contest 2024

## 

### Solution 1: 

```py

```

## 

### Solution 1: 

```py

```

## C. Crazy Dance

### Solution 1:  log factorial, dp, combinatorics, combinations, binomial coefficient

This solution gives TLE, but for my skill level this teaches me a lot, not going to go further.  The further optimization is using divide and conquer algorithm cleverly, to compute the dp faster.  This is about O(nm^2), where m equals the maximum any c value will be, which is kind of just a guess you make from my info.  I really only understood up to the O(n^3) solution. 

dp(i, j) = maximum log count of valid ways to arrange dancers, where i is the current sum of c array, and j is the last c picked.  

```cpp
const int MAXN = 40'000 + 2, MAXK = 811;
int N;
long double log_fac[MAXN], dp[MAXN][MAXK];

void log_factorial() {
    log_fac[0] = 0.0;
    for (int i = 1; i < MAXN; i++) {
        log_fac[i] = log_fac[i - 1] + log2(i);
    }
}

long double choose(int n, int k) {
    return log_fac[n] - log_fac[k] - log_fac[n - k];
}

void solve() {
    cin >> N;
    if (N & 1) {
        cout << 0 << endl;
        return;
    }
    for (int i = 1; i <= N; i++) {
        for (int j = min(i, MAXK - 1); j >= 0; j--) {
            for (int k = 0; k <= min(i, MAXK); k++) {
                if (j + k > i) break;
                dp[i][j] = max(dp[i][j], dp[i - j][k] + choose(j + k, j));
            }
        }
    }
    cout << fixed << setprecision(20) << dp[N / 2][0] - (long double)N << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    log_factorial();
    int T = 1;
    while (T--) {
        solve();
    }
    return 0;
}
```

## 

### Solution 1: 

```py

```

## E. Ezra and Experiments

### Solution 1:  postorder dfs, preorder dfs

You only care about how current node affects the root node. 

To get accepted this must be int, not long long.  So it uses 32 bit integers. 

```cpp
const int MAXN = 2e5 + 5;
int N, l;
vector<vector<int>> adj;
int S[MAXN], ans[MAXN], incr[MAXN], decr[MAXN], neut[MAXN];

int aliveness(int s) {
    return max(0, l - abs(l - s));
}

int dfs1(int u, int p) {
    S[u] = 1;
    for (int v: adj[u]) {
        if (v == p) continue;
        S[u] += dfs1(v, u);
    }
    return aliveness(S[u]);
}

void dfs2(int u, int p) {
    if (p == -1) {
        incr[u] = aliveness(S[u] + 1);
        decr[u] = aliveness(S[u] - 1);
        neut[u] = aliveness(S[u]);
        ans[u] = incr[u];
    } else {
        if (aliveness(S[u] + 1) > aliveness(S[u])) {
            incr[u] = incr[p];
        } else if (aliveness(S[u] + 1) < aliveness(S[u])) {
            incr[u] = decr[p];
        } else {
            incr[u] = neut[p];
        }
        if (aliveness(S[u] - 1) < aliveness(S[u])) {
            decr[u] = decr[p];
        } else if (aliveness(S[u] - 1) > aliveness(S[u])) {
            decr[u] = incr[p];
        } else {
            decr[u] = neut[p];
        }
        neut[u] = neut[p];
        ans[u] = incr[u];
    }
    for (int v: adj[u]) {
        if (v == p) continue;
        dfs2(v, u);
    }
}

void solve() {
    cin >> N >> l;
    adj.assign(N, vector<int>());
    for (int i = 0; i < N - 1; i++) {
        int u, v;
        cin >> u >> v;
        u--, v--;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    memset(S, 0, sizeof(S));
    dfs1(0, -1);
    dfs2(0, -1);
    for (int i = 0; i < N; i++) {
        cout << ans[i] << " ";
    }
    cout << endl;
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

## 

### Solution 1: 

```py

```

## G. GCD Spanning Tree

### Solution 1:  maximum spanning tree, math, constrained spanning tree

```py

```

## H. Haphazard Reconstruction

### Solution 1:  pattern

```py
def main():
    n, k = map(int, input().split())
    ans = "NO"
    if k <= n * n:
        if n & 1:
            if k % 4 == 0: ans = "YES"
            elif (k - 1) % 4 == 0: ans = "YES"
        else:
            if k % 4 == 0: ans = "YES"
    print(ans)
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## 

### Solution 1: 

```py

```

## 

### Solution 1: 

```py

```

## 

### Solution 1: 

```py

```

## 

### Solution 1: 

```py

```

## 

### Solution 1: 

```py

```

