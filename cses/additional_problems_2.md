# Additional Problems 2

## Bouncing Balls Steps

### Solution 1:  modular arithmetic, independent variables, lowest common multiple

1. One key is to identify that the position horizontal or vertical are independent of each other. 
1. And the number of corners you hit is based on the lowest common multiple.

```cpp
int64 N, M, K;

int calc(int n) {
    int v = K / n;
    if (v & 1) {
        return n - (K % n);
    }
    return K % n;
}

void solve() {
    cin >> N >> M >> K;
    int64 a = K / (N - 1); // side 
    int64 b = K / (M - 1); // side
    int64 corners = K / lcm<int64>(N - 1, M - 1); // corners
    int64 cnt = a + b - corners;
    int r = calc(N - 1), c = calc(M - 1);
    cout << r + 1 << " " << c + 1 << " " << cnt << endl;
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

## Book Shop II

### Solution 1: bounded knapsack dp, binary grouping or binary splitting optimization

1. You convert the multiple copies into single copies with some multiple of weight and value, such you can recover any possible number of copies you can take from it.
1. Reduces it to the 0/1 knapsack problem

```cpp
int N, W;
vector<int> ow, ov, counts, weights, values;

void solve() {
    cin >> N >> W;
    ow.resize(N);
    ov.resize(N);
    counts.resize(N);
    for (int i = 0; i < N; ++i) {
        cin >> ow[i];
    }
    for (int i = 0; i < N; ++i) {
        cin >> ov[i];
    }
    for (int i = 0; i < N; ++i) {
        cin >> counts[i];
    }
    for (int i = 0; i < N; ++i) {
        int c = 1;
        while (counts[i] > c) {
            counts[i] -= c;
            weights.emplace_back(c * ow[i]);
            values.emplace_back(c * ov[i]);
            c <<= 1;
        }
        // leftover
        if (counts[i]) {
            weights.emplace_back(counts[i] * ow[i]);
            values.emplace_back(counts[i] * ov[i]);
        }
    }
    int M = weights.size();
    vector<int> dp(W + 1, 0);
    for (int i = 0; i < M; ++i) {
        for (int j = W; j >= weights[i]; --j) {
            dp[j] = max(dp[j], dp[j - weights[i]] + values[i]);
        }
    }
    int ans = *max_element(dp.begin(), dp.end());
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

## 

### Solution 1: 

```cpp

```