# Atcoder Beginner Contest 354

## E - Remove Pairs 

### Solution 1:  dynamic programming, bitmasks, turn based game, minimax algorithm

```cpp
int N, end_mask;
vector<int> F, B;
vector<vector<int>> dp;

bool recurse(int mask, int idx) {
    if (mask == end_mask) return false;
    if (dp[mask][idx] != -1) return dp[mask][idx];
    int win = false;
    for (int i = 0; i < N; i++) {
        if ((mask >> i) & 1) continue;
        for (int j = 0; j < N; j++) {
            if (i == j) continue;
            if ((mask >> j) & 1) continue;
            if (F[i] == F[j] || B[i] == B[j]) win |= recurse(mask | (1 << i) | (1 << j), idx + 1) ^ 1;
        }
    }
    return dp[mask][idx] = win;
}

signed main() {
    cin >> N;
    F.resize(N);
    B.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> F[i] >> B[i];
    }
    dp.assign(1 << N, vector<int>(N, -1));
    end_mask = (1 << N) - 1;
    bool ans = recurse(0, 0);
    cout << (ans ? "Takahashi" : "Aoki") << endl;
    return 0;
}
```

## F - Useless for LIS 

### Solution 1: 

```py

```

## G - Select Strings 

### Solution 1: 

```py

```