# UTCP Spring 2024 Open Contest

## 

### Solution 1: 

```py

```

## 

### Solution 1: 

```py

```

## I. Record Compression

### Solution 1:  unbounded knapsack problem with O(n* sqrt(n)) with the constraints

```cpp
const int MAXN = 2e5 + 5;
int N, M;
int items[MAXN];
vector<int> values, weights, dp;

void solve() {
    cin >> N >> M;
    memset(items, 0, sizeof(items));
    for (int i = 0; i < N; i++) {
        int v;
        string s;
        cin >> s >> v;
        items[s.size()] = max(items[s.size()], v);
    }
    for (int i = 1; i < MAXN; i++) {
        if (!items[i]) continue;
        weights.push_back(i);
        values.push_back(items[i]);
    }
    int V = values.size();
    dp.assign(M + 1, 0);
    for (int cap = 0; cap <= M; cap++) {
        for (int i = 0; i < V; i++) {
            if (cap < weights[i]) break;
            dp[cap] = max(dp[cap], dp[cap - weights[i]] + values[i]);
        }
    }
    cout << dp.end()[-1] << endl;
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