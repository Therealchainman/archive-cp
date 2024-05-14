# TJIOI 2024

## J. Tower Upgrades

### Solution 1:  dfs, graph, tree, subtrees, recursion

```cpp
int n, h, ans;
vector<int> C;
vector<vector<int>> adj;

void dfs(int u, int p, int sum) {
    vector<pair<int, int>> level = {{u, p}};
    vector<pair<int, int>> nlevel;
    while (!level.empty()) {
        nlevel.clear();
        int csum = 0;
        bool terminate = false;
        for (auto [u, p] : level) {
            bool leaf = true;
            for (int v : adj[u]) {
                if (v == p) continue;
                nlevel.push_back({v, u});
                csum += C[v] * h;
                leaf = false;
            }
            terminate |= leaf;
        }
        swap(level, nlevel);
        if (terminate) {
            ans += sum;
            break;
        }
        if (csum < sum) {
            for (auto [u, p] : level) {
                dfs(u, p, C[u] * h);
            }
            break;
        }
    }
}

void solve() {
    cin >> n >> h;
    C.resize(n);
    for (int i = 0; i < n; i++) {
        cin >> C[i];
    }
    adj.assign(n, vector<int>());
    for (int i = 0; i < n - 1; i++) {
        int u, v;
        cin >> u >> v;
        u--, v--;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    ans = 0;
    dfs(0, -1, C[0] * h);
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

## I. Feeding Elmo

### Solution 1:  dynamic programming, size of arithmetic subsequence, 

```cpp
int n, d, m;
vector<int> arr;

void solve() {
    cin >> n >> d >> m;
    arr.resize(n);
    for (int i = 0; i < n; i++) {
        cin >> arr[i];
    }
    sort(arr.begin(), arr.end());
    arr.erase(unique(arr.begin(), arr.end()), arr.end());
    int p1 = 0, ans = 0;
    vector<int> dp(arr.size(), 1);
    for (int i = 0; i < arr.size(); i++) {
        while (arr[p1] + d < arr[i]) p1++;
        if (arr[i] == arr[p1] + d) dp[i] = dp[p1] + 1;
        if (dp[i] >= m) ans++;
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

## M. Telephone Fever Dream

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

##

### Solution 1: 

```cpp

```

##

### Solution 1: 

```cpp

```