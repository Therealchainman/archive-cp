# Special Codeforce Rounds

# Good Bye 2024: 2025 is NEAR

## D. Refined Product Optimality

### Solution 1:  binary search, sorting

1. don't overthink, this one is not that hard as it looks

```cpp
const int MOD = 998244353;
int N, Q;
vector<int> A, B, C, D;

int inv(int i, int m) {
  return i <= 1 ? i : m - (m/i) * inv(m % i, m) % m;
}

void solve() {
    cin >> N >> Q;
    A.assign(N, 0);
    B.assign(N, 0);
    C.assign(N, 0);
    D.assign(N, 0);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
        C[i] = A[i];
    }
    for (int i = 0; i < N; i++) {
        cin >> B[i];
        D[i] = B[i];
    }
    sort(C.begin(), C.end());
    sort(D.begin(), D.end());
    int ans = 1;
    for (int i = 0; i < N; i++) {
        ans = ans * min(C[i], D[i]) % MOD;
    }
    cout << ans << " ";
    while (Q--) {
        int x, y;
        cin >> x >> y;
        x--; y--;
        if (x == 0) {
            int i = upper_bound(C.begin(), C.end(), A[y]) - C.begin() - 1;
            ans = ans * inv(min(C[i], D[i]), MOD) % MOD;
            A[y]++; C[i]++;
            ans = ans * min(C[i], D[i]) % MOD;
        } else {
            int i = upper_bound(D.begin(), D.end(), B[y]) - D.begin() - 1;
            ans = ans * inv(min(C[i], D[i]), MOD) % MOD;
            B[y]++; D[i]++;
            ans = ans * min(C[i], D[i]) % MOD;
        }
        cout << ans << " ";
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

## E. Resourceful Caterpillar Sequence

### Solution 1:  tree, dp on tree, reroot tree, leaf, degrees

1. There is the case when q is on a leaf node, that is precalculated.
1. Then it needs to calculate with dp the case when q is distance of 2 from a leaf node.  

```cpp
int N, ans;
vector<vector<int>> adj;
vector<int> deg, values;

void dfs(int u, int p = -1) {
    bool nonLeafChild = true;
    for (int v : adj[u]) {
        if (deg[v] == 1) nonLeafChild &= false;
        if (v == p) continue;
        dfs(v, u);
        values[u] += values[v];
    }
    if (deg[u] == 1) return;
    values[u] += nonLeafChild;
}

void dfs2(int u, int p = -1, int pval = 0) {
    int cntNonLeaf = 0;
    bool hasLeaf = false;
    for (int v : adj[u]) {
        if (deg[v] == 1) hasLeaf = true;
        else cntNonLeaf++;
        if (v == p) continue;
        int nval = pval + values[u] - values[v];
        dfs2(v, u, nval);
    }
    if (deg[u] == 1) return;
    if (hasLeaf && cntNonLeaf > 0) ans += (cntNonLeaf - 1) * (values[u] + pval);
}

void solve() {
    cin >> N;
    adj.assign(N, vector<int>());
    deg.assign(N, 0);
    values.assign(N, 0);
    for (int i = 0; i < N - 1; i++) {
        int u, v;
        cin >> u >> v;
        u--, v--;
        adj[u].push_back(v);
        adj[v].push_back(u);
        deg[u]++;
        deg[v]++;
    }
    int leaf = 0;
    for (int i = 0; i < N; i++) {
        if (deg[i] == 1) leaf++;
    }
    ans = leaf * (N - leaf);
    dfs(0);
    dfs2(0);
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