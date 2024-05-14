# Starters 133

## Too Far For Comfort

### Solution 1:  factorials, modular inverse, combinatorics, prefix sum, calculate nearest occurrence of repeat

```cpp
const int MOD = 998244353;
int N, M;
vector<int> arr, marked, psum;

int inv(int i) {
  return i <= 1 ? i : MOD - (int)(MOD/i) * inv(MOD % i) % MOD;
}

vector<int> fact, inv_fact;

void factorials(int n) {
    fact.assign(n + 1, 1);
    inv_fact.assign(n + 1, 0);
    for (int i = 2; i <= n; i++) {
        fact[i] = (fact[i - 1] * i) % MOD;
    }
    inv_fact.end()[-1] = inv(fact.end()[-1]);
    for (int i = n - 1; i >= 0; i--) {
        inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % MOD;
    }
}

int choose(int n, int r) {
    if (n < r) return 0;
    return (((fact[n] * inv_fact[r]) % MOD) * inv_fact[n - r]) % MOD;
}

int sum_(int l, int r) {
    return psum[r] - (l > 0 ? psum[l - 1] : 0);
}

void solve() {
    cin >> N >> M;
    arr.resize(N);
    marked.assign(M + 1, 0);
    psum.assign(N, 0);
    for (int i = 0; i < N; i++) {
        cin >> arr[i];
        marked[arr[i]] = 1;
        psum[i] = arr[i] > 0 ? 1 : 0;
        if (i > 0) psum[i] += psum[i - 1];
    }
    vector<int> nxt(N + 1, N);
    vector<int> last(M + 1, N);
    for (int i = N - 1; i >= 0; i--) {
        if (arr[i] > 0) {
            nxt[i] = min(nxt[i + 1], last[arr[i]]);
            last[arr[i]] = i;
        } else {
            nxt[i] = nxt[i + 1];
        }
    }
    factorials(max(N, M));
    int start = count(marked.begin() + 1, marked.end(), 1);
    int ans = 0;
    for (int k = max(1LL, start); k <= N; k++) {
        int base = choose(M - start, k - start);
        int mul = 1;
        for (int s = 0; s < N; s += k) {
            int len_ = min(s + k, N) - s;
            int cnt = sum_(s, min(s + k - 1, N - 1));
            int z = len_ - cnt;
            if (nxt[s] < min(s + k, N)) {
                mul = 0;
                break;
            }
            mul = (mul * (fact[z] * choose(k - cnt, z)) % MOD) % MOD;
        }
        ans = (ans + (base * mul) % MOD) % MOD;
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

## Fireworks

### Solution 1:  tree dp, reroot tree, diameter of tree, iterate over sqrt(N) degrees

```cpp
const int MAXN = 1e5 + 5;
int N, x_deg;
vector<vector<int>> adj;
vector<int> deg, mx1, mx2, vis, lpath, trav1, trav2, res;

// calculate mx1 and mx2 for all subtrees
void dfs1(int u, int p = -1) {
    mx1[u] = mx2[u] = 1;
    for (int v : adj[u]) {
        if (v == p) continue;
        dfs1(v, u);
        if (mx1[v] + 1 > mx1[u]) {
            trav2[u] = trav1[u];
            trav1[u] = v;
            mx2[u] = mx1[u];
            mx1[u] = mx1[v] + 1;
        } else if (mx2[v] + 1 > mx2[u]) {
            mx2[u] = mx2[v] + 1;
            trav2[u] = v;
        }
    }
    if (deg[u] == x_deg) {
        mx1[u] = mx2[u] = 0;
    }
}

// reroot the tree to calculate longest path from each node
void dfs2(int u, int p = -1, int mxp = 0) {
    lpath[u] = max(mxp, max(mx1[u], mx2[u]));
    for (int v : adj[u]) {
        if (v == p) continue;
        if (deg[u] == x_deg) {
            dfs2(v, u, 0);
        } else if (trav1[u] == v) {
            dfs2(v, u, max(mxp, mx2[u]) + 1);
        } else if (trav2[u] == v) {
            dfs2(v, u, max(mxp, mx1[u]) + 1);
        } else {
            dfs2(v, u, max(mxp, max(mx1[u], mx2[u])) + 1);
        }
    }
    if (deg[u] == x_deg) lpath[u] = 0;
}

// calculate answer for longest path, star graph from each node with deg[u] == x_deg
void dfs3(int u, int p = -1) {
    for (int v : adj[u]) {
        if (deg[u] == x_deg) {
            res[u] += lpath[v];
        }
        if (v == p) continue;
        dfs3(v, u);
    }
}

void solve() {
    cin >> N;
    adj.assign(N, vector<int>());
    deg.assign(N + 1, 0);
    for (int i = 0; i < N - 1; i++) {
        int u, v;
        cin >> u >> v;
        u--; v--;
        adj[u].push_back(v);
        adj[v].push_back(u);
        deg[u]++; deg[v]++;
    }
    res.assign(N, 1);
    vector<int> degrees;
    for (int i = 0; i < N; i++) {
        if (deg[i] > 0) degrees.push_back(deg[i]);
    }
    sort(degrees.begin(), degrees.end());
    degrees.erase(unique(degrees.begin(), degrees.end()), degrees.end());
    for (int degree : degrees) {
        mx1.assign(N, 0);
        mx2.assign(N, 0);
        lpath.assign(N, 0);
        trav1.assign(N, -1);
        trav2.assign(N, -1);
        x_deg = degree;
        dfs1(0);
        dfs2(0);
        dfs3(0);
    }
    for (int i = 0; i < N; i++) {
        cout << res[i] << " ";
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

```cpp

```