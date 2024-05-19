# Starters 134

## Prison Escape

### Solution 1:  matrix, grid, dijkstra, flood fill, connected components

```cpp
int N, M;
vector<vector<int>> grid, dist;
vector<bool> vis;

int mat_id(int i, int j) {
    return i * M + j;
}

pair<int, int> mat_ij(int id) {
    return {id / M, id % M};
}

bool in_bounds(int i, int j) {
    return i >= 0 && i < N && j >= 0 && j < M;
}

vector<pair<int, int>> neighborhood(int i, int j) {
    return {{i - 1, j}, {i + 1, j}, {i, j - 1}, {i, j + 1}};
}

void dijkstra() {
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    vis.assign(N * M, false);
    for (int i = 0; i < N; i++) {
        pq.emplace(grid[i][0], mat_id(i, 0));
        pq.emplace(grid[i][M - 1], mat_id(i, M - 1));
    }
    for (int i = 0; i < M; i++) {
        pq.emplace(grid[0][i], mat_id(0, i));
        pq.emplace(grid[N - 1][i], mat_id(N - 1, i));
    }
    while (!pq.empty()) {
        auto [cost, u] = pq.top();
        pq.pop();
        if (vis[u]) continue;
        vis[u] = true;
        auto [i, j] = mat_ij(u);
        dist[i][j] = cost;
        for (auto [ni, nj] : neighborhood(i, j)) {
            if (!in_bounds(ni, nj)) continue;
            if (vis[mat_id(ni, nj)]) continue;
            pq.emplace(cost + grid[ni][nj], mat_id(ni, nj));
        }
    }
}

void solve() {
    cin >> N >> M;
    grid.assign(N, vector<int>(M));
    dist.assign(N, vector<int>(M));
    for (int i = 0; i < N; i++) {
            string row;
            cin >> row;
        for (int j = 0; j < M; j++) {
            grid[i][j] = row[j] - '0';
        }
    }
    dijkstra();
    vis.assign(N * M, false);
    int ans = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            if (grid[i][j] || vis[mat_id(i, j)]) continue;
            ans = max(ans, dist[i][j]);
        }
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

## Count Good RBS

### Solution 1: 

```cpp

```

## Chef and Bit Tree

### Solution 1:  tree, lca, bitwise xor operation, bit manipulation, min bitwise xor pair, by sorting, bit manipulation relation, pigeonhole principle

```cpp
const int INF = 1e5, MAXN = 1e3 + 5;
int N, Q, marked[MAXN];
vector<int> par, dep, A;
vector<vector<int>> adj;

void dfs(int u, int p = -1) {
    par[u] = p;
    for (int v : adj[u]) {
        if (v == p) continue;
        dep[v] = dep[u] + 1;
        dfs(v, u);
    }
}

void solve() {
    cin >> N >> Q;
    par.resize(N);
    dep.assign(N, 0);
    adj.assign(N, vector<int>());
    for (int i = 0; i < N - 1; i++) {
        int u, v;
        cin >> u >> v;
        u--, v--;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    A.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
        A[i] /= 2;
    }
    dfs(0);
    memset(marked, 0, sizeof(marked));
    for (int i = 1; i <= Q; i++) {
        int t, u, v;
        cin >> t >> u >> v;
        if (t == 1) {
            u--;
            A[u] = v / 2;
        } else {
            u--, v--;
            int ans = INF;
            while (u != v) {
                if (dep[u] < dep[v]) swap(u, v);
                if (marked[A[u]] == i) {
                    ans = 0;
                    break;
                } else {
                    marked[A[u]] = i;
                }
                u = par[u];
            }
            if (marked[A[u]] == i) ans = 0;
            else marked[A[u]] = i;
            if (ans != 0) {
                int prv = INF;
                for (int j = 0; j < MAXN; j++) {
                    if (marked[j] == i) {
                        if (prv != INF) ans = min(ans, prv ^ j);
                        prv = j;
                    } 
                }
            }
            cout << ans << endl;
        }
    }
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

## Permutation Cycle Queries

### Solution 1:  prefix sum, permutations, binary search, sort, permutation cycles

```cpp
int N, Q, M;
vector<int> A, B, psum1, psum2, psum3, pprod, sprod;

int sum_(vector<int> &psum, int l, int r) {
    if (l > r) return 0;
    return (psum[r] - (l == 0 ? 0 : psum[l - 1]) + M) % M;
}

void solve() {
    cin >> N >> Q >> M;
    A.resize(N);
    B.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
        B[i] = A[i];
    }
    sort(B.begin(), B.end());
    pprod.assign(N + 1, 1);
    sprod.assign(N + 2, 1);
    for (int i = 1; i <= N; i++) {
        pprod[i] = (pprod[i - 1] * i) % M;
    }
    for (int i = N; i > 0; i--) {
        sprod[i] = (sprod[i + 1] * i) % M;
    }
    psum1.assign(N, 0);
    psum2.assign(N, 0);
    psum3.assign(N, 0);
    for (int i = 0; i < N; i++) {
        psum1[i] = ((B[i] * pprod[i]) % M * sprod[i + 2]) % M; // B[i] / i
        if (i < N - 1) psum3[i] = ((B[i] * pprod[i + 1]) % M * sprod[i + 3]) % M; // B[i] / i + 1
        if (i > 0) {
            psum2[i] = ((B[i] * pprod[i - 1]) % M * sprod[i + 1]) % M; // B[i] / i - 1
            psum1[i] = (psum1[i] + psum1[i - 1]) % M;
            psum2[i] = (psum2[i] + psum2[i - 1]) % M;
            psum3[i] = (psum3[i] + psum3[i - 1]) % M;
        }
    }
    while (Q--) {
        int idx, u, ans, i, j;
        cin >> idx >> u;
        idx--;
        i = lower_bound(B.begin(), B.end(), A[idx]) - B.begin();
        j = upper_bound(B.begin(), B.end(), u) - B.begin();
        if (j > i) {
            j--;
            int s1 = sum_(psum1, 0, i - 1);
            int s2 = sum_(psum2, i + 1, j);
            int s3 = sum_(psum1, j + 1, N - 1);
            int v = (u * pprod[j] % M * sprod[j + 2]) % M;
            ans = (s1 + s2 + v + s3) % M;
        } else {
            int s1 = sum_(psum1, 0, j - 1);
            int s2 = sum_(psum3, j, i - 1);
            int s3 = sum_(psum1, i + 1, N - 1);
            int v = (u * pprod[j] % M * sprod[j + 2]) % M;
            ans = (s1 + v + s2 + s3) % M;
        }
        cout << ans << endl;
    }

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