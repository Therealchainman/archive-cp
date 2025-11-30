# Helvetic Coding Contest 2024

## A3. Balanced Unshuffle (Hard)

### Solution 1:  

```py

```

## B3. Exact Neighbours (Hard)

### Solution 1:  sort, greedy, constructive algorithm, implementation, zig-zag

```cpp
int n;
vector<int> arr;

void solve() {
    cin >> n;
    arr.resize(n);
    vector<int> wizards(n);
    for (int i = 0; i < n; i++) {
        cin >> arr[i];
        wizards[i] = i;
    }
    sort(wizards.begin(), wizards.end(), [&](const int &a, const int &b) {
        return arr[a] > arr[b];
    });
    sort(arr.begin(), arr.end());
    vector<pair<int, int>> ans(n);
    vector<int> houses(n);
    if (arr[0] == 0) {
        int y = 1;
        houses[wizards.end()[-1]] = wizards.end()[-1];
        ans[wizards[0]] = {1, 1};
        for (int i = 1; i < n; i++) {
            if (arr.end()[-i] > 0) {
                y += (i & 1 ? arr.end()[-i] - 1 : -arr.end()[-i] + 1);
            }
            ans[wizards[i]] = {i + 1, y};
            if (arr.end()[-i] > 0) houses[wizards[i - 1]] = wizards[i];
            else houses[wizards[i - 1]] = wizards[i - 1];
        }
    } else if (adjacent_find(arr.begin(), arr.end()) != arr.end()) { // duplicates
        bool skip = false;
        ans[wizards[0]] = {1, 1};
        int y = 1;
        int x = 1;
        int c = 0;
        for (int i = 1; i <= n; i++) {
            if (i > 1 && !skip && arr.end()[-i] == arr.end()[-i + 1]) {
                skip = true;
                houses[wizards[i - 1]] = wizards[i - 2];
                continue;
            }
            x++;
            c++;
            y += (c & 1 ? arr.end()[-i] - 1 : -arr.end()[-i] + 1);
            if (skip) {
                houses[wizards[i - 1]] = wizards[i - 2];
                ans[wizards[i - 1]] = {x, y};
            } else {
                houses[wizards[i - 1]] = wizards[i];
                ans[wizards[i]] = {x, y};
            }

        }
    } else if (n > 2) {
        ans[wizards[0]] = {1, 1};
        int y = 1;
        houses[wizards.end()[-1]] = wizards.end()[-2];
        for (int i = 1; i < n; i++) {
            if (arr.end()[-i] == 3) {
                y += (i & 1 ? 1 : -1);
                ans[wizards[i + 1]] = {i + 2, y};
                houses[wizards[i]] = wizards[i + 1];
            } else if (arr.end()[-i] == 2) {
                ans[wizards[i - 1]] = {i, y};
                houses[wizards[i - 1]] = wizards[i - 2];
            } else {
                y += (i & 1 ? arr.end()[-i] - 1 : -arr.end()[-i] + 1);
                ans[wizards[i]] = {i + 1, y};
                houses[wizards[i - 1]] = wizards[i];
            }
        }
    } else {
        cout << "NO" << endl;
        return;
    }
    cout << "YES" << endl;
    for (auto [x, y] : ans) {
        cout << x << " " << y << endl;
    }
    for (int x : houses) cout << x + 1 << " ";
    cout << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## C3. Game on Tree (Hard)

### Solution 1:  

```py

```

## D3. Arithmancy (Hard)

### Solution 1:  

```py

```

## E3. Trails (Medium)

### Solution 1:  dynamic programming, matrix exponentiation

```cpp
const int MOD = 1e9 + 7;
vector<vector<int>> transition_matrix, base_matrix;

vector<vector<int>> mat_mul(const vector<vector<int>>& mat1, const vector<vector<int>>& mat2) {
    int rows1 = mat1.size(), cols1 = mat1[0].size();
    int rows2 = mat2.size(), cols2 = mat2[0].size();
    vector<vector<int>> result_matrix(rows1, vector<int>(cols2, 0));
    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            for (int k = 0; k < cols1; k++) {
                result_matrix[i][j] = (result_matrix[i][j] + (long long)mat1[i][k] * mat2[k][j]) % MOD;
            }
        }
    }
    return result_matrix;
}

vector<vector<int>> mat_pow(const vector<vector<int>>& matrix, int power) {
    if (power <= 0) {
        cout << "n must be non-negative integer" << endl;
        return {};
    }
    if (power == 1) return matrix;
    if (power == 2) return mat_mul(matrix, matrix);

    vector<vector<int>> t1 = mat_pow(matrix, power / 2);
    if (power % 2 == 0) {
        return mat_mul(t1, t1);
    }
    return mat_mul(t1, mat_mul(matrix, t1));
}

void solve() {
    int m, n;
    cin >> m >> n;
    vector<int> s(m), l(m);
    for (int i = 0; i < m; i++) cin >> s[i];
    for (int i = 0; i < m; i++) cin >> l[i];

    transition_matrix.assign(m, vector<int>(m));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            transition_matrix[i][j] = s[i] * s[j] + s[i] * l[j] + l[i] * s[j];
        }
    }

    base_matrix.assign(m, vector<int>(1, 0));
    base_matrix[0][0] = 1;
    vector<vector<int>> exponentiated_matrix = mat_pow(transition_matrix, n);
    vector<vector<int>> solution_matrix = mat_mul(exponentiated_matrix, base_matrix);

    int ans = 0;
    for (int i = 0; i < m; i++) {
        ans = (ans + solution_matrix[i][0]) % MOD;
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

## E3. Trails (Hard)

### Solution 1:  linear algebra, associativity of matrix multiplication, dimensionality reduction

```cpp
const int MOD = 1e9 + 7;
vector<vector<int>> transition_matrix, base_matrix, B, C;

vector<vector<int>> mat_mul(const vector<vector<int>>& mat1, const vector<vector<int>>& mat2) {
    int rows1 = mat1.size(), cols1 = mat1[0].size();
    int rows2 = mat2.size(), cols2 = mat2[0].size();
    vector<vector<int>> result_matrix(rows1, vector<int>(cols2, 0));
    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            for (int k = 0; k < cols1; k++) {
                result_matrix[i][j] = (result_matrix[i][j] + (long long)mat1[i][k] * mat2[k][j]) % MOD;
            }
        }
    }
    return result_matrix;
}

vector<vector<int>> mat_pow(const vector<vector<int>>& matrix, int power) {
    if (power <= 0) {
        cout << "n must be non-negative integer" << endl;
        return {};
    }
    if (power == 1) return matrix;
    if (power == 2) return mat_mul(matrix, matrix);

    vector<vector<int>> t1 = mat_pow(matrix, power / 2);
    if (power % 2 == 0) {
        return mat_mul(t1, t1);
    }
    return mat_mul(t1, mat_mul(matrix, t1));
}

void solve() {
    int m, n;
    cin >> m >> n;
    vector<int> s(m), l(m);
    for (int i = 0; i < m; i++) cin >> s[i];
    for (int i = 0; i < m; i++) cin >> l[i];
    B.assign(m, vector<int>(2));
    C.assign(2, vector<int>(m));
    for (int i = 0; i < m; i++) {
        B[i][0] = s[i];
        B[i][1] = l[i];
        C[0][i] = s[i] + l[i];
        C[1][i] = s[i];
    }
    transition_matrix = mat_mul(C, B);
    base_matrix.assign(m, vector<int>(1, 0));
    base_matrix[0][0] = 1;
    vector<vector<int>> solution_matrix = mat_mul(C, base_matrix);
    if (n > 1) {
        vector<vector<int>> exponentiated_matrix = mat_pow(transition_matrix, n - 1);
        solution_matrix = mat_mul(exponentiated_matrix, solution_matrix);
    }
    solution_matrix = mat_mul(B, solution_matrix);
    int ans = 0;
    for (int i = 0; i < m; i++) {
        ans = (ans + solution_matrix[i][0]) % MOD;
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

## G2. Min-Fund Prison (Medium)

### Solution 1:  bridge finding algorithm, dfs, undirected graph, subset sum problem, dynamic programming

```cpp
int n, m, c, timer, comp_count, bridge_count;
vector<vector<int>> adj, comp, dp;
vector<int> disc, low, bridges, comps;

int dfs(int u, int p) {
    int sz = 0;
    disc[u] = low[u] = ++timer;
    for (int v : adj[u]) {
        if (v == p) continue;
        if (!disc[v]) {
            int csz = dfs(v, u);
            if (disc[u] < low[v]) {
                bridges.push_back(csz);
            }
            sz += csz;
            low[u] = min(low[u], low[v]);
        } else {
            low[u] = min(low[u], disc[v]); // back edge, disc[v] because of ap of cycle
        }
    }
    return ++sz;
}

void solve() {
    cin >> n >> m >> c;
    adj.assign(n, vector<int>());
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        u--, v--;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    disc.assign(n, 0);
    low.assign(n, 0);
    comp.assign(n + 1, vector<int>());
    comps.assign(n + 1, 0);
    vector<int> comp_sizes;
    timer = 0;
    comp_count = 0;
    bridge_count = 0;
    // problem is that it might use the bridge and component
    for (int i = 0; i < n; i++) {
        if (!disc[i]) {
            comp_count++;
            bridges.clear();
            int sz = dfs(i, -1);
            comps[sz]++;
            comp_sizes.push_back(sz);
            bridge_count += bridges.size();
            comp[sz].insert(comp[sz].end(), bridges.begin(), bridges.end());
        }
    }
    for (int i = 1; i <= n; i++) {
        sort(comp[i].begin(), comp[i].end());
        comp[i].erase(unique(comp[i].begin(), comp[i].end()), comp[i].end());
    }
    sort(comp_sizes.begin(), comp_sizes.end());
    comp_sizes.erase(unique(comp_sizes.begin(), comp_sizes.end()), comp_sizes.end());
    if (comp_count == 1 && bridge_count == 0) {
        cout << -1 << endl;
        return;
    }
    // ssp begins
    dp.assign(n + 1, vector<int>(2, 0));
    dp[0][0] = 1;
    for (int j : comp_sizes) {
        for (int bs : comp[j]) {
            for (int i = n; i > 0; i--) {
                if (i - bs < 0) break;
                dp[i][1] |= dp[i - bs][0];
            }
        }
        for (int _ = 0; _ < comps[j]; _++) {
            for (int i = n; i > 0; i--) {
                if (i - j < 0) break;
                dp[i][1] |= dp[i - j][1];
                dp[i][0] |= dp[i - j][0];
            }
        }
    }
    int ans = n * n;
    for (int i = 1; i <= n; i++) {
        if (dp[i][0] || dp[i][1]) {
            ans = min(ans, i * i + (n - i) * (n - i));
        }
    }
    ans += (comp_count - 1) * c;
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

## G3. Min-Fund Prison (Hard)

### Solution 1:  optimization of dynamic programming, counting subset sum, reversible, two pointers, dynamic knapsack, reversible knapsack

```cpp
const int MOD = 1e9 + 7;
int n, m, c, timer, comp_count, bridge_count;
vector<vector<int>> adj, comp;
vector<int> disc, low, bridges, comps, dp;

int dfs(int u, int p) {
    int sz = 0;
    disc[u] = low[u] = ++timer;
    for (int v : adj[u]) {
        if (v == p) continue;
        if (!disc[v]) {
            int csz = dfs(v, u);
            if (disc[u] < low[v]) {
                bridges.push_back(csz);
            }
            sz += csz;
            low[u] = min(low[u], low[v]);
        } else {
            low[u] = min(low[u], disc[v]); // back edge, disc[v] because of ap of cycle
        }
    }
    return ++sz;
}

void solve() {
    cin >> n >> m >> c;
    adj.assign(n, vector<int>());
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        u--, v--;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    disc.assign(n, 0);
    low.assign(n, 0);
    comp.assign(n + 1, vector<int>());
    comps.assign(n + 1, 0);
    vector<int> comp_sizes;
    timer = 0;
    comp_count = 0;
    bridge_count = 0;
    // problem is that it might use the bridge and component
    for (int i = 0; i < n; i++) {
        if (!disc[i]) {
            comp_count++;
            bridges.clear();
            int sz = dfs(i, -1);
            comps[sz]++;
            comp_sizes.push_back(sz);
            bridge_count += bridges.size();
            comp[sz].insert(comp[sz].end(), bridges.begin(), bridges.end());
        }
    }
    for (int i = 1; i <= n; i++) {
        sort(comp[i].begin(), comp[i].end());
        comp[i].erase(unique(comp[i].begin(), comp[i].end()), comp[i].end());
    }
    for (int i = 1; i < n; i++) {
        if (comps[i] < 4) continue;
        int take = comps[i] / 2 - 1;
        comps[2 * i] += take;
        comp_sizes.push_back(2 * i);
        comps[i] -= 2 * take;
    }
    if (comp_count == 1 && bridge_count == 0) {
        cout << -1 << endl;
        return;
    }
    dp.assign(n + 1, 0);
    dp[0] = 1;
    int ans = n * n;
    // O(n^2) => O(n*sqrt(n))
    for (int j = 1; j <= n; j++) {
        for (int i = 0; i < comps[j]; i++) {
            for (int c = n; c >= j; c--) {
                dp[c] = (dp[c] + dp[c - j]) % MOD;
            }
        }
    }

    for (int i = 0; i < n; i++) {
        if (dp[i] > 0) ans = min(ans, i * i + (n - i) * (n - i));
    }

    // rollback knapsack for some component sizes
    for (int j = 1; j <= n; j++) {
        if (!comp[j].size()) continue;
        // remove
        for (int c = j; c <= n; c++) {
            dp[c] = (dp[c] - dp[c - j] + MOD) % MOD;
        }
        // replace with bridge split
        int i = 0;
        for (int c = n; c >= 0; c--) {
            if (!dp[c]) continue;
            while (i < comp[j].size() && c + comp[j][i] < n / 2) {
                int x1 = c + comp[j][i];
                int x2 = n - x1;
                ans = min(ans, x1 * x1 + x2 * x2);
                i++;
            }
            if (i == comp[j].size()) i--;
            int x1 = c + comp[j][i];
            int x2 = n - x1;
            ans = min(ans, x1 * x1 + x2 * x2);
        }
        // add
        for (int c = n; c >= j; c--) {
            dp[c] = (dp[c] + dp[c - j]) % MOD;
        }
    }
    ans += (comp_count - 1) * c;
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