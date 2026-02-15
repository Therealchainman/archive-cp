# Codeforces 2026

# Codeforces Round 1076 (Div. 3)

## E. Product Queries

### Solution 1: dynamic programming, harmonic series

bfs

numbers are made up of x^p, they are taken to a power, where that power is at most 25 let's say, so p <= 25.

dynammic programming over 

It’s a dynamic programming over integers 1..N that computes, for every value x, the minimum number of “available” numbers whose product equals x

the time complexity is governed by a harmonic-series style sum.

1 + 2 + 3 + 4 + ... + N = NlogN

```cpp

const int INF = numeric_limits<int>::max();
int N;
vector<int> A;

void solve() {
    cin >> N;
    A.assign(N, 0);
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
    }
    vector<int> ans(N + 1, INF);
    for (int x : A) {
        ans[x] = 1;
    }
    for (int i = 2; i <= N; ++i) {
        if (ans[i] == INF) continue;
        for (int j = 2; 1LL * i * j <= N; ++j) {
            if (ans[j] == INF) continue;
            ans[i * j] = min(ans[i * j], ans[i] + ans[j]);
        }
    }
    for (int i = 1; i <= N; ++i) {
        cout << (ans[i] < INF ? ans[i] : -1) << " ";
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

## F. Pizza Delivery

### Solution 1: map, dynamic programming

Always use transition to min or max y value for each x value, and pick the minimum. 

```cpp
const int INF = numeric_limits<int>::max();
int N, ax, ay, bx, by;
vector<int> X, Y;

void solve() {
    cin >> N >> ax >> ay >> bx >> by;
    X.assign(N, 0), Y.assign(N, 0);
    for (int i = 0; i < N; ++i) {
        cin >> X[i];
    }
    for (int i = 0; i < N; ++i) {
        cin >> Y[i];
    }
    map<int, int> minMap, maxMap;
    for (int i = 0; i < N; ++i) {
        if (minMap.find(X[i]) == minMap.end()) {
            minMap[X[i]] = INF;
        }
        minMap[X[i]] = min(minMap[X[i]], Y[i]);
        maxMap[X[i]] = max(maxMap[X[i]], Y[i]);
    }
    minMap[bx] = maxMap[bx] = by;
    X.emplace_back(bx);
    sort(X.begin(), X.end());
    X.erase(unique(X.begin(), X.end()), X.end());
    int y1 = ay, y2 = ay;
    int64 a1 = 0, a2 = 0;
    for (const int x : X) {
        int mn = minMap[x], mx = maxMap[x]; // min, max
        int64 c1 = mx - mn + min(abs(y1 - mx) + a1, abs(y2 - mx) + a2);
        int64 c2 = mx - mn + min(abs(y1 - mn) + a1, abs(y2 - mn) + a2);
        y1 = mn, y2 = mx, a1 = c1, a2 = c2;
    }
    int64 ans = min(a1, a2) + abs(bx - ax);
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

## G. Paths in a Tree

### Solution 1: tree, dfs, preorder traversal, query pairs, process of elimination

```cpp
int N;
vector<vector<int>> adj;
vector<int> pre;

int query(int u, int v) {
    cout << "? " << u + 1 << " " << v + 1 << endl;
    cout.flush();
    int resp;
    cin >> resp;
    return resp; 
}

void answer(int u) {
    cout << "! " << u + 1 << endl;
    cout.flush();
}

void dfs(int u, int p = -1) {
    pre.emplace_back(u);
    for (int v : adj[u]) {
        if (v == p) continue;
        dfs(v, u);
    }
}

void solve() {
    cin >> N;
    adj.assign(N, vector<int>());
    for (int i = 0; i < N - 1; ++i) {
        int u, v;
        cin >> u >> v;
        u--, v--;
        adj[u].emplace_back(v);
        adj[v].emplace_back(u);
    }
    pre.clear();
    dfs(0);
    for (int i = 1; i < N; i += 2) {
        int u = pre[i - 1], v = pre[i];
        if (query(u, v)) {
            query(u, u) ? answer(u) : answer(v);
            return;
        }
    }
    answer(pre.back());
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

## H. Remove the Grail Tree

### Solution 1: undirected, directed graph, topological ordering, tree, dfs on tree, 

```cpp
int N;
vector<bool> vis;
vector<int> A, ind, cnt, sz;
vector<vector<int>> adj, dadj, adj1;

void dfs(int u, int p = -1) {
    vis[u] = true;
    sz[u] = 1;
    for (int v : adj[u]) {
        if (v == p) continue;
        dfs(v, u);
        if (sz[v] & 1) {
            dadj[u].emplace_back(v);
            ind[v]++;
        } else {
            dadj[v].emplace_back(u);
            ind[u]++;
        }
        sz[u] += sz[v];
    }
}

void solve() {
    cin >> N;
    A.assign(N, 0);
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
        A[i] %= 2;
    }
    adj.assign(N, vector<int>()); // 1-node undirected graph
    adj1.assign(N, vector<int>()); // edge from 1 -> 0 node
    cnt.assign(N, 0);
    for (int i = 0; i < N - 1; ++i) {
        int u, v;
        cin >> u >> v;
        u--, v--;
        cnt[u] += A[v];
        cnt[v] += A[u];
        if (A[u] && A[v]) {
            adj[u].emplace_back(v);
            adj[v].emplace_back(u);
        }
        if (A[u] && !A[v]) {
            adj1[u].emplace_back(v);
        }
        if (!A[u] && A[v]) {
            adj1[v].emplace_back(u);
        }
    }
    dadj.assign(N, vector<int>());
    vis.assign(N, false);
    sz.assign(N, 0), ind.assign(N, 0);
    for (int i = 0; i < N; ++i) {
        if (!A[i] || vis[i]) continue;
        dfs(i);
        if (sz[i] % 2 == 0) {
            cout << "NO" << endl;
            return;
        }
    }
    vector<int> ans;
    vector<bool> zero(N, false);
    queue<int> q;
    for (int i = 0; i < N; ++i) {
        if (!A[i] && cnt[i] % 2) {
            ans.emplace_back(i);
        } else if (!A[i]) {
            zero[i] = true;
        } else if (A[i] && ind[i] == 0) {
            q.emplace(i);
        }
    }
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        ans.emplace_back(u);
        for (int v : adj1[u]) {
            if (!zero[v]) continue;
            zero[v] = false;
            ans.emplace_back(v);
        }
        for (int v : dadj[u]) {
            if (--ind[v] == 0) {
                q.emplace(v);
            }
        }
    }
    if (ans.size() < N) {
        cout << "NO" << endl;
        return;
    }
    cout << "YES" << endl;
    for (int x : ans) {
        cout << x + 1 << " ";
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

# Codeforces Round 1080 (Div. 3)

## B. Heapify 1

### Solution 1: connected components, looping, sorting

Find connected components by looping over all multiples of 2, sort the values in each component, and check if the resulting array is strictly increasing.

```cpp
int N;
vector<int> A;
vector<bool> vis;
vector<vector<int>> adj, values;

void solve() {
    cin >> N;
    A.assign(N, 0);
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
    }
    adj.assign(N, vector<int>());
    values.assign(N, vector<int>());
    vis.assign(N, false);
    for (int i = 1; i <= N / 2; ++i) {
        if (vis[i - 1]) continue;
        for (int j = i; j <= N; j <<= 1) {
            vis[j - 1] = true;
            adj[i - 1].emplace_back(j - 1);
            values[i - 1].emplace_back(A[j - 1]);
        }
    }
    for (int i = 0; i < N; ++i) {
        if (adj[i].empty()) continue;
        sort(values[i].begin(), values[i].end());
        for (int j = 0; j < adj[i].size(); ++j) {
            A[adj[i][j]] = values[i][j];
        }
    }
    for (int i = 1; i < N; ++i) {
        if (A[i] < A[i - 1]) {
            cout << "NO" << endl;
            return;
        }
    }
    cout << "YES" << endl;
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

## C. Dice Roll Sequence

### Solution 1: grouping, counting

You count the size of consecutive elements that cannot be adjacent, and the answer is the sum of half of each group size.

```cpp
int N;
vector<int> A;

void solve() {
    cin >> N;
    A.assign(N, 0);
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
    }
    int ans = 0, cur = 1;
    for (int i = 1; i < N; ++i) {
        if (7 - A[i] == A[i - 1] || A[i] == A[i - 1]) cur++;
        else {
            ans += cur / 2;
            cur = 1;
        }
    }
    ans += cur / 2;
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

## D. Absolute Cinema

### Solution 1: 

This one was all about figuring out how to perform math on neighboring functions such as take f(i - 1), f(i) and f(i + 1) and can you solve for a_i with some how combining these so they cancel all the other terms.  But you kind of have to see that or think of it.

And that will leave only a1 and an unsolved, but you can solve those now by using the solution for the rest of the a values. 

```cpp
int N;
vector<int64> F;

void solve() {
    cin >> N;
    F.assign(N, 0);
    for (int i = 0; i < N; ++i) {
        cin >> F[i];
    }
    vector<int> ans(N, 0);
    for (int i = 1; i + 1 < N; ++i) {
        ans[i] = ((F[i + 1] - F[i]) - (F[i] - F[i - 1])) / 2;
    }
    ans[N - 1] = F[0];
    ans[0] = F[N - 1];
    for (int i = 1; i + 1 < N; ++i) {
        ans[N - 1] -= i * ans[i];
    }
    ans[N - 1] /= (N - 1);
    for (int i = 1; i + 1 < N; ++i) {
        ans[0] -= (N - i - 1) * ans[i];
    }
    ans[0] /= (N - 1);
    for (int x : ans) {
        cout << x << ' ';
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

## E. Idiot First Search

### Solution 1: binary tree, dfs, dp on tree, postorder traversal, preorder traversal

You can use a postorder traversal to compute the dp values, and then a preorder traversal to compute the answer for each node. The dp value of a node is the sum of dp values of its children plus 2 for each child, (because you will travel down and back up to get to back to this node u) and the answer for a node is the sum of its dp value plus 1 and the answer for its parent (because you will still have to perform all the operations from its parent)

```cpp
const int MOD = 1e9 + 7;
int N;
vector<vector<int>> adj;
vector<int64> dp, ans;

void dfs(int u) {
    for (int v : adj[u]) {
        dfs(v);
        dp[u] += dp[v] + 2;
    }
}

void dfs1(int u, int64 val) {
    int64 nval = (val + dp[u] + 1) % MOD;
    ans[u] = nval;
    for (int v : adj[u]) {
        dfs1(v, nval);
    }
}

void solve() {
    cin >> N;
    adj.assign(N, vector<int>());
    for (int i = 0; i < N; ++i) {
        int u, v;
        cin >> u >> v;
        if (!u && !v) continue;
        --u, --v;
        adj[i].emplace_back(u);
        adj[i].emplace_back(v);
    }
    dp.assign(N, 0);
    ans.assign(N, 0);
    dfs(0);
    dfs1(0, 0);
    for (int x : ans) {
        cout << x << ' ';
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
