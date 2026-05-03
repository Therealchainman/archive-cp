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

# Codeforces Round 1087 (Div. 2)

## A. Flip Flops

### Solution 1: sorting, greedy

```cpp
int64 N, C, K;
vector<int> A;

void solve() {
    cin >> N >> C >> K;
    A.assign(N, 0);
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
    }
    sort(A.begin(), A.end());
    for (int x : A) {
        if (x > C) break;
        int64 delta = C - x;
        int64 take = min(delta, K);
        C += take + x;
        K -= take;
    }
    cout << C << endl;
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

## B. Array

### Solution 1: fenwick tree, coordinate compression, counting smaller and greater elements

```cpp
int N;
vector<int> A, C;

template <typename T>
struct FenwickTree {
    vector<T> nodes;
    T neutral;

    FenwickTree() : neutral(T(0)) {}

    void init(int n, T neutral_val = T(0)) {
        neutral = neutral_val;
        nodes.assign(n + 1, neutral);
    }

    void update(int idx, T val) {
        while (idx < (int)nodes.size()) {
            nodes[idx] += val;
            idx += (idx & -idx);
        }
    }

    T query(int idx) {
        T result = neutral;
        while (idx > 0) {
            result += nodes[idx];
            idx -= (idx & -idx);
        }
        return result;
    }

    T query(int left, int right) {
        return right >= left ? query(right) - query(left - 1) : T(0);
    }
};

void solve() {
    cin >> N;
    A.assign(N, 0);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
        C.emplace_back(A[i]);
    }
    sort(C.begin(), C.end());
    C.erase(unique(C.begin(), C.end()), C.end());
    int M = C.size();
    FenwickTree<int> seg;
    seg.init(M);
    vector<int> ans(N, 0);
    for (int i = N - 1; i >= 0; --i) {
        int idx = lower_bound(C.begin(), C.end(), A[i]) - C.begin() + 1;
        ans[i] = max(seg.query(idx - 1), seg.query(idx + 1, M));
        seg.update(idx, 1);
    }
    for (int x : ans) {
        cout << x << " ";
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

## C. Find the Zero

### Solution 1: logic, process of elimination

if you run n queries, for each adjacent pair (1, 2), and so on. If you do not get a 0 anywhere, that means for each query there must have been a nonzero and zero.  So for any of the queries, one of them must be the zero. take (1, 2), and now how do you determine which one is the zero with just a single query? You do n - 1 queries, and leave (1, 2) unqueried. 

Now it will take at most 2 more queries to determine which one is zero
the first pair must be any of these possibilities.
(0, 0)
(x, 0)
(0, x)
So if you query(1, 3) and query(1, 4) and one returns 1, then you know a1 is zero, else it must be the case (x, 0) and a2 is zero.


```cpp
int N;

int query(int i, int j) {
    cout << "?" << " " << i << " " << j << endl;
    cout.flush();
    int resp;
    cin >> resp;
    return resp;
}

void answer(int x) {
    cout << "!" << " " << x << endl;
    cout.flush();
}

void solve() {
    cin >> N;
    for (int i = 4; i <= 2 * N; i += 2) {
        int resp = query(i - 1, i);
        if (resp == -1) exit(0);
        if (resp == 1) {
            answer(i);
            return;
        }
    }
    int resp1 = query(1, 3);
    int resp2 = query(1, 4);
    if (resp1 == 1 || resp2 == 1) {
        answer(1);
    } else {
        answer(2);
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

## D. Ghostfires

### Solution 1: greedy, constructive

It's hard to get the gist of this one, but just know the best way is to pair up greedily into pairs RG, RB, GB.  And then try adding a singleton in front, and that will then uniquely determine a path of arranging all the RG, RB, GB pairs, which you can still arrange all of them.

```cpp
int r, g, b;
string ans;

void solve() {
    cin >> r >> g >> b;
    ans.clear();
    int rg = 0, rb = 0, gb = 0;
    while ((r > 0) + (g > 0) + (b > 0) > 1) {
        if (r <= g && r <= b) {
            gb++, g--, b--;
        } else if (g <= r && g <= b) {
            rb++, r--, b--;
        } else {
            rg++, r--, g--;
        }
    }
    if (r > 0) {
        ans += 'R';
        while (rg > 0) {
            ans += "GR";
            rg--;
        }
        bool flag = false;
        while (rb > 0) {
            ans += "BR";
            rb--;
            flag = true;
        }
        while (gb > 0) {
            if (flag) {
                ans += "BG";
            } else {
                ans += "GB";
            }
            gb--;
        }
    } else if (b > 0) {
        ans += 'B';
        while (rb > 0) {
            ans += "RB";
            rb--;
        }
        bool flag = false;
        while (gb > 0) {
            ans += "GB";
            gb--;
            flag = true;
        }
        while (rg > 0) {
            if (flag) {
                ans += "GR";
            } else {
                ans += "RG";
            }
            rg--;
        }
    } else {
        if (g > 0) {
            ans += 'G';
        }
        while (rg > 0) {
            ans += "RG";
            rg--;
        }
        bool flag = false;
        while (gb > 0) {
            ans += "BG";
            gb--;
            flag = true;
        }
        while (rb > 0) {
            if (flag) {
                ans += "BR";
            } else {
                ans += "RB";
            }
            rb--;
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

# Codeforces Round 1088 (Div. 1 + Div. 2)

## A. Antimedian Deletion

### Solution 1: clever trick

This is basically a trick problem.

The actual values of the array never matter, so the code does not even store them.
The only thing that matters is that we can always answer with the same small deletion size for every position.

Using `2` everywhere is enough, and if `N = 1` then we use `1`.
So every answer is simply

$$
\min(2, N).
$$

That is why the whole solution is just reading the array and printing the same value `N` times.

```cpp
int N;

void solve() {
    cin >> N;
    for (int i = 0; i < N; ++i) {
        int x;
        cin >> x;
    }
    for (int i = 0; i < N; ++i) {
        cout << min(2, N) << " ";
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

## B. Mickey Mouse Constructive

### Solution 1: count divisors

The key quantity is the total imbalance of the array:

$$
\Delta = |x - y|.
$$

Whatever parameter the statement counts as "good", it can only come from a divisor of this total imbalance, so the answer can never be larger than the number of divisors of $\Delta$.

The nice part is that the very simple construction

- first `x` values equal to `1`
- last `y` values equal to `-1`

already reaches that upper bound.

So the constructive part is trivial, and the real work is only counting the divisors of $\Delta$ in

$$
O(\sqrt{\Delta}).
$$

If `x = y`, then $\Delta = 0`, so the divisor counting argument degenerates, but we still have at least one valid construction, which is why the code prints `max(1, ans)`.

```cpp
int x, y;
void solve() {
    cin >> x >> y;
    int ans = 0;
    int delta = abs(x - y);
    for (int x = 1; 1LL * x * x <= delta; ++x) {
        if (delta % x == 0) {
            ans++;
            if (1LL * x * x != delta) ans++;
        }
    }
    ans = max(1, ans);
    cout << ans << endl;
    for (int i = 0; i < x + y; ++i) {
        if (i < x) cout << 1 << ' ';
        else cout << -1 << ' ';
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

## C1. Equal Multisets (Easy Version)

### Solution 1: frequency, array, edge cases

In the easy version, the only positions that can really move are the overlap of the first `K` positions and the last `K` positions:

$$
[N-K,\, K-1].
$$

Everything outside that interval is forced, because it does not belong to the always-rearrangeable part.

So there are two checks:

1. Any specified `B[i]` outside the overlap must already equal `A[i]`.
2. Inside the overlap, only the multiset matters.

The first `freq` array also catches an immediate bad case: if some fixed value appears twice in `B`, then it is impossible.

After that, we just count frequencies inside the overlap:

- `fa[v]` = how many times value `v` appears in `A`
- `fb[v]` = how many times value `v` is already required by fixed entries of `B`

If for some value `v` we need more than we have, meaning

$$
fb[v] > fa[v],
$$

then the answer is `NO`.
Otherwise we can fill the remaining `-1` positions and make it work.

```cpp
int N, K;
vector<int> A, B, freq;

void solve() {
    cin >> N >> K;
    A.assign(N, 0);
    B.assign(N, 0);
    freq.assign(N + 1, 0);
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
    }
    for (int i = 0; i < N; ++i) {
        cin >> B[i];
        if (B[i] == -1) continue;
        freq[B[i]]++;
    }
    if (any_of(freq.begin(), freq.end(), [](int x) { return x > 1; })) {
        cout << "NO" << endl;
        return;
    }
    int l = N - K, r = K - 1;
    for (int i = 0; i < N; ++i) {
        if (B[i] == -1) continue;
        if (i >= l && i <= r) continue; // these are the ones we always rearrange
        if (A[i] != B[i]) {
            cout << "NO" << endl;
            return;
        }
    }
    vector<int> fa(N + 1, 0), fb(N + 1, 0);
    for (int i = l; i <= r; ++i) {
        fa[A[i]]++;
        if (B[i] != -1) fb[B[i]]++;
    }
    for (int i = 0; i <= N; ++i) {
        if (fb[i] > fa[i]) {
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

## C2. Equal Multisets (Hard Version)

### Solution 1: frequency, multisets, positions, residue modulo

The hard version has the same idea, but the movable part is no longer one contiguous interval.
Instead, positions split into residue classes modulo `K`:

$$
i,\ i+K,\ i+2K,\ \dots
$$

Handle each residue class independently.

If the values of `A` on one residue class are not all the same, then that class is rigid.
So every specified value in `B` on that class must already match `A`, otherwise it is impossible immediately.

If a residue class of `A` is constant, then that whole class behaves like one movable token labeled by that value.
The array `cnt` counts how many such flexible classes we have for each value.

Now inspect the fixed values of `B` on the same class:

- if they are all `-1`, the class is still free
- if there is exactly one distinct non-`-1` value `v`, then this class demands one token of value `v`
- if there are multiple distinct fixed values, then this class is no longer flexible, and every fixed position must already match `A`

The array `cnt2` stores these demands.
In the end, for every value `v`, we need enough flexible classes carrying `v`:

$$
cnt[v] \ge cnt2[v].
$$

If this holds for all values, the answer is `YES`; otherwise it is `NO`.

```cpp
int N, K;
vector<int> A, B;

void solve() {
    cin >> N >> K;
    A.assign(N, 0);
    B.assign(N, 0);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    for (int i = 0; i < N; i++) {
        cin >> B[i];
    }
    vector<int> cnt(N + 1, 0), pos(K, -1);
    for (int i = 0; i < K; ++i) {
        pos[i] = A[i];
        for (int j = i; j < N; j += K) {
            if (A[j] != pos[i]) pos[i] = -1;
        }
        if (pos[i] != -1) {
            cnt[pos[i]]++;
            continue;
        }
        for (int j = i; j < N; j += K) {
            if (B[j] != -1 && B[j] != A[j]) {
                cout << "NO" << endl;
                return;
            }
        }
    }
    vector<int> cnt2(N + 1, 0);
    for (int i = 0; i < K; ++i) {
        if (pos[i] == -1) continue;
        set<int> values;
        for (int j = i; j < N; j += K) {
            values.emplace(B[j]);
        }
        values.erase(-1);
        if (values.size() == 1) {
            cnt2[*values.begin()]++;
        }
        if (values.size() < 2) continue;
        for (int j = i; j < N; j += K) {
            if (B[j] != -1 && A[j] != B[j]) {
                cout << "NO" << endl;
                return;
            }
        }
    }
    for (int i = 0; i <= N; ++i) {
        if (cnt[i] < cnt2[i]) {
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

## D. AND Array

### Solution 1: factorials, combinatorics, backwards calculating, frequency of bitmasks

Think about each bit independently.

If some bit appears in exactly $x$ elements of the answer array, then for any chosen subset of size $k$, that bit survives in the bitwise AND iff all $k$ chosen elements come from those $x$ positions.

So that one bit contributes

$$
\binom{x}{k}
$$

to the total value $B_{k-1}$.

Let $S_x$ be the number of bit positions that appear in exactly $x$ elements of the final array.
Then for every $k \in [1, N]$ we get

$$
B_{k-1} = \sum_{x = k}^{N} S_x \binom{x}{k}.
$$

This is an upper-triangular binomial transform, so we can recover the values $S_x$ from right to left.

When we are computing $S_{i+1}$, all values $S_j$ with $j > i + 1$ are already known, so

$$
S_{i+1}
=
B_i - \sum_{j = i + 2}^{N} S_j \binom{j}{i+1}.
$$

That is exactly what the loop is doing:

- start from `val = B[i]`
- subtract every already-known contribution $S_j \binom{j}{i+1}$
- whatever remains must be $S_{i+1}$

The factorial and inverse factorial precomputation is only there so that each

$$
\binom{j}{i+1}
$$

can be computed in $O(1)$ modulo $10^9 + 7$.

After we know all frequencies $S_x$, we still need to build one valid array.
The clean way to think about it is as columns:

- each bit counted in $S_x$ becomes a column of height $x$
- put that bit into the first $x$ elements of the array

Then the number of bits in the $(i+1)$-th element is exactly the number of columns whose height is at least $i+1$, which is

$$
A_i = \sum_{x = i + 1}^{N} S_x.
$$

So the final array is just the suffix sums of $S$.

This also explains the last loop in the code: once the exact frequencies are known, accumulating a suffix reconstructs the answer array directly.

```cpp
const int MOD = 1e9 + 7, MAXN = 1e5 + 5;
int N;
vector<int> B;

int64 inv(int i, int64 m) {
  return i <= 1 ? i : m - (m / i) * inv(m % i, m) % m;
}

vector<int64> fact, inv_fact;

void factorials(int n, int64 m) {
    fact.assign(n + 1, 1);
    inv_fact.assign(n + 1, 0);
    for (int i = 2; i <= n; i++) {
        fact[i] = (fact[i - 1] * i) % m;
    }
    inv_fact.end()[-1] = inv(fact.end()[-1], m);
    for (int i = n - 1; i >= 0; i--) {
        inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % m;
    }
}

int64 choose(int n, int r, int64 m) {
    if (n < r) return 0;
    return (fact[n] * inv_fact[r] % m) * inv_fact[n - r] % m;
}

void solve() {
    cin >> N;
    B.assign(N, 0);
    for (int i = 0; i < N; ++i) {
        cin >> B[i];
    }
    vector<int> S(N + 1, 0); // the frequency of the ith bitmask in the array A;
    vector<int> nonzero; // indices of nonzero elements in S;
    for (int i = N - 1; i >= 0; --i) {
        int val = B[i];
        for (int j : nonzero) {
            int cur = 1LL * S[j] * choose(j, i + 1, MOD) % MOD;
            val -= cur;
            if (val < 0) val += MOD;
        }
        S[i + 1] = val;
        if (S[i + 1]) nonzero.emplace_back(i + 1);
    }
    vector<int> ans(N, 0);
    int suf = 0;
    for (int i = N - 1; i >= 0; --i) {
        suf += S[i + 1];
        ans[i] = suf;
    }
    for (int x : ans) {
        cout << x << ' ';
    }
    cout << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    factorials(MAXN, MOD);
    int T;
    cin >> T;
    while (T--) {
        solve();
    }
    return 0;
}
```

# Codeforces Round 1090 (Div. 4)

## D. The 67th OEIS Problem

### Solution 1: constructive, primes, pairwise gcd pattern

The key observation is that products of consecutive primes give a very clean overlap structure. If we build

$$
1,\ p_1p_2,\ p_2p_3,\ p_3p_4,\ \dots
$$

then every nontrivial element shares exactly one prime factor with its immediate neighbors, and shares no prime factor with elements farther away.

That is why the construction in the code is so short: precompute primes with a sieve, print `1` first, and then print `P[i-1] * P[i]` for each remaining position. Using distinct consecutive primes keeps the factorization unique, so the required gcd behavior is automatic and there is no need for any search or casework.

The sieve fills `P` once, and each test case just outputs the first `N` constructed values.

```cpp
const int MAXN = 2e5 + 5;
int N;
bool primes[MAXN];
vector<int> P;

void sieve(int n) {
    fill(primes, primes + n, true);
    primes[0] = primes[1] = false;
    P.emplace_back(1);
    int p = 2;
    for (int p = 2; p * p <= n; p++) {
        if (primes[p]) {
            for (int i = p * p; i < n; i += p) {
                primes[i] = false;;
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        if (primes[i]) {
            P.emplace_back(i);
        }
    }
}

void solve() {
    cin >> N;
    int cnt = 0;
    vector<int64> ans;
    ans.emplace_back(1);
    for (int i = 1; i < N; ++i) {
        ans.emplace_back(1LL * P[i - 1] * P[i]);
    }
    for (int64 x : ans) {
        cout << x << ' ';
    }
    cout << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    sieve(MAXN);
    int T;
    cin >> T;
    while (T--) {
        solve();
    }
    return 0;
}
```

## E. The 67th XOR Problem

### Solution 1: brute force, xor over all pairs

This one is small enough that we can check every pair directly. The answer only depends on choosing two positions, so the simplest correct approach is to try all

$$
(i, j),\quad i < j
$$

and keep the maximum value of `A[i] ^ A[j]`.

The nested loops do exactly that. `ans` stores the best xor seen so far, and every unordered pair is examined once, so nothing can be missed. The implementation is only `O(N^2)`, but for the intended constraints that is already fast enough.

```cpp
int N;
vector<int> A;

void solve() {
    cin >> N;
    A.assign(N, 0);
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
    }
    int ans = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            ans = max(ans, A[i] ^ A[j]);
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

## F. The 67th Tree Problem

### Solution 1: constructive, root-centered gadgets

The construction treats node `1` as the center and builds the tree out of two simple pieces:

- a length-2 arm `1 - u - (u+1)`, which consumes one unit from both counters
- a direct leaf `1 - u`, which consumes one unit only from the second counter

So the natural greedy move is to use as many length-2 arms as possible first, because that is the only way to spend one unit from both groups at the same time. After that, any leftover amount from the second group can be finished by attaching plain leaves to the root.

The only subtlety is that the root itself already belongs to one of the two required groups, and which group that is depends on the parity of the final tree size. That is why the code subtracts `1` from `X` when `X + Y` is even, and from `Y` otherwise, before building the remaining gadgets.

The vector `edges` stores exactly this construction. If at some point one counter would need to go negative, then no such tree exists and the code prints `No`; otherwise the produced edges give a valid tree immediately.

```cpp
int X, Y;
vector<pair<int, int>> edges;

void solve() {
    cin >> X >> Y;
    int N = X + Y, u = 2;
    edges.clear();
    if (N % 2 == 0) {
        X--;
    } else {
        Y--;
    }
    while (X > 0) {
        edges.emplace_back(1, u);
        edges.emplace_back(u, u + 1);
        X--;
        Y--;
        u += 2;
    }
    if (Y < 0 || X < 0) {
        cout << "No" << endl;
        return;
    }
    while (Y > 0) {
        edges.emplace_back(1, u++);
        Y--;
    }
    cout << "Yes" << endl;
    for (const auto &[u, v] : edges) {
        cout << u << " " << v << endl;
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

## G. The 67th Iteration of Counting is Fun

### Solution 1: counting, process values in increasing order, prefix frequencies

Process the values layer by layer from `0` up to `M - 1`. By the time we handle value `i`, all smaller values are already fixed, so every position with `B[j] = i` must connect to something smaller right next to it. If both neighbors are at least `i`, then there is no way to place this `i`, so the answer is immediately `0`.

After that, there are two cases for a position `j` with value `i`:

- if its best smaller neighbor is exactly `i - 1`, then any previously processed value smaller than `i` can serve, which gives `pref[i - 1]` choices
- if the best smaller neighbor is below `i - 1`, then the newest layer `i - 1` is forced, so the number of choices is exactly the count of value `i - 1`, which is `pref[i - 1] - pref[i - 2]`

The array `pref` stores how many positions have value at most each threshold, and `bucket[i]` stores all indices whose value is exactly `i`. For each layer we multiply the contributions of all positions in that bucket, then multiply that into the global answer modulo `676767677`.

```cpp
const int MOD = 676767677;
int N, M;
vector<int> B;

const int INF = numeric_limits<int>::max();

void solve() {
    cin >> N >> M;
    B.assign(N, 0);
    vector<int> pref(M, 0);
    for (int i = 0; i < N; ++i) {
        cin >> B[i];
        pref[B[i]]++;
    }
    for (int i = 1; i < M; ++i) {
        pref[i] += pref[i - 1];
    }
    vector<vector<int>> bucket(M, vector<int>());
    for (int i = 0; i < N; ++i) {
        bucket[B[i]].emplace_back(i);
    }
    int ans = 1;
    for (int i = 1; i < M; ++i) {
        int cur = 1;
        for (int j : bucket[i]) {
            int neigh = INF;
            if (j > 0) {
                neigh = min(neigh, B[j - 1]);
            }
            if (j + 1 < N) {
                neigh = min(neigh, B[j + 1]);
            }
            if (neigh >= i) {
                cout << 0 << endl;
                return;
            }
            if (neigh == i - 1) {
                cur = 1LL * cur * pref[i - 1] % MOD;
            } else {
                cur = 1LL * cur * (pref[i - 1] - pref[i - 2]) % MOD;
            }
        }
        ans = 1LL * ans * cur % MOD;
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

# Spectral::Cup 2026 Round 1 (Codeforces Round 1094, Div. 1 + Div. 2)

## A. A wonderful Contest

### Solution 1: linear scan

The whole problem reduces to checking whether the array contains the value 100.

```cpp
int N;
vector<int> A;

void solve() {
    cin >> N;
    A.assign(N, 0);
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
    }
    if (any_of(A.begin(), A.end(), [](int x) { return x == 100; })) {
        cout << "Yes" << endl;
        return;
    }
    cout << "No" << endl;
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

## B. Artistic Balance Tree

### Solution 1: parity invariant, greedy, sorting

A symmetric reversal around some center swaps positions:

u - i  <->  u + i

The two indices u - i and u + i always have the same parity.

So an element that starts on an even index can only ever move to an even index.
An element that starts on an odd index can only ever move to an odd index.

That means the problem separates into two independent groups:

even-index elements
odd-index elements

The marked positions also split by parity. If there are evenCnt marked even positions, then we can choose up to evenCnt elements from the even group to become marked. Same for odd.

```cpp
int N, M;
vector<int> A, B;

void solve() {
    cin >> N >> M;
    A.assign(N, 0);
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
    }
    B.assign(M, 0);
    for (int i = 0; i < M; ++i) {
        cin >> B[i];
        B[i]--;
    }
    vector<int> odd, even;
    for (int i = 0; i < N; ++i) {
        if (i % 2 == 0) {
            even.emplace_back(A[i]);
        } else {
            odd.emplace_back(A[i]);
        }
    }
    sort(odd.begin(), odd.end());
    sort(even.begin(), even.end());
    int oddCnt = 0, evenCnt = 0;
    for (int i = 0; i < M; ++i) {
        if (B[i] % 2 == 0) {
            evenCnt++;
        } else {
            oddCnt++;
        }
    }
    while (evenCnt-- && !even.empty()) {
        even.pop_back();
        if (!even.empty() && even.back() <= 0) break;
    }
    while (oddCnt-- && !odd.empty()) {
        odd.pop_back();
        if (!odd.empty() && odd.back() <= 0) break;
    }
    int64 ans = accumulate(odd.begin(), odd.end(), 0LL) + accumulate(even.begin(), even.end(), 0LL);
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

## C. Median Partition

### Solution 1: dp, fenwick tree, coordinate compression, kth order statistic

The algorithm considers partitions of the first i elements.

dp[i][med]

roughly means:

The best number of valid segments using the prefix A[0..i-1], where the last segment has median med.

For every ending position i, the code tries every possible starting position j:

for (int j = i; j > 0; --j)

So the current segment is:

A[j - 1], A[j], ..., A[i - 1]

Only odd-length segments matter, because the median is well-defined as the middle element.

Why coordinate compression is used

The actual array values may be large, negative, or sparse.

So values are compressed:

actual value -> rank among sorted distinct values

This allows the Fenwick tree to count frequencies over ranks from 1 to M.

What the Fenwick tree does

As j moves backward, the segment grows by one element.

The Fenwick tree stores the frequency of values currently inside the segment.

Then this line finds the median rank:

int med = seg.kth((len + 1) / 2);

For an odd-length segment, the median is the (len + 1) / 2-th smallest element.

The Fenwick kth(k) function returns the smallest index whose prefix count is at least k.

In other words, it answers:

What value rank is the kth smallest element in the current segment?

DP transition
dp[i][med] = max(dp[i][med], dp[j - 1][med] + 1);

This means:

If the prefix ending before this segment can be valid with this same median, then append the current segment and increase the segment count by 1.

```cpp
const int INF = numeric_limits<int>::max();
int N;
vector<int> A;

template <typename T>
struct FenwickTree {
    vector<T> nodes;
    T neutral;

    FenwickTree() : neutral(T(0)) {}

    void init(int n, T neutral_val = T(0)) {
        neutral = neutral_val;
        nodes.assign(n + 1, neutral);
    }

    void update(int idx, T val) {
        while (idx < (int)nodes.size()) {
            nodes[idx] += val;
            idx += (idx & -idx);
        }
    }

    int kth(int k) const {
        int n = nodes.size() - 1;
        int idx = 0;

        int step = 1;
        while ((step << 1) <= n) step <<= 1;

        while (step > 0) {
            int next = idx + step;

            if (next <= n && nodes[next] < k) {
                idx = next;
                k -= nodes[next];
            }

            step >>= 1;
        }

        return idx + 1;
    }
};

void solve() {
    cin >> N;
    A.assign(N, 0);
    vector<int> vals;
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
        vals.emplace_back(A[i]);
    }
    sort(vals.begin(), vals.end());
    vals.erase(unique(vals.begin(), vals.end()), vals.end());
    vector<int> compressed(N);
    for (int i = 0; i < N; ++i) {
        compressed[i] = lower_bound(vals.begin(), vals.end(), A[i]) - vals.begin();
    }
    int M = vals.size();
    vector<vector<int>> dp(N + 1, vector<int>(M + 1, -INF));
    for (int i = 0; i <= M; ++i) {
        dp[0][i] = 0;
    }
    for (int i = 1; i <= N; ++i) {
        FenwickTree<int> seg;
        seg.init(M);
        for (int j = i; j > 0; --j) {
            seg.update(compressed[j - 1] + 1, 1);
            int len = i - j + 1;
            if (len % 2 == 0) continue;
            int med = seg.kth((len + 1) / 2);
            if (dp[j - 1][med] == -INF) continue;
            dp[i][med] = max(dp[i][med], dp[j - 1][med] + 1);
        }
    }
    int ans = *max_element(dp[N].begin(), dp[N].end());
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

## D. Permutation Construction

### Solution 1: prefix sum, sorting, greedy

The value of an inversion (i, j) is:

a[i] + a[i + 1] + ... + a[j - 1]

Using prefix sums:

prefix[0] = 0
prefix[k] = a[0] + a[1] + ... + a[k - 1]

Then:

a[i] + ... + a[j - 1] = prefix[j] - prefix[i]

So every inversion contributes:

prefix[j] - prefix[i]

An inversion happens when:

p[i] > p[j]

To maximize total beauty, we want inversions to happen mostly when:

prefix[j] - prefix[i] is positive

That is:

prefix[j] > prefix[i]

So when prefix[i] is smaller, position i should get a larger permutation value.
When prefix[i] is larger, position i should get a smaller permutation value.

```cpp
int N;
vector<int> A;

void solve() {
    cin >> N;
    A.assign(N, 0);
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
    }
    vector<pair<int64, int>> pref;
    int64 psum = 0;
    for (int i = 0; i < N; ++i) {
        pref.emplace_back(psum, i);
        psum += A[i];
    }
    sort(pref.begin(), pref.end());
    vector<int> ans(N);
    for (int i = 0; i < N; ++i) {
        ans[pref[i].second] = N - i;
    }
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