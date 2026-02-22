# Constructor Open Cup 2026

# Practice Round

## E. Chocolate Split

### Solution 1: finite arithmetic series, math, formula

```cpp
int K;

int64 calc(int64 n) {
    return n * (n + 1) / 2;
}

void solve() {
    cin >> K;
    K += 2;
    int64 ans = calc(K / 2) + calc((K - 1) / 2);
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

## F. Divisibility Problem

### Solution 1: dfs, backtracking, recursion

```cpp
int N, M, ans;

bool dfs(int i, int rem, int cand) {
    if (i == N) {
        if (rem == 0) ans = cand;
        return rem == 0;
    }
    for (int d = 1; d <= 2; ++d) {
        if (dfs(i + 1, (rem * 10 + d) % M, cand * 10 + d)) return true;
    }
    return false;
}

void solve() {
    cin >> N;
    M = 1;
    for (int i = 0; i < N; ++i) {
        M <<= 1;
    }
    dfs(0, 0, 0);
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

## G. Comics Collection

### Solution 1: principle of inclusion-exclusion

This is a good introduction to the principle of inclusion-exclusion. We can count how many numbers from 1 to N are divisible by 5, 3, and 2, and then use inclusion-exclusion to find how many numbers are divisible by at least one of them. Finally, we can calculate the sum of all these numbers.

```cpp
int N;

void solve() {
    cin >> N;
    int countFive = N / 5;
    int countThree = N / 3 - N / 15;
    int countTwo = N / 2 - N / 6 - N / 10 + N / 30;
    int countOne = N - countTwo - countThree - countFive;
    int64 ans = 5LL * countFive + 3LL * countThree + 2LL * countTwo + 1LL * countOne;
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

## H. Bad Sectors

### Solution 1: scanning and tracking distance from last bad sector, reverse, symmetry

```cpp
int N, K;
string S, ans;

void update() {
    for (int i = 0, d = K + 1; i < N; ++i, ++d) {
        if (S[i] == '*') d = 0;
        if (d <= K) ans[i] = '*';
    }
}

void solve() {
    cin >> N >> K >> S;
    ans.assign(N, '.');
    update();
    reverse(S.begin(), S.end());
    reverse(ans.begin(), ans.end());
    update();
    reverse(ans.begin(), ans.end());
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

## I. Friends at the Cafeteria

### Solution 1: line sweep, sorting, two pointers, fixed sized window, counting active intervals

I think easiest way is to collect all the start and end times, sort them, and then do a line sweep.  Where you define the right endpoint to be that element, and you can calculate the left endpoint, and determine how many intervals currently overlap with it. 

```cpp
int N, M;
vector<int> A, B;

void solve() {
    cin >> N >> M;
    A.assign(N, 0);
    B.assign(N, 0);
    vector<int> events;
    for (int i = 0; i < N; i++) {
        cin >> A[i] >> B[i];
        B[i] += A[i];
        events.emplace_back(A[i]);
        events.emplace_back(B[i]);
    }
    sort(A.begin(), A.end());
    sort(B.begin(), B.end());
    sort(events.begin(), events.end());
    events.erase(unique(events.begin(), events.end()), events.end());
    int ans = 0, cnt = 0, i = 0, j = 0;
    for (int r : events) {
        int l = r - M;
        while (i < N && A[i] <= r) {
            cnt++;
            i++;
        }
        while (j < N && B[j] < l) {
            cnt--;
            j++;
        }
        ans = max(ans, cnt);
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


## J. Math Exam

### Solution 1: greedy feasibility check, binary search

It is feasible if I can use all elements in the array to create sums that are less than or equal to the target.  Basically for the values we will have feasibility looking like FFFTTTT, and you want to return the smallest T. 

```cpp
int N, K;
vector<int> A;

bool possible(int64 target) {
    for (int k = 0, i = 0; k < K; ++k) {
        int64 sum = 0;
        while (i < N && sum + A[i] <= target) {
            sum += A[i++];
        }
        if (i == N) return true;
    }
    return false;
}

void solve() {
    cin >> N >> K;
    A.assign(N, 0);
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
    }
    int64 lo = 0, hi = 1e16;
    while (lo < hi) {
        int64 mid = lo + (hi - lo) / 2;
        if (possible(mid)) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    cout << lo << endl;
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
## K. The Alarm

### Solution 1: directed graph, strongly connected components, condensation, counting sources in the condensed graph, dag, combinatorics, counting subsets, modular arithmetic

I was being dumb it was just SCC condensation and counting number of nodes with indegree equal to 0, because only these can reach all nodes downstream from it. 

```cpp
struct Tower {
    int x, y, p;
    Tower(int x, int y, int p) : x(x), y(y), p(p) {}
};

const int MOD = 998244353;
int N, numScc, cnt;
int64 val;
vector<vector<int>> adj;
vector<int> ind, pre, comp, low;
vector<bool> vis;
stack<int> stk;

bool isEdge(const Tower& a, const Tower& b, int p) {
    int dx = a.x - b.x;
    int dy = a.y - b.y;
    return dx * dx + dy * dy <= p * p;
}

void dfs(int u) {
    if (pre[u] != -1) return;
    pre[u] = cnt;
    low[u] = cnt++;
    stk.emplace(u);
    for (int v : adj[u]) {
        dfs(v);
        low[u] = min(low[u], low[v]);
    }
    if (pre[u] == low[u]) {
        while (true) {
            int v = stk.top();
            stk.pop();
            comp[v] = numScc;
            low[v] = N;
            if (u == v) break;
        }
        numScc++;
    }
}

void solve() {
    cin >> N;
    vector<Tower> towers;
    for (int i = 0; i < N; ++i) {
        int x, y, p;
        cin >> x >> y >> p;
        towers.emplace_back(x, y, p);
    }
    vector<int64> pow2(N + 1, 1);
    for (int i = 1; i <= N; ++i) {
        pow2[i] = pow2[i - 1] * 2 % MOD;
    }
    adj.assign(N, vector<int>());
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            if (isEdge(towers[i], towers[j], towers[i].p)) {
                adj[i].emplace_back(j);
            }
        }
    }
    vis.assign(N, false);
    pre.assign(N, -1);
    low.assign(N, -1);
    comp.assign(N, -1);
    int64 ans = 1;
    for (int i = 0; i < N; ++i) {
        dfs(i);
    }
    vector<int> sz(numScc, 0);
    vector<vector<int>> dag(numScc, vector<int>());
    for (int u = 0; u < N; ++u) {
        int cu = comp[u];
        sz[cu]++;
        for (int v : adj[u]) {
            int cv = comp[v];
            if (cu != cv) {
                dag[cu].emplace_back(cv);
            }
        }
    }
    ind.assign(numScc, 0);
    for (int i = 0; i < numScc; ++i) {
        sort(dag[i].begin(), dag[i].end());
        dag[i].erase(unique(dag[i].begin(), dag[i].end()), dag[i].end());
        for (int v : dag[i]) {
            ind[v]++;
        }
    }
    for (int i = 0; i < numScc; ++i) {
        int64 cand = pow2[sz[i]];
        if (ind[i] == 0) cand--;
        if (cand < 0) cand += MOD;
        ans = ans * cand % MOD;
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

## L. Extended Fibonacci

### Solution 1: 

Finishing it up, it is linear diophantine equation. 

```cpp

```

# Final Round

## Minimum Value Cell

### Solution 1: sorting, binary search

```cpp
const int INF = numeric_limits<int>::max();
int N, Q;
vector<int> A, B, X, Y, sA, sB;
 
int get(int x, const vector<int>& arr) {
    int i = upper_bound(arr.begin(), arr.end(), x) - arr.begin();
    int ans = INF;
    if (i < N) ans = min(ans, abs(arr[i] - x));
    if (i > 0) ans = min(ans, abs(arr[i - 1] - x));
    return ans;
}
 
void solve() {
    cin >> N >> Q;
    A.resize(N);
    B.resize(N);
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
    }
    for (int i = 0; i < N; ++i) {
        cin >> B[i];
    }
    sA.assign(A.begin(), A.end());
    sB.assign(B.begin(), B.end());
    sort(sA.begin(), sA.end());
    sort(sB.begin(), sB.end());
    X.resize(Q);
    Y.resize(Q);
    for (int i = 0; i < Q; ++i) {
        cin >> X[i];
        X[i]--;
    }
    for (int i = 0; i < Q; ++i) {
        cin >> Y[i];
        Y[i]--;
    }
    for (int i = 0; i < Q; ++i) {
        int c1 = get(A[X[i]], sB), c2 = get(B[Y[i]], sA);
        int ans = min(c1, c2);
        cout << ans << endl;
    }
}
 
signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## E. To the Store and Back

### Solution 1: bfs, queue, visited array, directed graph, state space search

The only thing that makes this a little tricky is to know for bfs state you want it to be (node, seen a store), and then you want to keep track of visited for both of those states.  Then you just do a normal bfs, and if you reach the starting node with seen store = true, then you can return the distance.

```cpp
const int INF = numeric_limits<int>::max();
int N, M;
string S;
vector<int> X, Y;
vector<bool> store, vis[2];
vector<vector<int>> adj;
 
void solve() {
    cin >> N >> M;
    X.resize(M);
    Y.resize(M);
    for (int i = 0; i < M; ++i) {
        cin >> X[i];
        X[i]--;
    }
    for (int i = 0; i < M; ++i) {
        cin >> Y[i];
        Y[i]--;
    }
    cin >> S;
    store.assign(N, false);
    for (int i = 0; i < N; ++i) {
        if (S[i] == '1') store[i] = true;
    }
    adj.assign(N, vector<int>());
    for (int i = 0; i < M; ++i) {
        adj[X[i]].emplace_back(Y[i]);
    }
    int src;
    cin >> src;
    src--;
    queue<pair<int, int>> q; // (node, seen store)
    q.emplace(src, store[src]);
    vis[0].assign(N, false);
    vis[1].assign(N, false);
    vis[store[src]][src] = true;
    int ans = 0;
    while (!q.empty()) {
        int sz = q.size();
        while (sz--) {
            auto [u, i] = q.front();
            q.pop();
            if (u == src && i == 1) {
                cout << ans << endl;
                return;
            }
            for (int v : adj[u]) {
                int ni = i;
                if (store[v]) ni = 1;
                if (vis[ni][v]) continue;
                vis[ni][v] = true;
                q.emplace(v, ni);
            }
        }
        ans++;
    }
    cout << -1 << endl;
}
 
signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## F. Exciting Showmatch

### Solution 1: combinatorics, counting even and odd length segments, precomputation of factorials and inverse factorials, modular arithmetic

Consider segments of the array where the difference between adjacent elements is at most 1, and there is a gap between segments.

The observation is that for any even length segment, it doesn't matter which player goes first, and each one has two possible ways, cause either first player goes to red or blue, and that determines the rest of the matching for elements in the even length segment because you want to maximize excitement.

The odd length segments you need to arrange a particular way to get an even number of players in team red and blue, you must put half of the odd length segments with team red and other half in team blue.  You must have even number of these, because that means.  

```cpp
const int MOD = 998244353, MAXN = 3e5 + 5;
int N;
vector<int> A;

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
    A.assign(2 * N, 0);
    for (int i = 0; i < 2 * N; ++i) {
        cin >> A[i];
    }
    A.emplace_back(1e9);
    int64 ans = 1;
    int games = 0, odd = 0;
    for (int i = 1, len = 1; i <= 2 * N; ++i, ++len) {
        if (A[i] > A[i - 1] + 1) {
            if (len % 2 == 0) ans = ans * 2 % MOD;
            else odd++;
            len = 0;
        } else {
            games++;
        }
    }
    ans = ans * choose(odd, odd / 2, MOD) % MOD;
    cout << games << " " << ans << endl;
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

## G. Minimum Inversion Value

### Solution 1: dfs with backtracking, prefix max, greedy

You greedily pick the pair of index where it gives the maximal inversion, cause removing either of those are only option that can make it better. 

Recursion with backtracking works because K is so small, and the depth is at most K, so 2^K

```cpp
const int INF = numeric_limits<int>::max();
int N, K, ans;
vector<int> A;
vector<bool> marked;

void dfs(int k) {
    int pmax = 0;
    if (k == K) {
        int cand = 0;
        for (int i = 0; i < N; ++i) {
            if (marked[i]) continue;
            pmax = max(pmax, A[i]);
            cand = max(cand, pmax - A[i]);
        }
        ans = min(ans, cand);
        return;
    }
    int cand = -1, idx = -1, pi = -1, jdx = -1;
    for (int i = 0; i < N; ++i) {
        if (marked[i]) continue;
        if (A[i] >= pmax) {
            pmax = A[i];
            pi = i;
        }
        int delta = pmax - A[i];
        if (delta >= cand) {
            cand = delta;
            idx = i;
            jdx = pi;
        }
    }
    // try both
    marked[idx] = true;
    dfs(k + 1);
    marked[idx] = false;
    if (jdx != idx) {
        marked[jdx] = true;
        dfs(k + 1);
        marked[jdx] = false;
    }
}

void solve() {
    cin >> N >> K;
    A.assign(N, 0);
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
    }
    marked.assign(N, false);
    ans = INF;
    dfs(0);
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

## H. Corridor

### Solution 1: constructive, pattern, greedy, casework, dynamic programming

There are some patterns and rules you can derive. 

Such as you must have even number of consecutive 1s in top row, and that you cannot have any 1s appear after 0s in the top row. 
And the answer is to start with +-+- pattern, and once you hit 0s, just fill the rest with "-".

For the bottom row, you must have a 1 at every even index, and you should fill those with "-", but you can also have consecutive 1s if they appear consecutively in odd length, and you again switch between "+" and "-" for those segments, always starting with "-" though, so it is "-+-+-" pattern for those segments.

And these segments cannot appear after the 0s in top row. ,

Actually you just need for even length from start to place '+' there else '-'. 

Then with dp check that the conditions are satisfied, if not then return NO.  If they are satisfied, then you can just print the answer you constructed.

```cpp
const int INF = numeric_limits<int>::max();
int N;
string s1, s2;

void solve() {
    cin >> N >> s1 >> s2;
    string ans[2];
    for (int i = 0; i < 2; ++i) {
        ans[i].assign(N, '-');
    }
    for (int i = 0; i < N; ++i) {
        if (i % 2 == 0 && s1[i] == '1') ans[0][i] = '+';
        if (i & 1 && s2[i] == '1') ans[1][i] = '+';
    }
    int dp1 = 0, dp2 = 0;
    for (int i = 0; i < N; ++i) {
        if (dp1 != -INF) dp1 += ans[0][i] == '+' ? 1 : -1;
        if (dp1 != -INF || dp2 != -INF) dp2 = max(dp1, dp2) + (ans[1][i] == '+' ? 1 : -1);
        if (dp1 < 0) dp1 = -INF;
        if (dp2 < 0) dp2 = -INF;
        if (s1[i] == '1' && dp1 < 0) {
            cout << "NO" << endl;
            return;
        }
        if (s2[i] == '1' && dp2 < 0) {
            cout << "NO" << endl;
            return;
        }
        if (dp1 >= 0 && s1[i] == '0') {
            cout << "NO" << endl;
            return;
        }
        if (dp2 >= 0 && s2[i] == '0') {
            cout << "NO" << endl;
            return;
        }
    }
    cout << "YES" << endl;
    for (int i = 0; i < 2; ++i) {
        cout << ans[i] << endl;
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

## I. Invalid Insertion Sort

### Solution 1: local changes, binary search, set


```cpp
int N;
vector<int> L, R;

void solve() {
    cin >> N;
    L.assign(N + 1, 0);
    R.assign(N + 1, 0);
    set<int> S;
    int64 ans = 0, cur = 0;
    for (int i = 0; i < N; i++) {
        int x;
        cin >> x;
        auto it = S.upper_bound(x);
        // l, x, r     
        if (it != S.end()) {
            int r = *it;
            cur -= L[r];
            L[r] = x - 1;
            R[x] = N - r;
            cur += L[r] + R[x];
        }
        if (it != S.begin()) {
            int l = *prev(it);
            cur -= R[l];
            R[l] = N - x;
            L[x] = l - 1;
            cur += R[l] + L[x];
        }
        S.emplace(x);
        ans += cur;
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

## J. Another Road Reform

### Solution 1: undirected graph, 2-edge connected components, finding bridges, bridge trees

If the graph is 2-edge connected, you don't need to add any edges. 
But the problem reduces to how can I add the minimum number of edges to make the graph 2-edge connected. 
I just don't know how to solve such a problem.

A bridge is an edge that if you remove it, the graph becomes disconnected.  So you want to add edges to make sure there are no bridges.

If you construct a bridge tree with just the edges that are bridges, then you get a forest of trees, where each tree you have all edges are bridges.  Then any adjacent bridges can both be fixed by adding one edge.  So the answer is just the number of edges of each bridge tree divided by 2, rounded up.

A trivial case, any bridge tree that consists of two nodes and one edge, is obviously have to be fixed by creating one edge. 

```cpp
int N, M, cnt;
vector<int> pre, low;
vector<vector<int>> adj, badj;
vector<bool> vis;

void dfs(int u, int p = -1) {
    if (pre[u] != -1) return;
    pre[u] = cnt;
    low[u] = cnt++;
    for (int v : adj[u]) {
        if (v == p) continue;
        dfs(v, u);
        low[u] = min(low[u], low[v]);
        if (pre[u] < low[v]) {
            badj[u].emplace_back(v);
            badj[v].emplace_back(u);
        }
    }
}

int dfs1(int u, int p = -1) {
    if (vis[u]) return 0;
    vis[u] = true;
    int res = 1;
    for (int v : badj[u]) {
        if (v == p) continue;
        res += dfs1(v, u);
    }
    return res;
}

void solve() {
    cin >> N >> M;
    adj.assign(N, vector<int>());
    badj.assign(N, vector<int>());
    for (int i = 0; i < M; ++i) {
        int u, v;
        cin >> u >> v;
        --u, --v;
        adj[u].emplace_back(v);
        adj[v].emplace_back(u);
    }
    pre.assign(N, -1);
    low.assign(N, -1);
    cnt = 0;
    dfs(0);
    int ans = 0;
    vis.assign(N, false);
    for (int i = 0; i < N; ++i) {
        if (vis[i]) continue;
        int sz = dfs1(i);
        ans += sz / 2;
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
