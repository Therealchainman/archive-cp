# WWPIT 2024 Advanced

## War

### Solution 1:  Prefix Tree, sorted array, base 2 representation, maximizing bitwise and operation

```cpp
const int BITS = 18;
int N, M;
vector<int> arr;

// bit variant of prefix tree

int length(int l, int r) {
    return r - l;
}

// node structure for prefix tree
struct Node {
    int children[2];
    vector<int> indices;
    void init(vector<int>& arr) {
        memset(children, 0, sizeof(children));
        swap(arr, indices);
    }
};
struct PrefixTree {
    vector<Node> tree;
    void init(int n) {
        vector<int> nums(N);
        iota(nums.begin(), nums.end(), 0);
        Node root;
        root.init(nums);
        tree.push_back(root);
        build();
    }
    void build() {
        stack<int> stk, nstk;
        stk.push(0);
        for (int b = BITS; b >= 0; b--) {
            while (!stk.empty()) {
                int cur = stk.top();
                stk.pop();
                vector<int> zero, one;
                for (int idx : tree[cur].indices) {
                    if ((arr[idx] >> b) & 1) {
                        one.push_back(idx);
                    } else {
                        zero.push_back(idx);
                    }
                }
                if (!zero.empty()) {
                    Node root;
                    root.init(zero);
                    tree[cur].children[0] = tree.size();
                    nstk.push(tree.size());
                    tree.push_back(root);
                }
                if (!one.empty()) {
                    Node root;
                    root.init(one);
                    tree[cur].children[1] = tree.size();
                    nstk.push(tree.size());
                    tree.push_back(root);
                }
            }
            swap(nstk, stk);
        }
    }
    int search(int l, int r, int k) {
        vector<int> level, next_level;
        level.push_back(0);
        int res = 0;
        for (int b = BITS; b >= 0; b--) {
            next_level.clear();
            int cnt = 0;
            for (int cur : level) {
                int nxt = tree[cur].children[1];
                if (nxt) {
                    int s, e;
                    e = upper_bound(tree[nxt].indices.begin(), tree[nxt].indices.end(), r) - tree[nxt].indices.begin();
                    s = lower_bound(tree[nxt].indices.begin(), tree[nxt].indices.end(), l) - tree[nxt].indices.begin();
                    cnt += length(s, e);
                    next_level.push_back(nxt);
                }
                if (cnt >= k) break;
            }
            if (cnt < k) {
                for (int cur : level) {
                    int nxt = tree[cur].children[0];
                    if (nxt) {
                        next_level.push_back(nxt);
                    }
                }
            } else {
                res |= (1 << b);
            }
            swap(next_level, level);
        }
        return res;
    }
};

void solve() {
    cin >> N >> M;
    PrefixTree pretree;
    arr.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> arr[i];
    }
    pretree.init(N);
    for (int i = 0; i < M; i++) {
        int l, r, k;
        cin >> l >> r >> k;
        l--, r--;
        int ans = pretree.search(l, r, k);
        cout << ans << endl;
    }
}

signed main() {
    solve();
    return 0;
}
```

## Quantum Tunneling

### Solution 1:  undirected graph, expectation value, degrees, matrix, matrix exponentiation, modular inverse, probability

Still getting TLE for few two test cases in subtask 1

```cpp
const int MOD = 1e9 + 7;
int N, M, E, T;
vector<vector<int>> transition_matrix, base_matrix, adj;

int inv(int i) {
  return i <= 1 ? i : MOD - (long long)(MOD/i) * inv(MOD % i) % MOD;
}

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
    cin >> N >> M >> E >> T;
    vector<int> deg(N, 0);
    adj.assign(N, vector<int>());
    for (int i = 0; i < M; i++) {
        int u, v;
        cin >> u >> v;
        u--; v--;
        deg[u]++;
        deg[v]++;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    transition_matrix.assign(N, vector<int>(N, 0));
    for (int u = 0; u < N; u++) {
        for (int v : adj[u]) {
            transition_matrix[v][u] = inv(deg[u]);
        }
    }
    base_matrix.assign(N, vector<int>(1, 0));
    for (int i = 0; i < E; i++) {
        int u;
        cin >> u;
        u--;
        base_matrix[u][0]++;
    }
    vector<vector<int>> exponentiated_matrix = mat_pow(transition_matrix, T);
    vector<vector<int>> solution_matrix = mat_mul(exponentiated_matrix, base_matrix);

    for (int i = 0; i < N; i++) {
        cout << solution_matrix[i][0] << " ";
    }
    cout << endl;
}

signed main() {
    solve();
    return 0;
}
```

## AP Chemistry

### Solution 1:  binary search, dfs, tarjan's bridge finding algorithm, connected graph

```cpp
const int INF = 1e9;
int N, M, timer;
vector<bool> vis;
vector<pair<int, int>> edges;
vector<int> times, disc, low;
vector<vector<int>> adj;

bool dfs(int u, int p) {
    vis[u] = true;
    disc[u] = low[u] = ++timer;
    for (int v : adj[u]) {
        if (v == p) continue;
        if (!disc[v]) {
            if (dfs(v, u)) return true;
            if (disc[u] < low[v]) {
                return true;
            }
            low[u] = min(low[u], low[v]);
        } else {
            low[u] = min(low[u], disc[v]); // back edge, disc[v] because of ap of cycle
        }
    }
    return false;
}

bool possible(int target) {
    adj.assign(N, vector<int>());
    disc.assign(N, 0);
    low.assign(N, 0);
    vis.assign(N, false);
    timer = 0;
    for (int i = 0; i < M; i++) {
        if (times[i] >= target) {
            auto [u, v] = edges[i];
            adj[u].push_back(v);
            adj[v].push_back(u);
        }
    }
    bool is_bridge = dfs(0, -1);
    // if not all vertex in single component, then return false
    for (int i = 0; i < N; i++) {
        if (!vis[i]) return false;
    }
    return !is_bridge; // if there is a bridge, then return false
}

void solve() {
    cin >> N >> M;
    edges.resize(M);
    times.resize(M);
    for (int i = 0; i < M; i++) {
        int u, v, t;
        cin >> u >> v >> t;
        u--, v--;
        edges[i] = {u, v};
        times[i] = t;
    }
    int lo = 0, hi = INF;
    while (lo < hi) {
        int mid = lo + (hi - lo + 1) / 2;
        if (possible(mid)) lo = mid;
        else hi = mid - 1;
    }
    cout << lo << endl;
}

signed main() {
    solve();
    return 0;
}
```

## Power Outage

### Solution 1:  dynamic programming, sorting, events, sort by end points, minimize the cost to reach a point

```cpp
const int INF = 1e18;
int N, M;

void solve() {
    cin >> N >> M;
    vector<int> dp(N, INF);
    vector<tuple<int, int, int>> events(M);
    for (int i = 0; i < M; i++) {
        int u, v, c;
        cin >> u >> v >> c;
        u--; v--;
        events[i] = {u, v, c};
    }
    sort(events.begin(), events.end(), [&](const tuple<int, int, int>& a, const tuple<int, int, int>& b) {
        return get<1>(a) < get<1>(b);
    });
    for (auto [s, e, c] : events) {
        dp[e] = min(dp[e], (s > 0 ? dp[s - 1]: 0) + c);
    }
    if (dp.end()[-1] == INF) cout << -1 << endl;
    else cout << dp.end()[-1] << endl;
}

signed main() {
    solve();
    return 0;
}
```

## AP Physics

### Solution 1: 

```cpp
#define x first
#define y second
int side(pair<int,int>& a, pair<int,int>& b, pair<int,int>& c){
    return (c.y-a.y)*(b.x-a.x)-(c.x-a.x)*(b.y-a.y);
}

for(int i = 0; i<n; i++) for(int j = i+1; j<n; j++) for(int k = i+1; k<n; k++)
    if(j != k && side(p[i],p[j],p[k])>0) for(int l = i+1; l<n; l++) if(l != j && l != k)
        if(side(p[j],p[k],p[l])>0 && side(p[k],p[l],p[i])>0 && side(p[l],p[i],p[j])>0)
```

## Congestion (Hard)

### Solution 1:  depth array, set, binary search, dfs, tree

```cpp
int N, cnt;
vector<vector<int>> adj;
vector<set<int>> subtree_depths;
vector<int>depth, subtree_id, ans;

int dfs(int u, int p) {
    subtree_id[u] = cnt;
    depth[u] = depth[p] + 1;
    int max_depth = depth[u];
    for (int v : adj[u]) {
        if (v == p) continue;
        max_depth = max(max_depth, dfs(v, u));
    }
    return max_depth;
}

void solve() {
    cin >> N;
    adj.assign(N, vector<int>());
    for (int i = 0; i < N - 1; i++) {
        int u, v;
        cin >> u >> v;
        u--, v--;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    cnt = 0;
    depth.assign(N, 0);
    subtree_id.resize(N);
    for (int u : adj[0]) {
        subtree_depths.push_back({});
        int max_depth = dfs(u, 0);
        for (int i = 1; i <= max_depth + 1; i++) {
            subtree_depths[cnt].insert(i);
        }
        cnt++;
    }
    ans.assign(cnt, 0);
    int mx = 0;
    cout << 0 << " ";
    for (int i = 1; i < N; i++) {
        int id = subtree_id[i];
        auto it = subtree_depths[id].lower_bound(depth[i]);
        if (it == subtree_depths[id].end()) {
            ans[id]++;
        } else {
            ans[id] = max(ans[id], *it);
            subtree_depths[id].erase(it);
        }
        mx = max(mx, ans[id]);
        cout << mx << " ";
    }
    cout << endl;
}

signed main() {
    solve();
    return 0;
}
```

## Mirrors and Lasers

### Solution 1:  grid, dijkstra, backtracking, direction

```cpp
struct State {
    int r, c, d, cost;
    bool operator<(const State& other) const {
        if (r != other.r) return r < other.r;
        if (c != other.c) return c < other.c;
        if (d != other.d) return d < other.d;
        return cost < other.cost;
    }
};

struct stateComp {
    bool operator()(const State& a, const State& b) const {
        return a.cost > b.cost;
    }
};

const char EMPTY = '_', BACKWARDS_LEANING = '\\', FORWARDS_LEANING = '/';
const int INF = 1e9;
int N;
string row;
vector<vector<char>> mat;
vector<vector<array<int, 4>>> dp;
priority_queue<State, vector<State>, stateComp> maxheap;
map<State, State> parent;

bool in_bounds(int r, int c) {
    return r >= 0 && r < N && c >= 0 && c < N;
}

map<int, pair<int, int>> VECTORIZE = {
    {0, {0, 1}},
    {1, {1, 0}},
    {2, {0, -1}},
    {3, {-1, 0}}
};

map<pair<int, int>, int> INV_VECTORIZE = {
    {{0, 1}, 0},
    {{1, 0}, 1},
    {{0, -1}, 2},
    {{-1, 0}, 3}
};

void solve() {
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
    cin >> N;
    mat.resize(N, vector<char>(N));
    for (int i = 0; i < N; i++) {
        cin >> row;
        for (int j = 0; j < N; j++) {
            mat[i][j] = row[j];
        }
    }
    dp.resize(N, vector<array<int, 4>>(N, {INF, INF, INF, INF}));
    maxheap.push({0, 0, 0, 0});
    State last_state;
    while (!maxheap.empty()) {
        State s = maxheap.top();
        maxheap.pop();
        last_state = s;
        if (s.r == N - 1 && s.c == N - 1) break;
        if (s.cost > dp[s.r][s.c][s.d]) continue;
        dp[s.r][s.c][s.d] = s.cost;
        if (mat[s.r][s.c] == BACKWARDS_LEANING || mat[s.r][s.c] == EMPTY) {
            auto [dc, dr] = VECTORIZE[s.d];
            int nr = s.r + dr, nc = s.c + dc;
            if (in_bounds(nr, nc)) {
                int nd = INV_VECTORIZE[{dr, dc}];
                int ncost = s.cost + (mat[s.r][s.c] == EMPTY ? 1 : 0);
                if (ncost < dp[nr][nc][nd]) {
                    dp[nr][nc][nd] = ncost;
                    State nstate = {nr, nc, nd, ncost};
                    maxheap.push(nstate);
                    parent[nstate] = s;
                }
            }
        }
        if (mat[s.r][s.c] == FORWARDS_LEANING || mat[s.r][s.c] == EMPTY) {
            auto [dc, dr] = VECTORIZE[s.d];
            dc = -dc; dr = -dr;
            int nr = s.r + dr, nc = s.c + dc;
            if (in_bounds(nr, nc)) {
                int nd = INV_VECTORIZE[{dr, dc}];
                int ncost = s.cost + (mat[s.r][s.c] == EMPTY ? 1 : 0);
                if (ncost < dp[nr][nc][nd]) {
                    dp[nr][nc][nd] = ncost;
                    State nstate = {nr, nc, nd, ncost};
                    maxheap.push(nstate);
                    parent[nstate] = s;
                }
            }
        }
        if (mat[s.r][s.c] == EMPTY) {
            auto [dr, dc] = VECTORIZE[s.d];
            int nr = s.r + dr, nc = s.c + dc;
            if (in_bounds(nr, nc) && s.cost < dp[nr][nc][s.d]) {
                State nstate = {nr, nc, s.d, s.cost};
                dp[nr][nc][s.d] = s.cost;
                maxheap.push(nstate);
                parent[nstate] = s;
            }
        }
    }
    // backtrack
    while (parent.find(last_state) != parent.end()) {
        auto [ndr, ndc] = VECTORIZE[last_state.d];
        auto [dr, dc] = VECTORIZE[parent[last_state].d];
        int r = parent[last_state].r, c = parent[last_state].c;
        if (dr == ndc && dc == ndr) {
            mat[r][c] = BACKWARDS_LEANING;
        } else if (dr == -ndc && dc == -ndr) {
            mat[r][c] = FORWARDS_LEANING;
        }
        last_state = parent[last_state];
    } 
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++) {
            cout << mat[r][c];
        }
        cout << endl;
    }
    cout << endl;
}

signed main() {
    solve();
    return 0;
}
```

## Minimum XOR

### Solution 1: 

```cpp

```

## Mountain Climbing

### Solution 1:  binary search, arithmetic progression, math

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