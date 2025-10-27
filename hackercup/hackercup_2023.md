# Meta Hacker Cup 2023

# Practice Round

## Cheeseburger Corollary 1

```cpp
void solve(int t) {
    int S = read(), D = read(), K = read();
    int buns = 2 * (S + D);
    int patties = S + 2 * D;
    int decker = min(buns - 1, patties);
    string res = decker >= K ? "YES" : "NO";
    cout << "Case #" << t << ": " << res << endl;
}

int32_t main() {
	ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    string in = "inputs/" + name;
    string out = "outputs/" + name;
    freopen(in.c_str(), "r", stdin);
    freopen(out.c_str(), "w", stdout);
    int T = read();
    for (int i = 1; i <= T ; i++) {
        solve(i);
    }
    return 0;
}
```

## Cheeseburger Corollary 2

```cpp
void solve(int t) {
    int A = read(), B = read(), C = read();
    int res = 0LL;
    res = max(res, C / A);
    res = max(res, 2 * (C / B) - 1);
    if (C >= A) {
        res = max(res, 2 * ((C - A) / B) + 1);
    }
    if (C >= 2 * A) {
        res = max(res, 2 * ((C - 2 * A) / B) + 2);
    }
    cout << "Case #" << t << ": " << res << endl;
}

int32_t main() {
	ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    string in = "inputs/" + name;
    string out = "outputs/" + name;
    freopen(in.c_str(), "r", stdin);
    freopen(out.c_str(), "w", stdout);
    int T = read();
    for (int i = 1; i <= T ; i++) {
        solve(i);
    }
    return 0;
}

```

## Dim Sum Delivery

### Solution 1:  ad hoc

```cpp
void solve(int t) {
    int R = read(), C = read(), A = read(), B = read();
    string res = R > C ? "YES" : "NO";
    cout << "Case #" << t << ": " << res << endl;
}

int32_t main() {
	ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    string in = "inputs/" + name;
    string out = "outputs/" + name;
    freopen(in.c_str(), "r", stdin);
    freopen(out.c_str(), "w", stdout);
    int T = read();
    for (int i = 1; i <= T ; i++) {
        solve(i);
    }
    return 0;
}
```

## Two Apples a Day

### Solution 1:  two pointers + 3 candidate Ks to check + math 

Use observation that the summation of 2*N elements should be divisible by N.

```cpp
vector<int> arr;
int N;

bool check(int x, int k) {
    if (x <= 0) return false;
    int left = 0, right = 2 * N - 2;
    bool used = false;
    while (left < right) {
        if (arr[left] + arr[right] != k && !used) {
            if (arr[left] + x == k) {
                left++;
                used = true;
            } else if (x + arr[right] == k) {
                right--;
                used = true;
            } else {
                return false;
            }
        }
        else if (arr[left] + arr[right] != k) {
            return false;
        }
        left++;
        right--;
    }
    if (left == right && (arr[left] + x != k || used)) return false;
    return true;
}

int solve() {
    N = read();
    int M = 2 * N - 1;
    arr.resize(M);
    for (int i = 0; i < M; i++) {
        arr[i] = read();
    }
    int res = LLONG_MAX;
    if (N == 1) {
        return 1;
    }
    sort(arr.begin(), arr.end());
    int S = accumulate(arr.begin(), arr.end(), 0LL);
    vector<int> vals{arr[0] + arr.end()[-2], arr[0] + arr.end()[-1], arr[1] + arr.end()[-1]};
    for (int K : vals) {
        int x = K * N - S;
        if (check(x, K)) res = min(res, x);
    }
    if (res == LLONG_MAX) {
        res = -1;
    }
    return res;
}
int32_t main() {
	ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    string in = "inputs/" + name;
    string out = "outputs/" + name;
    freopen(in.c_str(), "r", stdin);
    freopen(out.c_str(), "w", stdout);
    int T = read();
    for (int i = 1; i <= T ; i++) {
        cout << "Case #" << i << ": " << solve() << endl;
    }
    return 0;
}
```

## Road to Nutella

The knowledge needed for this problem is graph theory, 2-edge connected components, bridge trees, bipartite graph is 2 colorable or in other words contain no odd length cycle. 

The idea is to get all the non-bipartite 2-edge connected components, can call these blocks.  

The bridge tree is a conversion of graph to a tree where the edges are the bridge edges and the nodes are 2-edge connected components (blocks).  A 2-edge connected component could be bipartite or non-bipartite, this can indicate if it has odd length cycle within it or not.  

Convert to bridge tree by using Tarjan's bridge finding algorithm.

For each node pair ai, bi for the simple path connecting them within the bridge tree, find the path minimum along the simple path from ai to bi and their corresponding blocks in the bridge tree. The value you are minimizing is that for each block in the tree you need to calculate the shortest distance to a non bipartite block in the tree.  

Can calculate the shortest distance by using a multisource bfs from each non bipartite block in the tree and storing the shortest distance to each other block. 

Then you have these values so you want the path minimum query for each ai to bi block.  You can solve this efficiently in a tree with binary jumping/lifting, heavy light decomposition or link-cut tree

### Heavy light tree decomposition for path min queries

```cpp
vector<vector<pair<int,int>>> adj;
vector<vector<int>> bridge_adj;
vector<pair<int,int>> edges;
vector<bool> bridge_edge, bipartite_block;
vector<int> tin, low, colors, blocks, dist;
int n, m, timer, block_id;

const int inf = LLONG_MAX;

void dfs(int u, int p = -1) {
    colors[u] = 1;
    tin[u] = low[u] = timer++;
    for (auto &[v, i] : adj[u]) {
        if (v == p) continue;
        if (colors[v] != 0) { // back edge
            low[u] = min(low[u], tin[v]);
        } else {
            dfs(v, u);
            low[u] = min(low[u], low[v]);
            if (low[v] > tin[u]) bridge_edge[i] = true;
        }
    }
}

void block_dfs(int u, int p = -1) {
    colors[u] = 1;
    blocks[u] = block_id; // assigns block id to each node id
    for (auto &[v, i]: adj[u]) {
        if (v == p) continue;
        if (bridge_edge[i]) continue; // only dfs on block
        if (colors[v] != 0) continue;
        block_dfs(v, u);
    }
}

void bridge_tree() {
    timer = 0;
    colors.assign(n, 0);
    tin.assign(n, -1);
    low.assign(n, -1);
    bridge_edge.assign(m, false);
    dfs(0, -1); // Assume the graph is a one connected component
    block_id = 0;
    colors.assign(n, 0);
    blocks.assign(n, -1);
    for (int i = 0; i < n; i++) {
        if (colors[i] == 0) {
            block_dfs(i);
            block_id++;
        }
    }
    bridge_adj.assign(block_id, vector<int>{});
    // bridges will be edges in the bridge tree and blocks will be nodes
    for (int i = 0; i < m; i++) {
        if (bridge_edge[i]) {
            int u, v;
            tie(u, v) = edges[i];
            bridge_adj[blocks[u]].push_back(blocks[v]);
            bridge_adj[blocks[v]].push_back(blocks[u]);
        }
    }
}

bool bipartite_dfs(int source) {
    stack<int> stk;
    stk.push(source);
    colors[source] = 1;
    bool ans = true;
    while (!stk.empty()) {
        int u = stk.top();
        stk.pop();
        for (auto &[v, i] : adj[u]) {
            if (bridge_edge[i]) continue;
            if (colors[v] == 0) { // unvisited
                colors[v] = 3 - colors[u]; // needs to be different color of two possible values 1 and 2
                stk.push(v);
            } else if (colors[u] == colors[v]) {
                ans = false;
            }
        }
    }
    return ans;
}

// multisource bfs on bridge tree from every non-bipartite block to compute shortest distance from every bipartite block to non-bipartite block
void bfs() {
    queue<int> q;
    dist.assign(block_id, inf);
    for (int i = 0; i < block_id; i++) {
        if (!bipartite_block[i]) {
            q.push(i); 
            dist[i] = 0;
        }
    }
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (int v : bridge_adj[u]) {
            if (dist[u] + 1 < dist[v]) {
                dist[v] = dist[u] + 1;
                q.push(v);
            }
        }
    }
}

vector<int> parent, depth, head, sz, index_map, heavy;
int counter, neutral;

int func(int x, int y) {
    return min(x, y);
}

struct SegmentTree {
    int size;
    vector<int> nodes;

    void init(int num_nodes) {
        size = 1;
        while (size < num_nodes) size *= 2;
        nodes.assign(size * 2, neutral);
    }

    void ascend(int segment_idx) {
        while (segment_idx > 0) {
            int left_segment_idx = 2 * segment_idx, right_segment_idx = 2 * segment_idx + 1;
            nodes[segment_idx] = func(nodes[left_segment_idx], nodes[right_segment_idx]);
            segment_idx >>= 1;
        }
    }

    void update(int segment_idx, int val) {
        segment_idx += size;
        nodes[segment_idx] = val;
        segment_idx >>= 1;
        ascend(segment_idx);
    }

    int query(int left, int right) {
        left += size, right += size;
        int res = neutral;
        while (left <= right) {
            if (left & 1) {
                res = func(res, nodes[left]);
                left++;
            }
            if (~right & 1) {
                res = func(res, nodes[right]);
                right--;
            }
            left >>= 1, right >>= 1;
        }
        return res;
    }
};

SegmentTree seg;

int heavy_dfs(int u) {
    sz[u] = 1;
    int heavy_size = 0;
    for (int v : bridge_adj[u]) {
        if (v == parent[u]) continue;
        parent[v] = u;
        depth[v] = depth[u] + 1;
        int s = heavy_dfs(v);
        sz[u] += s;
        if (s > heavy_size) {
            heavy_size = s;
            heavy[u] = v;
        }
    }
    return sz[u];
}

void decompose(int u, int h) {
    index_map[u] = counter++;
    seg.update(index_map[u], dist[u]);
    head[u] = h;
    for (int v : bridge_adj[u]) {
        if (v == heavy[u]) {
            decompose(v, h);
        }
    }
    for (int v : bridge_adj[u]) {
        if (v == heavy[u] || v == parent[u]) continue;
        decompose(v, v);
    }
}

int query(int u, int v) {
    int res = neutral;
    while (true) {
        if (depth[u] > depth[v]) {
            swap(u, v);
        }
        int x = head[u];
        int y = head[v];
        if (x == y) {
            int left = index_map[u];
            int right = index_map[v];
            res = func(res, seg.query(left, right));
            break;
        } else if (depth[x] > depth[y]) {
            int left = index_map[x];
            int right = index_map[u];
            res = func(res, seg.query(left, right));
            u = parent[x];
        } else {
            int left = index_map[y];
            int right = index_map[v];
            res = func(res, seg.query(left, right));
            v = parent[y];
        }
    }
    return res;
}

int solve() {
    n = read(), m = read();
    adj.assign(n, vector<pair<int,int>>{});
    edges.resize(m);
    for (int i = 0; i < m; i++) {
        int u = read(), v = read();
        u--;
        v--;
        adj[u].push_back({v, i});
        adj[v].push_back({u, i});
        edges[i] = {u, v};
    }
    int res = 0;
    bridge_tree();
    // find all the bipartite blocks
    colors.assign(n, 0);
    bipartite_block.assign(block_id, false);
    for (int i = 0; i < n; i++) {
        if (colors[i] == 0) {
            if (bipartite_dfs(i)) {
                bipartite_block[blocks[i]] = true;
            }
        }
    }
    bfs();
    bool all_blocks_bipartite = all_of(bipartite_block.begin(), bipartite_block.end(), [&](bool x) {
        return x;
    });
    counter = 0;
    neutral = LLONG_MAX; // for min queries
    parent.assign(block_id, -1);
    depth.assign(block_id, 0);
    heavy.resize(block_id);
    head.assign(block_id, 0);
    sz.assign(block_id, 0);
    index_map.assign(block_id, 0);
    for (int i = 0; i < block_id; ++i) heavy[i] = i;
    seg.init(block_id);
    heavy_dfs(0);
    decompose(0, 0);
    int Q = read();
    while (Q--) {
        int u = read(), v = read();
        u--;
        v--;
        if (all_blocks_bipartite) {
            res--;
            continue;
        } 
        // path min query on the bridge tree
        res += query(blocks[u], blocks[v]);
    }
    return res;
}

int32_t main() {
	ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    string in = "inputs/" + name;
    string out = "outputs/" + name;
    freopen(in.c_str(), "r", stdin);
    freopen(out.c_str(), "w", stdout);
    int T = read();
    for (int i = 1; i <= T ; i++) {
        cout << "Case #" << i << ": " << solve() << endl;
    }
    return 0;
}
```

### Binary Jumping for path min queries with LCA

needs a sparse table and so on for also going up the tree along with the keeping the sparse table for ancestors, needs a sparse table for minimum value for some RMQ queries. 

```cpp
vector<vector<pair<int,int>>> adj;
vector<vector<int>> bridge_adj;
vector<pair<int,int>> edges;
vector<bool> bridge_edge, bipartite_block;
vector<int> tin, low, colors, blocks, dist;
int n, m, timer, block_id;

const int inf = INT_MAX;

void dfs(int u, int p = -1) {
    colors[u] = 1;
    tin[u] = low[u] = timer++;
    for (auto &[v, i] : adj[u]) {
        if (v == p) continue;
        if (colors[v] != 0) { // back edge
            low[u] = min(low[u], tin[v]);
        } else {
            dfs(v, u);
            low[u] = min(low[u], low[v]);
            if (low[v] > tin[u]) bridge_edge[i] = true;
        }
    }
}

void block_dfs(int u, int p = -1) {
    colors[u] = 1;
    blocks[u] = block_id; // assigns block id to each node id
    for (auto &[v, i]: adj[u]) {
        if (v == p) continue;
        if (bridge_edge[i]) continue; // only dfs on block
        if (colors[v] != 0) continue;
        block_dfs(v, u);
    }
}

void bridge_tree() {
    timer = 0;
    colors.assign(n, 0);
    tin.assign(n, -1);
    low.assign(n, -1);
    bridge_edge.assign(m, false);
    dfs(0, -1); // Assume the graph is a one connected component
    block_id = 0;
    colors.assign(n, 0);
    blocks.assign(n, -1);
    for (int i = 0; i < n; i++) {
        if (colors[i] == 0) {
            block_dfs(i);
            block_id++;
        }
    }
    bridge_adj.assign(block_id, vector<int>{});
    // bridges will be edges in the bridge tree and blocks will be nodes
    for (int i = 0; i < m; i++) {
        if (bridge_edge[i]) {
            int u, v;
            tie(u, v) = edges[i];
            bridge_adj[blocks[u]].push_back(blocks[v]);
            bridge_adj[blocks[v]].push_back(blocks[u]);
        }
    }
}

bool bipartite_dfs(int source) {
    stack<int> stk;
    stk.push(source);
    colors[source] = 1;
    bool ans = true;
    while (!stk.empty()) {
        int u = stk.top();
        stk.pop();
        for (auto &[v, i] : adj[u]) {
            if (bridge_edge[i]) continue;
            if (colors[v] == 0) { // unvisited
                colors[v] = 3 - colors[u]; // needs to be different color of two possible values 1 and 2
                stk.push(v);
            } else if (colors[u] == colors[v]) {
                ans = false;
            }
        }
    }
    return ans;
}

// multisource bfs on bridge tree from every non-bipartite block to compute shortest distance from every bipartite block to non-bipartite block
void bfs() {
    queue<int> q;
    dist.assign(block_id, inf);
    for (int i = 0; i < block_id; i++) {
        if (!bipartite_block[i]) {
            q.push(i); 
            dist[i] = 0;
        }
    }
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (int v : bridge_adj[u]) {
            if (dist[u] + 1 < dist[v]) {
                dist[v] = dist[u] + 1;
                q.push(v);
            }
        }
    }
}

// binary jumping algorithm

vector<int> depth, parent;
const int LOG = 20;
vector<vector<int>> ancestor, st;

// bfs from root of tree to calculate depth of nodes in the tree
void bfs(int root) {
    queue<int> q;
    depth.assign(block_id, inf);
    parent.assign(block_id, -1);
    q.push(root);
    depth[root] = 0;
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (int v : bridge_adj[u]) {
            if (depth[u] + 1 < depth[v]) {
                depth[v] = depth[u] + 1;
                parent[v] = u;
                q.push(v);
            }
        }
    }
}

// preprocess the ancestor and sparse table for minimum array
void preprocess() {
    ancestor.assign(LOG, vector<int>(block_id, -1));
    st.assign(LOG, vector<int>(block_id, inf));
    for (int i = 0; i < block_id; i++) {
        ancestor[0][i] = parent[i];
        st[0][i] = dist[i];
    }
    for (int i = 1; i < LOG; i++) {
        for (int j = 0; j < block_id; j++) {
            if (ancestor[i - 1][j] != -1) {
                ancestor[i][j] = ancestor[i - 1][ancestor[i - 1][j]];
                st[i][j] = min(st[i - 1][j], st[i - 1][ancestor[i - 1][j]]);
            }
        }
    }
}

// LCA queries to calculate the minimum node value in path from u to v
int lca(int u, int v) {
    if (depth[u] < depth[v]) swap(u, v);
    int ans = inf;
    int k = depth[u] - depth[v];
    if (k > 0) {
        for (int i = 0; i < LOG; i++) {
            if ((k >> i) & 1) {
                ans = min(ans, st[i][u]);
                u = ancestor[i][u];
            }
        }
    }
    if (u == v) {
        ans = min(ans, st[0][u]);
        return ans;
    }
    for (int i = LOG - 1; i >= 0; i--) {
        if (ancestor[i][u] != -1 && ancestor[i][u] != ancestor[i][v]) {
            ans = min(ans, st[i][u]);
            ans = min(ans, st[i][v]);
            u = ancestor[i][u]; v = ancestor[i][v];
        }
    }
    ans = min(ans, st[1][u]);
    ans = min(ans, st[1][v]);
    return ans;
}

int solve() {
    n = read(), m = read();
    adj.assign(n, vector<pair<int,int>>{});
    edges.resize(m);
    for (int i = 0; i < m; i++) {
        int u = read(), v = read();
        u--;
        v--;
        adj[u].push_back({v, i});
        adj[v].push_back({u, i});
        edges[i] = {u, v};
    }
    int res = 0;
    bridge_tree();
    // find all the bipartite blocks
    colors.assign(n, 0);
    bipartite_block.assign(block_id, false);
    for (int i = 0; i < n; i++) {
        if (colors[i] == 0) {
            if (bipartite_dfs(i)) {
                bipartite_block[blocks[i]] = true;
            }
        }
    }
    bfs();
    bool all_blocks_bipartite = all_of(bipartite_block.begin(), bipartite_block.end(), [&](bool x) {
        return x;
    });
    bfs(0);
    preprocess();
    int Q = read();
    while (Q--) {
        int u = read(), v = read();
        u--;
        v--;
        if (all_blocks_bipartite) {
            res--;
            continue;
        } 
        // path min query on the bridge tree
        res += lca(blocks[u], blocks[v]);
    }
    return res;
}

int32_t main() {
	ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    string in = "inputs/" + name;
    string out = "outputs/" + name;
    freopen(in.c_str(), "r", stdin);
    freopen(out.c_str(), "w", stdout);
    int T = read();
    for (int i = 1; i <= T ; i++) {
        cout << "Case #" << i << ": " << solve() << endl;
    }
    return 0;
}
```

# Round 1

## Problem B1: Sum 41 (Chapter 1)

### Solution 1: 

```cpp
const int N = 1e9;
vector<int> spf;
void sieve(int n) {
    for (int i = 2; i <= n; i++) {
        if (spf[i] != 1) continue;
        for (int j = i; j <= n; j += i) {
            spf[j] = i;
        }
    }
}

vector<int> factorize(int x) {
    vector<int> factors;
    while (x > 1) {
        factors.push_back(spf[x]);
        x /= spf[x];
    }
    return factors;
}

void solve() {
    int P = read();
    vector<int> factors = factorize(P);
    int fsum = 0;
    for (int f : factors) fsum += f;
    if (fsum > 41) {
        cout << -1;
        return;
    }
    while (fsum < 41) {
        factors.push_back(1);
        fsum++;
    }
    // then it will equal to 41 so print out all these factors
    cout << factors.size() << " ";
    for (int f : factors) cout << f << " ";
}

int32_t main() {
	ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    string in = "inputs/" + name;
    string out = "outputs/" + name;
    freopen(in.c_str(), "r", stdin);
    freopen(out.c_str(), "w", stdout);
    int T = read();
    spf.assign(N + 1, 1);
    sieve(N);
    for (int i = 1; i <= T ; i++) {
        cout << "Case #" << i << ": ";
        solve();
        cout << endl;
    }
    return 0;
}
```

## Problem B: Sum 41 (Chapter 2)

### Solution 1: 

```cpp

```

## Problem C: Back in Black (Chapter 1)

### Solution 1: 

```cpp

```

## Problem C: Back in Black (Chapter 1)

### Solution 1: 

```cpp

```

## Problem D: Today is Gonna be a Great Day

### Solution 1: 

```cpp

```

## Problem E: Bohemian Rap-sody

### Solution 1: 

```cpp

```

# Round 2

## Wiki Race

topics are strings

vertex disjoint path means a vertex will never belong to more than one path, it belongs to exactly one path.

if the root node has the candidate word, so if there are two leaves or more missing the candidate word then it doesn't match.

if the root node doesn't have the candidate word, then no leaves can miss the candidate. 

key insight is that if you compress the graph, if you iterate over each compressed graph L times for each leaf, there is L + L/2 + L/4 + ... + 1 = 2L total nodes in the compressed graph.  So you can run an algorithm that is O(L) time complexity for traversing from each leaf node to find the nearest node with the candidate word.  This means in the compression you need to merge all linear sections of nodes into the topmost node, and also need to track if you are at this node, where does it teleport to up in the tree.  This is the teleport array.  And now you can traverse from each leaf node up through the tree, where you visit each node exactly once cause it is vertex disjoint.

So that is why it would be L iterations for each candidate word, and since you can have at most sum(Wi) / L candidate words, you know the time complexity in total will be O(L).  

### Solution 1: 

```cpp
string base = "wiki_race";
// string name = base + "_sample_input.txt";
// string name = base + "_validation_input.txt";
string name = base + "_input.txt";

int counter;
int N;
vector<int> P, csz, teleport, leaves;
vector<bool> vis;
vector<vector<int>> adj;
unordered_map<string, int> freq;
vector<unordered_set<string>> words;

void dfs(int u, int p = -1) {
    for (int v : adj[u]) {
        if (v == p) continue;
        if (csz[u] == 1) {
            teleport[v] = teleport[u];
        }
        dfs(v, u);
        if (csz[u] == 1) {
            if (words[u].size() < words[v].size()) {
                swap(words[u], words[v]);
            }
            for (const string& s : words[v]) words[u].insert(s);
        }
    }
    if (csz[u] == 0) leaves.emplace_back(u);
}

bool dfs1(const string& word) {
    vector<int> nodes;
    bool works = true;
    for (int leaf : leaves) {
        bool found = false;
        int u = teleport[leaf];
        while (!found) {
            if (vis[u]) break;
            vis[u] = true;
            nodes.emplace_back(u);
            if (words[u].count(word)) {
                found = true;
                break;
            }
            if (u == 0) break;
            u = teleport[P[u]];
        }
        works &= found;
    }
    for (int x : nodes) {
        vis[x] = false;
    }
    return works;
}

void solve() {
    cin >> N;
    counter = 0;
    P.assign(N, 0);
    csz.assign(N, 0);
    freq.clear();
    words.assign(N, unordered_set<string>());   
    adj.assign(N, vector<int>());
    vis.assign(N, false);
    teleport.assign(N, 0);
    leaves.clear();
    iota(teleport.begin(), teleport.end(), 0);
    for (int i = 1; i < N; i++) {
        cin >> P[i];
        P[i]--;
        csz[P[i]]++;
        adj[P[i]].emplace_back(i);
        adj[i].emplace_back(P[i]);
    }
    for (int i = 0; i < N; i++) {
        int M;
        cin >> M;
        for (int j = 0; j < M; j++) {
            string s;
            cin >> s;
            freq[s]++;
            words[i].insert(s);
        }
    }
    // compression of tree
    dfs(0);
    int L = leaves.size();
    int ans = 0;
    for (const auto& [key, value] : freq) {
        if (value < L) continue;
        if (dfs1(key)) ans++;
    }
    cout << ans << endl;
}

signed main() {
	ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    string in = "inputs/" + name;
    string out = "outputs/" + name;
    freopen(in.c_str(), "r", stdin);
    freopen(out.c_str(), "w", stdout);
    int T;
    cin >> T;
    for (int i = 1; i <= T ; i++) {
        cout << "Case #" << i << ": ";
        solve();
        cout.flush();
    }
    return 0;
}
```

## Problem D: Tower Rush

### Solution 1: bezout's identity, combinatorics, inclusion exclusion, downward sieve

1. You need to figure out the assumption that it is interchangeable for if Alice or Bob has a particular block, it doesn't matter who has a block. 
1. It is easy to see that this is the case for solvability because you can apply bezout's identity cause you have a linear combination equation like x1h1+x2h2-y1h3-y2h4=D.
1. That actually is all you need to prove you don't care who picks the block
1. So you just need to find all k sized combinations of blocks that satisfy where the gcd of the blocks heights divides D. And count the number of those combinations.
1. And since if you choose different block on turn i, you need to multiple by k! at the end to account for different permutations of each combination.

suppose you are gcd is 6, then you will be divisible by 1,2,3 as well.  so the same sequence is counted in b[1], b[2], b[3], b[6].  So you need to use inclusion exclusion to get the exact count of sequences with gcd exactly g.

But you can also just use a downward sieve to calculate the exact count of sequences with gcd exactly g.  So for each g from maxH down to 1, you can subtract all multiples of g from the count of sequences with gcd g to get the exact count of sequences with gcd exactly g.

```cpp
const int MAXN = 1e6 + 5, MOD = 1e9 + 7;
int N, K, D;
int freq[MAXN];

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

void solve() {
    cin >> N >> K >> D;
    memset(freq, 0, sizeof(freq));
    for (int i = 0; i < N; ++i) {
        int x;
        cin >> x;
        freq[x]++;
    }
    vector<int64> dp(MAXN, 0);
    for (int x = 1; x < MAXN; ++x) {
        int cnt = 0;
        for (int y = x; y < MAXN; y += x) {
            cnt += freq[y];
        }
        if (cnt >= K) {
            dp[x] = fact[cnt] * inv_fact[cnt - K] % MOD;
        }
    }
    for (int x = MAXN - 1; x >= 1; --x) {
        for (int y = 2 * x; y < MAXN; y += x) {
            dp[x] -= dp[y];
            if (dp[x] < 0) dp[x] += MOD;
        }
    }
    int64 ans = 0;
    for (int x = 1; x <= D; ++x) {
        if (D % x == 0) {
            ans = (ans + dp[x]) % MOD;
        }
    }
    cout << ans << endl;
}
```

# Round 3

## Problem A: Spooky Splits

### Solution 1: recursion, dfs and backtracking, memoization, stars and bars combinatorics, multisets

1. Memoize the states, suppose the current group loads are [1, 1, 1, 2, 0, 3]. The three groups that currently have load 1 are interchangeable.
1. Fun way to formulate to understand the time complexity is to understand how many possible multisets there are.  You can use the stars and bars method to calculate this.  So you know you can allow any value in the range [0, N/K], so call these your stars so you have N/K + 1 - 1 bars, and then you have K stars and in a sense you say how many ways can I place the stars which is the combinatorics $\binom{K + (N/K + 1) - 1}{K}$.  This is fun way to define it.  Now you can run a program to calculate worst case time complexity.  So also to think more about this where you place the stars indicate the value of that element in the multiset so i if you have *||**||**, that means you have a multiset like this {0, 2, 2, 4, 4}.  In this case my alphabet is just the integers [0, N/K] also, these are the type of items.  But yeah the K stars are indistinguishable
1. The largest if you do a quick loop is 185,000 possible states.

```cpp
int N, M;
vector<int> A, cnts;
set<multiset<int>> memo;
multiset<int> key;

struct UnionFind {
    vector<int> parents, size;
    UnionFind(int n) {
        parents.resize(n);
        iota(parents.begin(),parents.end(),0);
        size.assign(n,1);
    }

    int find(int i) {
        if (i==parents[i]) {
            return i;
        }
        return parents[i]=find(parents[i]);
    }

    void unite(int i, int j) {
        i = find(i), j = find(j);
        if (i!=j) {
            if (size[j]>size[i]) {
                swap(i,j);
            }
            size[i]+=size[j];
            parents[j]=i;
        }
    }

    bool same(int i, int j) {
        return find(i) == find(j);
    }
    
    vector<int> groups() {
        int n = parents.size();
        unordered_map<int, vector<int>> group_map;
        for (int i = 0; i < n; ++i) {
            group_map[find(i)].emplace_back(i);
        }
        vector<int> ans;
        for (auto& [_, group] : group_map) {
            ans.emplace_back(group.size());
        }
        return ans;
    }
};

bool dfs(int idx) {
    int K = cnts.size();
    if (idx == A.size()) return true;
    if (memo.find(key) != memo.end()) return false;
    memo.insert(key);
    for (int i = 0; i < K; ++i) {
        if (cnts[i] + A[idx] > N / K) continue;
        key.erase(key.find(cnts[i]));
        cnts[i] += A[idx];
        key.insert(cnts[i]);
        if (dfs(idx + 1)) return true;
        key.erase(key.find(cnts[i]));
        cnts[i] -= A[idx];
        key.insert(cnts[i]);
    }
    return false;
}

void solve() {
    cin >> N >> M;
    UnionFind dsu(N);
    for (int i = 0; i < M; ++i) {
        int u, v;
        cin >> u >> v;
        u--, v--;
        dsu.unite(u, v);
    }
    A = dsu.groups();
    sort(A.rbegin(), A.rend());
    for (int k = 1; k <= N; ++k) {
        if (N % k != 0) continue;
        cnts.assign(k, 0);
        memo.clear();
        key = multiset<int>(cnts.begin(), cnts.end());
        if (dfs(0)) {
            cout << k << " ";
        }
    }
    cout << endl;
}
```

## Problem B: Hash Slinger

### Solution 1: dp, dijkstra

1. Using this earliest index is the key to solving this problem. 
1. Straightforward to derive an O(N^2M) time complexity dp solution.
1. To get the O(N^2 + M^2) time complexity, you need to use the earliest index in which you can get a segment that has a sum equal to some value after an index first[i][d].
1. It is not the normal way you'd see dijkstra implemented, but it is because it is vising each node what has the smallest index at which it can be achieved, so it is visiting in order and is assigning a priority to some value.

```cpp
int N, M;
vector<int> A;
vector<vector<int>> first;

void solve() {
    cin >> N >> M;
    A.assign(N, 0);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    first.assign(N + 1, vector<int>(M + 1, N + 1));
    for (int i = N - 1; i >= 0; --i) {
        for (int j = 0; j <= M; j++) {
            first[i][j] = first[i + 1][j];
        }
        for (int j = i, val = 0; j < N; ++j) {
            val = (val + A[j]) % M;
            first[i][val] = min(first[i][val], j);
        }
    }
    int bitWidth = 32 - __builtin_clz(M);
    int sz = 1 << bitWidth;
    vector<int> dist(sz, N + 1);
    dist[0] = 0;
    vector<bool> vis(sz, false);
    for (int i = 0; i < sz; ++i) {
        int u = -1, minIndex = N + 1;
        for (int j = 0; j < sz; ++j) {
            if (vis[j]) continue;
            if (dist[j] < minIndex) {
                minIndex = dist[j];
                u = j;
            }
        }
        if (u == -1) break;
        vis[u] = true;
        for (int j = 0; j <= M; ++j) {
            int idx = first[minIndex][j];
            int v = u ^ j;
            if (idx < dist[v]) {
                dist[v] = idx + 1;
            }
        }
    }
    int ans = accumulate(dist.begin(), dist.end(), 0, [](const int total, const int x) {
        return total + (x != N + 1 ? 1 : 0);
    });
    cout << ans << endl;
}
```

## Problem C: Krab-otage

### Solution 1: 

1. Find the best path for Mr. Krabs, for certain Plankton will act to block out part of that path. 
1. The intersection of the paths is all in one column or one row.
1. 

Need more work on this one, this solution probably works, taken from a submission. 

```cpp
    int n, m;
    cin >> n >> m;

    const int64 INF = (int64)4e18;
    const int64 NEG = -INF;

    // 1-index the grid, pad by 1 on each side to avoid bounds checks
    vector<vector<int64>> A(n + 2, vector<int64>(m + 2, 0));

    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= m; ++j)
            cin >> A[i][j];

    // d1: best from (1,1) to (i,j) going right or down
    vector<vector<int64>> d1(n + 2, vector<int64>(m + 2, 0));
    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= m; ++j)
            d1[i][j] = max(d1[i][j - 1], d1[i - 1][j]) + A[i][j];

    // d2: best from (i,j) to (n,m) going right or down
    vector<vector<int64>> d2(n + 2, vector<int64>(m + 2, 0));
    for (int i = n; i >= 1; --i)
        for (int j = m; j >= 1; --j)
            d2[i][j] = max(d2[i][j + 1], d2[i + 1][j]) + A[i][j];

    // dp states
    // up[i][j] = dp if Plankton is about to go VERTICALLY down starting somewhere above (i,j) and
    //            the next segment enters column j and ends at row i
    // le[i][j] = dp if Plankton is about to go HORIZONTALLY left starting somewhere right of (i,j) and
    //            the next segment enters row i and ends at column j
    vector<vector<int64>> up(n + 2, vector<int64>(m + 2, INF));
    vector<vector<int64>> le(n + 2, vector<int64>(m + 2, INF));

    // starting corner cannot already include the intersection
    up[1][m] = le[1][m] = NEG;

    // sweep i increasing and j decreasing to break the mutual dependency
    for (int i = 1; i <= n; ++i) {
        for (int j = m; j >= 1; --j) {

            // extend a VERTICAL segment in column j from its top at row i to every bottom k > i
            {
                int64 bestArrive = d1[i - 1][j];   // best arrival before entering the column j segment
                int64 sumMax = NEG;                // best if Krabs exits inside the vertical segment

                for (int k = i + 1; k <= n; ++k) {
                    // if Plankton extends to row k, Krabs can enter at any row t in [i..k]
                    // bestArrive tracks max over t of d1[t-1][j] or d1[t][j-1]
                    bestArrive = max(bestArrive, d1[k - 1][j - 1]);

                    if (k > i + 1) {
                        // Krabs enters above k-1 inside the segment, then exits to the right at (k-1, j+1)
                        sumMax = max(sumMax, bestArrive + d2[k - 1][j + 1]);
                    }

                    // If Plankton ends the vertical segment at row k,
                    // Krabs may:
                    //  - enter somewhere above k and then exit just below at (k+1, j)
                    //  - enter somewhere above k and then exit to the right at (k, j+1)
                    //  - or he may have intersected on the previous HORIZONTAL segment, which is le[i][j]
                    int64 now = max({
                        bestArrive + max(d2[k + 1][j], d2[k][j + 1]),
                        sumMax,
                        le[i][j]
                    });

                    up[k][j] = min(up[k][j], now);
                }
            }

            // extend a HORIZONTAL segment in row i from its right end at column j to every left end l < j
            {
                int64 bestFinish = d2[i][j + 1];   // best finish after leaving row i segment to the right
                int64 sumMax = NEG;               // best if Krabs exits inside the horizontal segment

                for (int l = j - 1; l >= 1; --l) {
                    // as we extend left, Krabs may finish later by going down at (i+1, l+1)
                    bestFinish = max(bestFinish, d2[i + 1][l + 1]);

                    if (l < j - 1) {
                        // Krabs first goes down at some column in (l+1..j-1), after arriving there
                        sumMax = max(sumMax, bestFinish + d1[i - 1][l + 1]);
                    }

                    // If Plankton ends the horizontal segment at column l,
                    // Krabs may:
                    //  - finish at the left neighbor (i, l-1) or above (i-1, l)
                    //  - or he may have intersected on the previous VERTICAL segment, which is up[i][j]
                    int64 now = max({
                        bestFinish + max(d1[i][l - 1], d1[i - 1][l]),
                        sumMax,
                        up[i][j]
                    });

                    le[i][l] = min(le[i][l], now);
                }
            }
        }
    }

    int64 ans = min(up[n][1], le[n][1]);
    cout << ans << endl;
```

## Problem D: Double Stars

### Solution 1: reroot tree dp, frequency maps

1. This may look it could risk O(N^2) time complexity but it actually doesn't because each node can only have about sqrt(N) distinct arm lengths.
1. A leaf node cannot be one of the centers for a double star. (leaf node means the degree is 1)
1. Any two nodes that are adjacent and neither are leaf nodes can form at least one double star.  (at minimum you should be able to take x = 1, y = 1) (that means 1 chain, length 1 from each center)
1. degree - 1 that is number of chains you can form from double stars centers, take min between the two centers.
1. tree reroot dp, calculate the longest arm length from each node. 
1. For each node u, have the frequency map of the frequency of each arm length. 

```cpp
int64 ans;
int N;
vector<vector<int>> adj;
vector<int> anc, desc;
vector<map<int, int>> chains;

void dfs1(int u, int p = -1) {
    for (int v : adj[u]) {
        if (v == p) continue;
        dfs1(v, u);
        desc[u] = max(desc[u], desc[v] + 1);
    }
}

void dfs2(int u, int p = -1) {
    int val1 = -1, val2 = -1, u1 = -1, u2 = -1;
    for (int v : adj[u]) {
        if (v == p) continue;
        if (desc[v] > val1) {
            val2 = val1;
            u2 = u1;
            val1 = desc[v];
            u1 = v;
        } else if (desc[v] > val2) {
            val2 = desc[v];
            u2 = v;
        }
    }
    for (int v : adj[u]) {
        if (v == p) continue;
        if (v == u1) {
            anc[v] = max(anc[u] + 1, val2 + 2);
        } else {
            anc[v] = max(anc[u] + 1, val1 + 2);
        }
        dfs2(v, u);
    }
}

void dfs3(int u, int p = -1) {
    chains[u][anc[u]]++;
    for (int v : adj[u]) {
        if (v == p) continue;
        chains[u][desc[v] + 1]++;
        dfs3(v, u);
    }
}

void dfs4(int u, int p = -1) {
    for (int v : adj[u]) {
        if (v == p) continue;
        // double star u -> v
        chains[u][desc[v] + 1]--;
        chains[v][anc[v]]--;
        auto itu = chains[u].rbegin();
        auto itv = chains[v].rbegin();
        int cntu = itu != chains[u].rend() ? itu->second : 0;
        int cntv = itv != chains[v].rend() ? itv->second : 0;
        while (itu != chains[u].rend() && itv != chains[v].rend()) {
            int armu = itu->first;
            int armv = itv->first;
            int arm = min(armu, armv);
            int cnt = min(cntu, cntv);
            ans += 1LL * arm * cnt;
            cntu -= cnt;
            cntv -= cnt;
            if (cntu == 0) {
                ++itu;
                cntu = itu != chains[u].rend() ? itu->second : 0;
            }
            if (cntv == 0) {
                ++itv;
                cntv = itv != chains[v].rend() ? itv->second : 0;
            }
        }
        chains[u][desc[v] + 1]++;
        chains[v][anc[v]]++;
        dfs4(v, u);
    }
}

void solve() {
    cin >> N;
    adj.assign(N, vector<int>());
    for (int i = 1; i < N; ++i) {
        int p;
        cin >> p;
        p--;
        adj[p].emplace_back(i);
        adj[i].emplace_back(p);
    }
    desc.assign(N, 0);
    dfs1(0);
    anc.assign(N, 0);
    dfs2(0);
    chains.assign(N, map<int, int>());
    dfs3(0);
    ans = 0;
    dfs4(0);
    cout << ans << endl;
}
```

## Problem E: Similar Ships

### Solution 1: tree diameter, bfs, combinatorics, path graph

1. They use the word subtree to refer to any induced connected subgraph.
1. Pick t2 that creates as few short paths as possible.
1. This happens to be picking a linear chain for t2. 
1. Branching puts many vertices close together. A path spreads vertices apart.
1. Among N-vertex trees, the star minimizes average distance and diameter. 
1. Among N-vertex trees, the path maximizes average distance and diameter.
1. If you want to minimize the number of close pairs make the tree look like a path. 

```cpp
const int MOD = 1e9 + 7;
int N;
vector<vector<int>> adj;

vector<int> bfs(int src) {
    vector<int> dist(N, -1);
    dist[src] = 0;
    queue<int> q;
    q.emplace(src);
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (int v : adj[u]) {
            if (dist[v] == -1) {
                dist[v] = dist[u] + 1;
                q.emplace(v);
            }
        }
    }
    return dist;
}

void solve() {
    cin >> N;
    adj.assign(N, vector<int>());
    for (int i = 1; i < N; ++i) {
        int p;
        cin >> p;
        --p;
        adj[p].emplace_back(i);
        adj[i].emplace_back(p);
    }
    vector<int> d0 = bfs(0);
    int best = -1, u = -1, v = -1;
    for (int i = 0; i < N; ++i) {
        if (d0[i] > best) {
            best = d0[i];
            u = i;
        }
    }
    vector<int> du = bfs(u);
    best = -1;
    for (int i = 0; i < N; ++i) {
        if (du[i] > best) {
            best = du[i];
            v = i;
        }
    }
    int64 ans = 0;
    for (int i = 0; i <= best; ++i) {
        ans += N - i;
        ans %= MOD;
    }
    cout << ans << endl;
}
```