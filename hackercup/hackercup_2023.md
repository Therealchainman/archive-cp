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