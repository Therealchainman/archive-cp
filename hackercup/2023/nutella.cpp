#include <bits/stdc++.h>
using namespace std;
#define int long long

inline int read() {
	int x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}

// string name = "road_to_nutella_sample_input.txt";
// string name = "cheeseburger_corollary_2_validation_input.txt";
string name = "road_to_nutella_input.txt";

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

// binary lifting algorithm

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

/*
problem solve

g++ "-Wl,--stack,1078749825" nutella.cpp -o main
*/
