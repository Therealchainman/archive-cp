# Offline Incremental Strongly Connected Components Algorithm

requires you have all the edges in advance
It uses a persistent disjoint union set data structure and divide and conquer algorithm to compute the size of all the strongly connected components at each point in time.

```cpp
struct Edge {
	int u, v;
};

int R, C, Q, sum;
vector<Edge> edges;
vector<int> ans, comp, order, values;
vector<bool> vis;
vector<int*> pointers;
vector<vector<int>> adj, adj_t;

int cost(int x) {
	if (x == 1) return 0;
	return x * x;
}

struct UnionFind {
    vector<int> parents, size;
    void init(int n) {
        parents.resize(n);
        iota(parents.begin(),parents.end(),0);
        size.assign(n,1);
    }

    int find(int i) {
		if (i == parents[i]) return i;
		return find(parents[i]);
    }

    void unite(int i, int j) {
        i = find(i), j = find(j);
		if (i == j) return;
		if (comp[i] != comp[j]) return;
		if (size[j] > size[i]) swap(i, j);
		pointers.push_back(&sum);
		values.push_back(sum);
		sum = sum - cost(size[i]) - cost(size[j]) + cost(size[i] + size[j]);
		pointers.push_back(&parents[j]);
		values.push_back(parents[j]);
		parents[j] = i;
		pointers.push_back(&size[i]);
		values.push_back(size[i]);
		size[i] += size[j];
    }
};

UnionFind dsu;

// similar to topological ordering of nodes for graph
void dfs1(int u) {
	if (vis[u]) return;
	vis[u] = true;
	for (int v : adj[u]) {
		dfs1(v);
	}
	order.push_back(u);
}

// assigning components to nodes with transpose graph
void dfs2(int u, int c) {
	if (comp[u] != -1) return;
	comp[u] = c;
	for (int v : adj_t[u]) {
		dfs2(v, c);
	}
}

void calc(int left, int right, vector<int>& events) {
	if (right - left == 1) {
		ans.push_back(sum);
		return;
	}
	int mid = (left + right) / 2;
	// construct the graph for this interval
	for (int i : events) {
		if (i < mid) {
			int u = dsu.find(edges[i].u), v = dsu.find(edges[i].v);
			for (int w : {u, v}) {
				if (vis[w]) {
					adj[w].clear();
					adj_t[w].clear();
					vis[w] = false;
					comp[w] = -1;
				}
			}
			adj[u].push_back(v);
			adj_t[v].push_back(u); // transpose
		}
	}
	// Kosaraju's algorithm for finding SCC
	order.clear();
	for (int i : events) {
		if (i < mid) {
			int u = dsu.find(edges[i].u), v = dsu.find(edges[i].v);
			for (int w: {u, v}) {
				if (vis[w]) continue;
				dfs1(w);
			}
		}
	}
	reverse(order.begin(), order.end());
	int cur_comp = 0;
	for (int i : order) {
		if (comp[i] == -1) {
			dfs2(i, cur_comp++);
		}
	}
	int snap_time = values.size();
	// unite
	for (int i : events) {
		if (i < mid) dsu.unite(edges[i].u, edges[i].v);
	}
	vector<int> tol, tor;
	for (int i : events) {
		if (i >= mid) tor.push_back(i);
		else if (dsu.find(edges[i].u) == dsu.find(edges[i].v)) tol.push_back(i);
		else tor.push_back(i);
	}
	calc(mid, right, tor);
	// rollback to prior version for the disjoint set data structure
	while (values.size() > snap_time) {
		*pointers.end()[-1] = values.end()[-1];
		values.pop_back();
		pointers.pop_back();
	}
	calc(left, mid, tol);
}

void solve() {
	cin >> R >> C >> Q;
	edges.resize(Q);
	for (int i = 0; i < Q; i++) {
		int r, c;
		char col;
		cin >> r >> c >> col;
		r--; c--;
		if (col == 'R') { // col -> row
			edges[i] = {c, r + C};
		} else { // row -> col
			edges[i] = {r + C, c};
		}
	}
	vector<int> events(Q);
	iota(events.begin(), events.end(), 0);
	adj.assign(R + C, vector<int>());
	adj_t.assign(R + C, vector<int>());
	vis.assign(R + C, false);
	comp.assign(R + C, -1);
	dsu.init(R + C);
	pointers.resize(R + C);
	values.resize(R + C);
	calc(0, Q + 1, events);
	ans.pop_back();
	reverse(ans.begin(), ans.end());
	for (int x : ans) {
		cout << x << endl;
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