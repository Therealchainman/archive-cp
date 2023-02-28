#include <bits/stdc++.h>
using namespace std;

inline int read()
{
	int x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}

inline long long readll() {
	long long x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}

long long neutral = 0;
struct FenwickTree {
    vector<long long> nodes;
    
    void init(int n) {
        nodes.assign(n + 1, neutral);
    }

    void update(int idx, long long val) {
        while (idx < (int)nodes.size()) {
            nodes[idx] += val;
            idx += (idx & -idx);
        }
    }

    int query(int left, int right) {
        return query(right) - query(left);
    }

    long long query(int idx) {
        long long result = neutral;
        while (idx > 0) {
            result += nodes[idx];
            idx -= (idx & -idx);
        }
        return result;
    }
};

class EulerTour {
public:
    int num_nodes;
    vector<vector<int>> edges;
    vector<vector<int>> adj_list;
    int root_node;
    vector<int> enter_counter, exit_counter;
    int counter;

    EulerTour(int n, vector<vector<int>>& e) {
        num_nodes = n;
        edges = e;
        adj_list.resize(num_nodes + 1);
        root_node = 1;
        enter_counter.resize(num_nodes + 1);
        exit_counter.resize(num_nodes + 1);
        counter = 1;
        build_adj_list();
        euler_tour(root_node, -1);
    }

    void build_adj_list() {
        for (auto edge : edges) {
            int u = edge[0], v = edge[1];
            adj_list[u].push_back(v);
            adj_list[v].push_back(u);
        }
    }

    void euler_tour(int node, int parent_node) {
        enter_counter[node] = counter;
        counter++;
        for (auto child_node : adj_list[node]) {
            if (child_node != parent_node) {
                euler_tour(child_node, node);
            }
        }
        exit_counter[node] = counter - 1;
    }
};

int main() {
    // freopen("input.txt", "r", stdin);
    // freopen("output.txt", "w", stdout);
    int n = read(), q = read();
    vector<int> arr(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        arr[i] = readll();
    }
    vector<vector<int>> edges;
    for (int i = 0; i < n - 1; i++) {
        int u = read(), v = read();
        edges.push_back({u, v});
    }
    EulerTour euler_tour(n, edges);
    FenwickTree fenwick_tree;
    fenwick_tree.init(n + 1);
    for (int node = 1; node <= n; node++) {
        int enter_counter = euler_tour.enter_counter[node];
        fenwick_tree.update(enter_counter, arr[node]);
    }
    for (int i = 0; i < q; i++) {
        int t = read();
        if (t == 1) {
            int u = read(); long long x = readll();
            int node_index_in_flatten_tree = euler_tour.enter_counter[u];
            int delta = x - arr[u];
            arr[u] = x;
            fenwick_tree.update(node_index_in_flatten_tree, delta);
        }
        else {
            int s = read();
            long long subtree_sum = fenwick_tree.query(euler_tour.exit_counter[s]) - fenwick_tree.query(euler_tour.enter_counter[s] - 1);
            cout << subtree_sum << endl;
        }
    }
}