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

const int MAXN = 200'005;

int n, ans[MAXN], freq[MAXN], max_dist_subtree1[MAXN], max_dist_subtree2[MAXN], child1[MAXN], child2[MAXN], parent_max_dist[MAXN];
vector<int> adj_list[MAXN];

int dfs1(int node, int parent) {
    for (int child : adj_list[node]) {
        if (child == parent) continue;
        int max_dist_subtree = dfs1(child, node);
        if (max_dist_subtree > max_dist_subtree1[node]) {
            max_dist_subtree2[node] = max_dist_subtree1[node];
            child2[node] = child1[node];
            max_dist_subtree1[node] = max_dist_subtree;
            child1[node] = child;
        } else if (max_dist_subtree > max_dist_subtree2[node]) {
            max_dist_subtree2[node] = max_dist_subtree;
            child2[node] = child;
        }
    }
    return max_dist_subtree1[node] + 1;
}

void dfs2(int node, int parent) {
    parent_max_dist[node] = parent_max_dist[parent] + 1;
    if (parent != 0) {
        if (node != child1[parent]) {
            parent_max_dist[node] = max(parent_max_dist[node], max_dist_subtree1[parent] + 1);
        } else {
            parent_max_dist[node] = max(parent_max_dist[node], max_dist_subtree2[parent] + 1);
        }
    }
    for (int child : adj_list[node]) {
        if (child == parent) continue;
        dfs2(child, node);
    }
}

int main() {
	n = read();
    for (int i = 1; i < n; i++) {
        int u = read(), v = read();
        adj_list[u].push_back(v);
        adj_list[v].push_back(u);
    }
    dfs1(1, 0);
    memset(parent_max_dist, -1, sizeof(parent_max_dist));
    dfs2(1, 0);
    for (int i = 1; i <= n; i++) {
        freq[max(max_dist_subtree1[i], parent_max_dist[i])] += 1;
    }
	fill(ans, ans + n + 1, n);
    int suffix_freq = 0;
    for (int i = n; i >= 1; i--) {
        suffix_freq += freq[i];
        if (suffix_freq > 0) {
            ans[i] = n - suffix_freq + 1;
        }
    }
    for (int i = 1; i <= n; i++) {
        cout << ans[i] << " ";
    }
    cout << endl;
    return 0;
}