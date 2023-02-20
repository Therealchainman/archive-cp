#include <bits/stdc++.h>
using namespace std;

int dfs1(int node, int parent, vector<vector<int>>& graph, vector<int>& leaf_lens1, vector<int>& path_node1, vector<int>& leaf_lens2, vector<int>& path_node2) {
	for (int& nei : graph[node]) {
		if (nei == parent) continue;
		int leaf_len = dfs1(nei, node, graph, leaf_lens1, path_node1, leaf_lens2, path_node2);
		if (leaf_len > leaf_lens1[node]) {
			leaf_lens2[node] = leaf_lens1[node];
			path_node2[node] = path_node1[node];
			leaf_lens1[node] = leaf_len;
			path_node1[node] = nei;
		} else if (leaf_len > leaf_lens2[node]) {
			leaf_lens2[node] = leaf_len;
			path_node2[node] = nei;
		}
    }
	return leaf_lens1[node] + 1;
}
void dfs2(int node, int parent, vector<vector<int>>& graph, vector<int>& leaf_lens1, vector<int>& path_node1, vector<int>& leaf_lens2, vector<int>& path_node2, vector<int>& parent_lens) {
	parent_lens[node] = parent > 0 ? parent_lens[parent] + 1 : 0;
	if (parent > 0 && node != path_node1[parent]) {
		parent_lens[node] = max(parent_lens[node], leaf_lens1[parent] + 1);
	}
	if (parent > 0 && node != path_node2[parent]) {
		parent_lens[node] = max(parent_lens[node], leaf_lens2[parent] + 1);
	}
	for (int& nei : graph[node]) {
		if (nei == parent) continue;
		dfs2(nei, node, graph, leaf_lens1, path_node1, leaf_lens2, path_node2, parent_lens);
    }
}
int main() {
    int n, a,b;
    // freopen("input.txt", "r", stdin);
    // freopen("output.txt", "w", stdout);
    cin>>n;
    vector<vector<int>> graph(n+1);
    for(int i=0;i<n-1;i++){
        cin>>a>>b;
        graph[a].push_back(b);
        graph[b].push_back(a);
    }
	vector<int> leaf_lens1(n+1, 0), leaf_lens2(n + 1, 0), parent_lens(n + 1, 0);
	vector<int> path_node1(n + 1, 0), path_node2(n + 1, 0);
	dfs1(1, 0, graph, leaf_lens1, path_node1, leaf_lens2, path_node2);
	dfs2(1, 0, graph, leaf_lens1, path_node1, leaf_lens2, path_node2, parent_lens);
	for (int i = 1; i <= n; i++) {
		cout << max(leaf_lens1[i], parent_lens[i]) << " ";
	}
	cout << endl;
}