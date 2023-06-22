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

vector<vector<int>> adj_list;
vector<int> num_leaves;

int dfs(int node, int parent) {
    bool is_leaf = true;
    int cnt = 0;
    for (int child : adj_list[node]) {
        if (child == parent)
            continue;
        is_leaf = false;
        cnt += dfs(child, node);
    }
    if (is_leaf)
        cnt = 1;
    num_leaves[node] = cnt;
    return cnt;
}

int32_t main() {
    int T = read();
    
    while (T--) {
        int n = read();
        
        adj_list.clear();
        adj_list.resize(n + 1);
        
        for (int i = 0; i < n - 1; i++) {
            int u = read(), v = read();
            adj_list[u].push_back(v);
            adj_list[v].push_back(u);
        }
        
        int q = read();
        
        vector<pair<int, int>> queries(q);
        for (int i = 0; i < q; i++) {
            int x = read(), y = read();    
            queries[i] = make_pair(x, y);
        }
        
        num_leaves.clear();
        num_leaves.resize(n + 1);
        
        dfs(1, 0);
        
        for (const auto& query : queries) {
            int x = query.first;
            int y = query.second;
            
            int res = num_leaves[x] * num_leaves[y];
            cout << res << endl;
        }
    }
    
    return 0;
}
