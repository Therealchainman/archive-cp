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

const int threshold = 300;
const int inf = 1e16;

struct dp_state
{
    vector<int> zero_trees, one_trees;
    void init()
    {
        zero_trees.resize(threshold + 1, inf);
        one_trees.resize(threshold + 1, inf);
        zero_trees[1] = 1;
        one_trees[1] = 2;
    }
};

int32_t  main(){
	int T = read();
    while (T--) {
        int n = read();
        vector<vector<int>> adj_list(n + 1);
        vector<int> sz(n + 1);
        for (int i = 0; i < n - 1; i++) {
            int u = read(), v = read();
            adj_list[u].push_back(v);
            adj_list[v].push_back(u);
        }
        function<int(int, int)> heavy_edge_first = [&](int node, int parent) {
            int sz = 1;
            pair<int, int> res;
            for (int i = 0; i < adj_list[node].size(); i++) {
                if (adj_list[node][i] == parent) continue;
                int child_sz = heavy_edge_first(adj_list[node][i], node);
                res = max(res, {child_sz, i});
                sz += child_sz;
            }
            if (!adj_list[node].empty()) {
                swap(adj_list[node][0], adj_list[node][res.second]);
            }
            return sz;
        };
        heavy_edge_first(1, 0);
        vector<vector<int>> merged(2 * threshold + 1, vector<int>(2, inf));
        function<dp_state(int, int)> dfs = [&](int node, int parent) {
            dp_state dp;
            sz[node] = 1;
            bool hasinit = false;
            for (auto nei : adj_list[node]) {
                if (nei == parent) continue;
                dp_state nei_dp = dfs(nei, node);
                if (!hasinit) {
                    dp.init();
                    hasinit = true;
                }
                for (int i = 0; i <= min(sz[node], threshold) + min(sz[nei], threshold); i++ ) {
                    merged[i][0] = merged[i][1] = inf;
                }
                for (int j = 1; j <= min(sz[node], threshold); j++) {
                    for (int k = 1; k <= min(sz[nei], threshold); k++) {
                        merged[j][0] = min(merged[j][0], dp.zero_trees[j] + nei_dp.one_trees[k]);
                        merged[j][1] = min(merged[j][1], dp.one_trees[j] + nei_dp.zero_trees[k]);
                        merged[j + k][0] = min(merged[j + k][0], dp.zero_trees[j] + nei_dp.zero_trees[k] + j * k);
                        merged[j + k][1] = min(merged[j + k][1], dp.one_trees[j] + nei_dp.one_trees[k] + j * k * 2);
                    }
                }
                sz[node] += sz[nei];
                for (int i = 1; i <= min(sz[node], threshold); i++) {
                    dp.zero_trees[i] = merged[i][0];
                    dp.one_trees[i] = merged[i][1];
                }
            }
            if (!hasinit) {
                dp.init();
                hasinit = true;
            }
            return dp;
        };
        dp_state dp = dfs(1, 0);
        int res = inf;
        for (int i = 1; i <= threshold; i++) {
            res = min(res, dp.zero_trees[i]);
            res = min(res, dp.one_trees[i]);
        }
        cout << n * (n + 1) - res << endl;
    }
}