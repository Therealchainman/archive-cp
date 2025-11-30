# tree dp 

## optimize 0/1 knapsack tree dp 

threshold may be relative to problem, it's the largest you need to make the dp state arrays, which you may need a just two possibly

```cpp
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
```

With this you can use a trick that will memory optimize the dp tree by putting the heavy edges first in the tree, which are the largest subtrees will be seen first in the adjacency list.

```cpp
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
```

This is example of the dfs part using these optimizations.  You have to initialize the heavy edge in the recursion and this optimizes the memory. 

```cpp
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
```

## Knapsack over children by maintaining a prefix dp

This is a common technique that is necessary for some tree dp problems. 

loop over the paths used so far in prefix
and then loop over paths us in child v.