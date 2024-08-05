
# Leetcode Biweekly Contest 136

## Minimum Number of Flips to Make Binary Grid Palindromic II

### Solution 1:  connected components, counting, greedy

```cpp
class Solution {
public:
    int minFlips(vector<vector<int>>& grid) {
        int R = grid.size(), C = grid[0].size();
        int ans = 0;
        for (int r = 0; r < R / 2; r++) {
            for (int c = 0; c < C / 2; c++) {
                int cnt1 = grid[r][c] + grid[r][C - c - 1] + grid[R - r - 1][c] + grid[R - r - 1][C - c - 1];
                ans += min(cnt1, 4 - cnt1);
            }
        }
        int freq[3];
        memset(freq, 0, sizeof(freq));
        if (R & 1) {
            for (int c = 0; c < C / 2; c++) {
                int cnt = grid[R / 2][c] + grid[R / 2][C - c - 1];
                freq[cnt]++;
            }
        }
        if (C & 1) {
            for (int r = 0; r < R / 2; r++) {
                int cnt = grid[r][C / 2] + grid[R - r - 1][C / 2];
                freq[cnt]++;
            }
        }
        if ((freq[2] & 1) && freq[1] == 0) freq[1] += 2;
        ans += freq[1];
        if ((R & 1) && (C & 1)) {
            ans += grid[R / 2][C / 2];
        }
        return ans;
    }
};
```

## 3241. Time Taken to Mark All Nodes

### Solution 1:  reroot dp on tree, dfs

```cpp
class Solution {
public:
    int N;
    vector<vector<int>> adj;
    vector<int> ans, max_d1, max_d2, max_c1, max_c2, max_p;
    void dfs1(int u, int p = -1) {
        for (int v : adj[u]) {
            if (v == p) continue;
            dfs1(v, u);
            int cd = max_d1[v] + (v % 2 == 0 ? 2 : 1);
            if (cd > max_d1[u]) {
                max_d2[u] = max_d1[u];
                max_c2[u] = max_c1[u];
                max_d1[u] = cd;
                max_c1[u] = v;
            } else if (cd > max_d2[u]) {
                max_d2[u] = cd;
                max_c2[u] = v;
            }
        }
    }
    void dfs2(int u, int p = -1) {
        ans[u] = max(max_p[u], max_d1[u]);
        for (int v : adj[u]) {
            if (v == p) continue;
            if (v != max_c1[u]) {
                max_p[v] = max(max_p[u], max_d1[u]) + (u % 2 == 0 ? 2 : 1);
            } else {
                max_p[v] = max(max_p[u], max_d2[u]) + (u % 2 == 0 ? 2 : 1);
            }
            dfs2(v, u);
        }
    }
    vector<int> timeTaken(vector<vector<int>>& edges) {
        N = edges.size() + 1;
        adj.assign(N, {});
        for (const auto &edge : edges) {
            int u = edge[0], v = edge[1];
            adj[u].push_back(v);
            adj[v].push_back(u);
        }
        max_d1.assign(N, 0);
        max_d2.assign(N, 0);
        max_c1.assign(N, -1);
        max_c2.assign(N, -1);
        dfs1(0);
        ans.resize(N);
        max_p.assign(N, 0);
        dfs2(0);
        return ans;
    }
};
```