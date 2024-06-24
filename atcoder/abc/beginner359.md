# Atcoder Beginner Contest 359

## F - Tree Degree Optimization 

### Solution 1:  min heap, tree

```cpp
int N;
vector<int> A, deg;

struct TreeData {
    int deg, val, idx;
};

class TreeDataComparator {
    public:
        int operator () (const TreeData& d1, const TreeData& d2) {
            int delta1 = d1.deg * d1.deg * d1.val - (d1.deg - 1) * (d1.deg - 1) * d1.val;
            int delta2 = d2.deg * d2.deg * d2.val - (d2.deg - 1) * (d2.deg - 1) * d2.val;
            return delta1 > delta2;
        }
};

void solve() {
    cin >> N;
    A.resize(N);
    deg.assign(N, 1);
    priority_queue<TreeData, vector<TreeData>, TreeDataComparator> minheap;
    for (int i = 0; i < N; i++) {
        cin >> A[i];
        minheap.push({2, A[i], i}); // {next degree, value, index}
    }
    int total_deg = N;
    for (int i = 0; i < N - 2; i++) {
        TreeData d = minheap.top();
        deg[d.idx] = d.deg;
        minheap.pop();
        minheap.push({d.deg + 1, d.val, d.idx});
    }
    int ans = 0;
    for (int i = 0; i < N; i++) {
        ans += deg[i] * deg[i] * A[i];
    }
    cout << ans << endl;
}

signed main() {
    solve();
    return 0;
}
```

## G - Sum of Tree Distance

### Solution 1:  small to large merging, pairing, depth_sum, counts

```cpp
int N, ans;
vector<int> A, depth;
vector<vector<int>> adj;
vector<map<int, int>> depth_sum, cnt;

void dfs(int u, int p) {
    cnt[u][A[u]] = 1;
    depth_sum[u][A[u]] = depth[u];
    for (int v : adj[u]) {
        if (v == p) continue;
        depth[v] = depth[u] + 1;
        dfs(v, u);
        if (cnt[u].size() < cnt[v].size()) {
            swap(cnt[u], cnt[v]);
            swap(depth_sum[u], depth_sum[v]);
        }
        for (auto [color, freq] : cnt[v]) {
            if (cnt[u].find(color) == cnt[u].end()) {
                cnt[u][color] = freq;
                depth_sum[u][color] = depth_sum[v][color];
            } else {
                ans += freq * (depth_sum[u][color] - depth[u] * cnt[u][color]);
                ans += cnt[u][color] * (depth_sum[v][color] - depth[u] * freq);
                cnt[u][color] += freq;
                depth_sum[u][color] += depth_sum[v][color];
            }
        }
    }
}

void solve() {
    cin >> N;
    adj.assign(N, vector<int>());
    for (int i = 0; i < N - 1; i++) {
        int u, v;
        cin >> u >> v;
        u--; v--;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    A.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    depth_sum.assign(N, map<int, int>());
    cnt.assign(N, map<int, int>());
    depth.assign(N, 0);
    ans = 0;
    dfs(0, -1);
    cout << ans << endl;
}

signed main() {
    solve();
    return 0;
}
```