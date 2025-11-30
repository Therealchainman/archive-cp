# Small to Large Merging Optimization in Tree

This is an algorithm that can bring the time complexity from O(n^2) to O(nlogn) in some problems related to trees.  

The idea is that you set the information for the current vertex to that of the child with largest data.  And then you merge all the children nodes with 
smaller data to the child with the largest data. This way, you can merge the children nodes in O(logn) time.

## Example implementation in C++

This is one where you are merging sets, so find the largest set, and than merge all the smaller sets into the larger set.

```cpp
const int MAXN = 2e5 + 5;
int N, color;
vector<int> adj[MAXN];
int ans[MAXN];
set<int> s[MAXN];

void dfs(int u, int p) {
    for (int v : adj[u]) {
        if (v == p) continue;
        dfs(v, u);
        if (s[v].size() > s[u].size()) {
            swap(s[u], s[v]);
        }
        for (int x : s[v]) {
            s[u].insert(x);
        }
    }
    ans[u] = s[u].size();
}

void solve() {
    cin >> N;
    for (int i = 0; i < N; i++) {
        cin >> color;
        s[i].clear();
        s[i].insert(color);
    }
    for (int i = 0; i < N - 1; i++) {
        int u, v;
        cin >> u >> v;
        u--, v--;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    dfs(0, -1);
    for (int i = 0; i < N; i++) {
        cout << ans[i] << " ";
    }
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    while (T--) {
        solve();
    }
    return 0;
}
```

## Here is an example of it applied to a mucher more difficult problem 

This time you are storing multiple values for each node, you are storing the count for each color for that node, and the depth_sum for each color for that node. 
The math for this one is a little crazy, but I have the derivation in notion notes.

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