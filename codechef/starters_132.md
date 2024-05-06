# Starters 132

## Random Array

### Solution 1:  dynammic programming, linearity of expectation value, expected value, probability

```cpp

```

## Goodness Over Good

### Solution 1: 

```py

```

## Mex Path

### Solution 1:  dijkstra, directed graph, mex, forward edges, backward edges

```cpp
const int MAXN = 5e5 + 5;
int N;
int arr[MAXN];
vector<vector<pair<int, int>>> adj;
vector<int> vis, last;

int dijkstra(int src, int dst) {
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    vector<bool> vis(N, false);
    pq.emplace(0, src);
    while (!pq.empty()) {
        auto [cost, u] = pq.top();
        pq.pop();
        if (u == dst) return cost;
        if (vis[u]) continue;
        vis[u] = true;
        for (auto [v, w] : adj[u]) {
            pq.emplace(cost + w, v);
        }
    }
    return -1;
}

void solve() {
    cin >> N;
    adj.assign(N, vector<pair<int, int>>());
    for (int i = 0; i < N; i++) cin >> arr[i];
    vis.assign(N + 2, 0);
    int mex = 0;
    // forward edges that start from first node
    for (int i = 0; i < N; i++) {
        vis[arr[i]] = 1;
        while (vis[mex]) mex++;
        adj[0].emplace_back(i, mex);
    }
    fill(vis.begin(), vis.end(), 0);
    mex = 0;
    // forward edges that point to last node
    for (int i = N - 1; i >= 0; i--) {
        vis[arr[i]] = 1;
        while (vis[mex]) mex++;
        adj[i].emplace_back(N - 1, mex);
    }
    // BACKWARD EDGES
    for (int i = 1; i < N; i++) {
        adj[i].emplace_back(i - 1, 0);
    }
    // FORWARD EDGES
    last.assign(N + 1, -1);
    for (int i = 0; i < N; i++) {
        if (last[arr[i]] != -1) {
            adj[last[arr[i]] + 1].emplace_back(i - 1, arr[i]);
        }
        last[arr[i]] = i;
    }
    int ans = dijkstra(0, N - 1);
    cout << ans << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    int T;
    cin >> T;
    while (T--) {
        solve();
    }
    return 0;
}
```