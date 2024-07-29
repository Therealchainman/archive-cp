# Steiner Tree Problem

Sometimes called minimum steiner tree problem

Given a graph and a subset of vertices in the graph, a Steiner tree spans through the given subset. The Steiner Tree may contain some vertices which are not in the given subset but are used to connect the vertices of the subset.  The given set of vertices is called Terminal Vertices and other vertices that are used to construct the Steiner tree are called Steiner vertices.  The Steiner Tree Problem is to find the minimum cost of Steiner Tree.

## Example code

Understand dp transitions is important. 

```cpp
const int INF = 1e16;
int N, M, K;
vector<vector<pair<int, int>>> adj;
vector<vector<int>> dp;

void solve() {
    cin >> N >> M >> K;
    K--;
    dp.assign(1 << K, vector<int>(N, INF));
    for (int i = 0; i < K; i++) {
        dp[1 << i][i] = 0; // fixed terminal nodes for steiner tree
    }
    adj.assign(N, vector<pair<int, int>>());
    // construct weighted graph
    for (int i = 0; i < M; i++) {
        int u, v, w;
        cin >> u >> v >> w;
        u--; v--;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }
    for (int mask = 1; mask < (1 << K); mask++) {
        // phase 1 of transitions
        for (int submask = mask; submask > 0; submask = (submask - 1) & mask) {
            for (int i = 0; i < N; i++) {
                dp[mask][i] = min(dp[mask][i], dp[submask][i] + dp[mask - submask][i]); // mask - submask works because it is a submask, this gets the set difference
            }
        }
        // phase 2 of transitions
        // dijkstra part to find shortest path given this bitmask or set of elements in a steiner tree
        // And calculate the shortest path to be able to reach vertex v.
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> minheap;
        for (int i = 0; i < N; i++) {
            minheap.emplace(dp[mask][i], i);
        }
        // shortest distance from any node in the mask or set of nodes (steiner tree) to any other node outside of the current steiner tree. 
        while (!minheap.empty()) {
            auto [dist, u] = minheap.top();
            minheap.pop();
            if (dist > dp[mask][u]) continue;
            for (auto [v, w] : adj[u]) {
                if (dp[mask][u] + w < dp[mask][v]) {
                    dp[mask][v] = dp[mask][u] + w;
                    minheap.emplace(dp[mask][v], v);
                }
            }
        }
    }
    for (int i = K; i < N; i++) {
        cout << dp.end()[-1][i] << endl;
    }
}

signed main() {
    solve();
    return 0;
}
```