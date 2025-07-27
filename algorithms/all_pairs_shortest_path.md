# All Pairs Shortest path 

## Floyd Warshall Algorithm

This is a classic O(V^3) all‑pairs shortest‑paths (APSP) algorithm that works on both directed and undirected graphs, and for dense graphs it’s often simpler (and even faster in practice) than running Dijkstra’s from every node.

```cpp
const int64 INF = numeric_limits<int64>::max();
vector<vector<int64>> dist;

void floyd_warshall(int n) {
    // floyd warshall, all pairs shortest path
    for (int k = 0; k < n; k++) {  // Intermediate vertex
        for (int i = 0; i < n; i++) {  // Source vertex
            for (int j = 0; j < n; j++) {  // Destination vertex
                if (dist[i][k] == INF || dist[k][j] == INF) continue;
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
            }
        }
    }
}

dist.assign(N, vector<int64>(N, INF));
for (int i = 0; i < M; i++) {
    int u, v, w;
    cin >> u >> v >> w;
    u--; v--;
    adj[u].emplace_back(v, w);
    adj[v].emplace_back(u, w);
    dist[u][v] = dist[v][u] = w;
}
for (int i = 0; i < N; i++) {
    dist[i][i] = 0;
}
floyd_warshall(N);
```

Another cool thing about floyd warshall is that it works on directed or undirectd graphs, it doesn't matter, if it is an undirected graph it just has the additional following symmetric property which is the dist[i][j] = dist[j][i]

## Dynamic single edge update

Another useful thing is you can perform edge updates, so if there a decrase on the weight between node u -> v. 

If you update the weight of edge (u, v), you can perform update to recalculate the shortest distance between all pair of nodes given that updated edge in O(N^2) time. 

```cpp
void relax_edge(int u, int v, int64 w) {
    // Relax edges for the new edge u-v with weight w
    for (int i = 0; i < N; ++i) {
        if (dist[i][u] == INF) continue;
        for (int j = 0; j < N; ++j) {
            if (dist[v][j] == INF) continue;
            dist[i][j] = min(dist[i][j], dist[i][u] + w + dist[v][j]); // i -> u -> v -> j
        }
    }
}

// always call it in both direction, for directed graph with appropriate weights. 
relax_edge(u, v, w);
relax_edge(v, u, w);
```