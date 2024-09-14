# All Pairs Shortest path 



## Floyd Warshall Algorithm

Solves it in O(V^3), really good for dense graphs


```cpp
dist.assign(N, vector<int>(N, INF));
for (int i = 0; i < M; i++) {
    int u, v, w;
    cin >> u >> v >> w;
    u--; v--;
    adj[u].push_back({v, w});
    adj[v].push_back({u, w});
    dist[u][v] = dist[v][u] = w;
}
for (int i = 0; i < N; i++) {
    dist[i][i] = 0;
}
// floyd warshall, all pairs shortest path
for (int k = 0; k < N; k++) {  // Intermediate vertex
    for (int i = 0; i < N; i++) {  // Source vertex
        for (int j = 0; j < N; j++) {  // Destination vertex
            dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
        }
    }
}
```