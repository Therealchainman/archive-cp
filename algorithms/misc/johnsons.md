# Johnsons Algorithm

## Weighted Directed Graphs, Negative Edge Weights, No Negative Cycles

Purpose:
    Computes shortest path distances between every ordered pair of vertices
    in a weighted directed graph.

    In other words, for every source vertex u and every target vertex v,
    it computes the minimum total edge weight of any path from u to v.

Use this when:
    - You need all-pairs shortest paths.
    - The graph may contain negative edge weights.
    - The graph does NOT contain a negative-weight cycle.
    - The graph is sparse enough that running Dijkstra from every node is
      better than using Floyd-Warshall.

Do NOT use this when:
    - The graph has a negative-weight cycle.
      In that case, shortest paths are not well-defined.
    - The graph is small and dense.
      Floyd-Warshall is simpler and may be fine.
    - All edge weights are already non-negative.
      Then just run Dijkstra from every vertex directly.
    - You only need shortest paths from one source.
      Use Bellman-Ford if negative edges exist, otherwise Dijkstra.

High-level idea:
    Johnson's algorithm converts a graph with possible negative edge weights
    into an equivalent graph with only non-negative edge weights.

    It does this by computing a potential value h[v] for every vertex v using
    Bellman-Ford. Then every edge (u -> v) with weight w is reweighted as:

        w'(u, v) = w(u, v) + h[u] - h[v]

    This reweighting has two important properties:

        1. Every new edge weight w' is non-negative.
           That makes Dijkstra valid.

        2. Shortest paths are preserved.
           The actual numeric distance changes, but the best path from u to v
           remains the same.

    After running Dijkstra on the reweighted graph, the original distance is
    restored using:

        dist_original(u, v) = dist_reweighted(u, v) + h[v] - h[u]

What it returns:
    A matrix dist where:

        dist[i][j] = shortest path distance from i to j

    If j is unreachable from i, dist[i][j] stays INF.

    If the graph contains a negative cycle, the algorithm returns an empty
    matrix.

Time complexity:
    Let V = number of vertices and E = number of edges.

    Bellman-Ford:
        O(VE)

    Dijkstra from every vertex:
        O(V * (E log V))

    Total:
        O(VE + V * E log V)

    This is usually good for sparse graphs.

Space complexity:
    O(V^2 + E)

    The output matrix itself takes O(V^2).

Important implementation detail:
    A correct Johnson implementation adds one extra "super source" vertex s
    with zero-weight edges to every original vertex:

        s -> 0 with weight 0
        s -> 1 with weight 0
        ...
        s -> n - 1 with weight 0

    Then Bellman-Ford is run from that super source.

    This guarantees Bellman-Ford can reach every vertex and compute a valid
    potential h[v] for all vertices.

    If you do not add this super source, the algorithm is incomplete because
    some vertices may be unreachable from the Bellman-Ford source, causing
    their h[v] values to remain INF.

Negative cycles:
    Bellman-Ford is used only to detect negative cycles and compute the
    potential array h.

    If Bellman-Ford detects a negative cycle, Johnson's algorithm cannot
    continue, because shortest paths may be undefined.

Why the reweighting works:
    For any path:

        u -> a -> b -> ... -> v

    The reweighted path cost becomes:

        original_path_cost + h[u] - h[v]

    All intermediate h values cancel out.

    So every path from u to v gets shifted by the same amount h[u] - h[v].
    Because all paths between the same endpoints are shifted equally, the
    shortest path stays the shortest path.

Common pitfalls:
    1. Forgetting the super source.
    2. Mutating the input edge weights in-place.
    3. Using int when distances can overflow.
    4. Using INT_MAX and then doing dist[u] + w without guarding overflow.
    5. Running Dijkstra on edges that are still negative.
    6. Forgetting to convert Dijkstra distances back to original weights.

```cpp
vector<int> bellmanFord(int n, int src, vector<vector<int>>& edges) {
    vector<int> dist(n, INT_MAX);
    dist[src] = 0;
    for (int i = 0; i < n-1; i++) {
        bool any_relaxed = false;
        for (auto& e : edges) {
            int u = e[0], v = e[1], w = e[2];
            if (dist[u] != INT_MAX && dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                any_relaxed = true;
            }
        }
        if (!any_relaxed) break;
    }
    for (auto& e : edges) {
        int u = e[0], v = e[1], w = e[2];
        if (dist[u] != INT_MAX && dist[u] + w < dist[v]) {
            return {};
        }
    }
    return dist;
}

vector<int> dijkstra(int n, int src, vector<vector<pair<int, int>>>& adj) {
    vector<int> dist(n, INT_MAX);
    dist[src] = 0;
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    pq.push({0, src});
    while (!pq.empty()) {
        int u = pq.top().second;
        int d = pq.top().first;
        pq.pop();
        if (d > dist[u]) continue;
        for (auto& e : adj[u]) {
            int v = e.first, w = e.second;
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});
            }
        }
    }
    return dist;
}

vector<vector<int>> johnsons(int n, vector<vector<int>>& edges) {
    vector<int> h = bellmanFord(n, n-1, edges);
    if (h.empty()) return {};
    for (auto& e : edges) {
        int u = e[0], v = e[1], w = e[2];
        e[2] = w + h[u] - h[v];
    }
    if (h.empty()) return {};
    vector<vector<pair<int, int>>> adj(n);
    for (auto& e : edges) {
        int u = e[0], v = e[1], w = e[2];
        adj[u].emplace_back(v, w);
    }
    vector<vector<int>> dist(n);
    for (int i = 0; i < n; i++) {
        dist[i] = dijkstra(n, i, adj);
        for (int j = 0; j < n; j++) {
            if (dist[i][j] != INT_MAX) {
                dist[i][j] += h[j] - h[i];
            }
        }
    }
    return dist;
}
```