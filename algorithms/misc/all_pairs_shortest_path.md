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

## All Pairs Shortest Paths using repeated Dijkstra, Undirected Graph, Non-negative weights

All-Pairs Shortest Paths using repeated Dijkstra.

What this computes:
- Given a weighted graph with n nodes and edges {u, v, w},
    this computes the shortest path distance between every pair of nodes.
- The result is a matrix dist where:

        dist[src][v] = minimum total weight needed to travel from src to v

- If v cannot be reached from src, then dist[src][v] will be INF.

How it works:
- Build an adjacency list from the edge list.
- For every node src from 0 to n - 1, run Dijkstra's algorithm.
- Each Dijkstra run computes the shortest distance from that src to every
    other node.
- Combining all runs gives all-pairs shortest paths.

Requirements:
- Edge weights must be non-negative.
- Nodes are expected to be numbered from 0 to n - 1.
- This implementation treats edges as undirected:

        adj[u].push_back({v, w});
        adj[v].push_back({u, w});

    Remove the second line if the graph is directed.

Template parameter:
- Weight is the numeric type used for edge weights and distances.
- Common choices are int and long long.
- Use long long when path costs may exceed int range.

Time complexity:
- One Dijkstra run: O((V + E) log V)
- Repeated for every source: O(V * (V + E) log V)

Space complexity:
- Adjacency list: O(V + E)
- Distance matrix: O(V^2)

Notes:
- Dijkstra does not work correctly with negative edge weights.
- INF_VALUE<Weight>() is used as the unreachable distance sentinel.
- Using numeric_limits<Weight>::max() / 4 is usually safer than max(),
    because relaxations compute dist[u] + w.

```cpp
using int64 = long long;
template <class Weight>
constexpr Weight INF_VALUE()
{
    return numeric_limits<Weight>::max();
}

template <class Weight>
vector<Weight> dijkstra(int n, int src, const vector<vector<pair<int, Weight>>> &adj)
{
    const Weight INF = INF_VALUE<Weight>();
    vector<Weight> dist(n, INF);
    dist[src] = 0;
    priority_queue<pair<Weight, int>, vector<pair<Weight, int>>, greater<pair<Weight, int>>> minheap;
    minheap.emplace(0, src);

    while (!minheap.empty())
    {
        auto [d, u] = minheap.top();
        minheap.pop();

        if (d != dist[u])
            continue;

        for (const auto &[v, w] : adj[u])
        {
            if (dist[u] + w < dist[v])
            {
                dist[v] = dist[u] + w;
                minheap.emplace(dist[v], v);
            }
        }
    }
    return dist;
}

template <class Weight>
vector<vector<Weight>> shortestPaths(int n, const vector<array<Weight, 3>> &edges)
{
    vector<vector<pair<int, Weight>>> adj(n);
    for (const auto &e : edges)
    {
        int u = e[0], v = e[1];
        Weight w = e[2];
        adj[u].emplace_back(v, w);
        adj[v].emplace_back(u, w);
    }
    vector<vector<Weight>> dist(n);
    for (int src = 0; src < n; src++)
    {
        dist[src] = dijkstra<Weight>(n, src, adj);
    }
    return dist;
}
vector<vector<int64>> costDist = shortestPaths<int64>(n, costEdges);
```