# Single Source Shortest Path Problem

In graph theory, the shortest path problem is the problem of finding a path between two vertices (or nodes) in a graph such that the sum of the weights of its constituent edges is minimized.

## Dijkstra's Algorithm

Dijkstra's algorithm can solve this problem if the graph contained positive edge weights.

For a given source node in the graph, the algorithm finds the shortest path between that node and every other.  It can also be used for finding the shortest paths from a single node to a single destination node by stopping the algorithm once the shortest path to the destination node has been determined. 

At a high level, Dijkstra's algorithm assumes that once a node has been visited with the smallest known cost, no cheaper path to it will ever be found. To ensure this holds, the following property must be satisfied:

- Edge weights must be non-negative.
- Cost of a path must be non-decreasing as the path extends.

This ensures that once you find a "shortest path" to a node, no alternative path discovered later will be better.

So commonly it is used for path sum and path max. 

### C++ implementation for source to all nodes

```cpp
namespace Graph {
    template<typename CostType, typename Transition>
    vector<CostType> dijkstra(int src, Transition transitionFunction) {
        priority_queue<pair<CostType, int>, vector<pair<CostType, int>>, greater<pair<CostType, int>>> minheap;
        vector<CostType> dist(N, numeric_limits<CostType>::max());
        minheap.emplace(0, src);
        dist[src] = 0;
        while (!minheap.empty()) {
            auto [cost, u] = minheap.top();
            minheap.pop();
            if (dist[u] < cost) continue;
            for (auto [v, w] : adj[u]) {
                CostType ncost = transitionFunction(cost, w);
                if (dist[v] <= ncost) continue;
                dist[v] = ncost;
                minheap.emplace(ncost, v);
            }
        }
        return dist;
    }
}
```

```cpp
Graph::dijkstra<int>(src, [](int cost, int w) { return cost + w; });
```

### C++ implementation for source to destination node

```cpp
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
            if (vis[v]) continue;
            pq.emplace(cost + w, v);
        }
    }
    return -1;
}
```
```

### Python implementation for source to destination node

This is a good implementation that has a constant time improvement over a few other ways you could implement it.  The vis[v]: continue prevents it from adding values that are not necessary.

w >= 0

```py
def dijkstra(adj, src, dst):
    N = len(adj)
    min_heap = [(0, src)]
    vis = [0] * N
    while min_heap:
        cost, u = heapq.heappop(min_heap)
        if u == dst: return cost
        if vis[u]: continue
        vis[u] = 1
        for v, w in adj[u]:
            if vis[v]: continue
            heapq.heappush(min_heap, (cost + w, v))
    return -1
```

### Python implementation for all nodes

It is the same, just doesn't early terminate

Is this one still necessary? I think the one above is the better approach

```py
import math
def dijkstra(adj, src):
    N = len(adj)
    min_heap = [(0, src)]
    vis = [math.inf] * N
    while min_heap:
        cost, u = heapq.heappop(min_heap)
        if vis[u] < math.inf: continue
        vis[u] == cost
        for v, w in adj[u]:
            if vis[v] < math.inf: continue
            heapq.heappush(min_heap, (cost + w, v))
    return -1
```

## dijkstra for when you need the distance to all nodes from source node

```py
import math
import heapq
def dijkstra(adj, src):
    N = len(adj)
    min_heap = [(0, src)]
    dist = [math.inf] * N
    while min_heap:
        cost, u = heapq.heappop(min_heap)
        if cost >= dist[u]: continue
        dist[u] = cost
        for v, w in adj[u]:
            if cost + w < dist[v]: heapq.heappush(min_heap, (cost + w, v))
    return dist
```

for weighted undirected graph

```cpp

```

## dijkstra for matrix or grid search, where you can only move in 4 directions

For this particular implementation it is multisource, and every position that is on the boundary is a source.  Then it computes the minimum distance to get from boundary to any other position inside the grid.  In this case it finds the cheapest way to get from every position to escape the grid.

```cpp
void dijkstra() {
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    vis.assign(N * M, false);
    for (int i = 0; i < N; i++) {
        pq.emplace(grid[i][0], mat_id(i, 0));
        pq.emplace(grid[i][M - 1], mat_id(i, M - 1));
    }
    for (int i = 0; i < M; i++) {
        pq.emplace(grid[0][i], mat_id(0, i));
        pq.emplace(grid[N - 1][i], mat_id(N - 1, i));
    }
    while (!pq.empty()) {
        auto [cost, u] = pq.top();
        pq.pop();
        if (vis[u]) continue;
        vis[u] = true;
        auto [i, j] = mat_ij(u);
        dist[i][j] = cost;
        for (auto [ni, nj] : neighborhood(i, j)) {
            if (!in_bounds(ni, nj)) continue;
            if (vis[mat_id(ni, nj)]) continue;
            pq.emplace(cost + grid[ni][nj], mat_id(ni, nj));
        }
    }
}
```

## Dijkstra with Backtracking

You can use parent map to keep track to backtrack, than you just move backwards from end node.

```py
def dijkstra(src, skip_mod):
    minheap = [(0, src)]
    dist = [math.inf] * n
    dist[src] = 0
    parent = {src: None}
    while minheap:
        cost, node = heapq.heappop(minheap)
        if cost > dist[node]: continue
        for nei, wei in adj_list[node]:
            if wei == -1:
                if skip_mod: continue
                wei = lower_bound
            ncost = cost + wei
            if ncost < dist[nei]:
                dist[nei] = ncost
                heapq.heappush(minheap, (ncost, nei))
                parent[nei] = node
    return dist, parent
```

## Dijkstra with unweighted graph

Not really Dijkstra, can just use BFS when graph is unweighted. 

```cpp
int N;
vector<vector<int>> adj;
vector<int> bfs(int src) {
    queue<int> q;
    vector<int> dist(N, N + 1);
    q.push(src);
    dist[src] = 0;
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (int v : adj[u]) {
            if (dist[u] + 1 < dist[v]) {
                q.push(v);
                dist[v] = dist[u] + 1;
            }
        }
    }
    return dist;
}
```

## Dijkstra from source and destination

This is sometimes useful to calculate the shortest path from source to all nodes and from destination to all nodes.  Then you can use the inverse shortest distance combined with shorest distance to find the second shortest path.