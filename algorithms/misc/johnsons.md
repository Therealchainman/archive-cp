# Johnsons Algorithm

A graph algorithm for solving the all-pairs shortest path problem

```py
from typing import List
import math
import heapq
from itertools import product

def bellmanFord(n: int, src: int, edges: List[List[int]]) -> List[int]:
    dist = [math.inf]*n
    dist[src] = 0
    for _ in range(n-1):
        any_relaxed = False
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                any_relaxed = True
                dist[v] = dist[u] + w
        if not any_relaxed: break
    # check for any negative cycles
    for u, v, w in edges:
        if dist[v] > dist[u] + w: return []
    return dist

def dijkstra(n: int, src: int, adj_list: List[List[int]]) -> List[int]:
    dist = [math.inf]*n
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: continue
        for v, w in adj_list[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    return dist

def johnsons(n: int, edges: List[List[int]]) -> List[List[int]]:
    # create a new vertex that is connected to all other vertices with weight 0
    # new vertex that will be the source for bellman fourd is going to be n
    # run bellman ford to find shortest paths from the new vertex to all other vertices
    dist = bellmanFord(n+1, n, edges + [[n, i, 0] for i in range(n)])
    if not dist: return [] # if it has negative cycle
    # reweight the edges
    for i in range(len(edges)):
        u, v, w = edges[i]
        edges[i][2] = w + dist[u] - dist[v]
    # run dijkstra for each vertex
    adj_list = [[] for _ in range(n)]
    for u, v, w in edges:
        adj_list[u].append((v, w))
    shortest_paths = [dijkstra(n, i, adj_list) for i in range(n)]
    # undo the reweighting
    for u, v in product(range(n), repeat = 2):
        if shortest_paths == math.inf: continue
        shortest_paths[u][v] = shortest_paths[u][v] + dist[v] - dist[u]
    return shortest_paths
```

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
        adj[u].push_back({v, w});
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