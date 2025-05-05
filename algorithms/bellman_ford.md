# Bellman Ford Algorithm

A graph algorithm

## Bellman ford algorithm for single source shortest path problem

- works for negative edge weights
- can detect negative cycles in graph
- O(VE) time complexity

```py
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
```

```cpp
vector<int64> bellmanFord(int src) {
    vector<int64> dist(N, INF);
    dist[src] = 0;
    // Relax up to n-1 times, early exit if no change
    for (int i = 0; i < N - 1; i++) {
        bool any_relaxed = false;
        for (const auto& e : edges) {
            int u = e[0], v = e[1], w = e[2];
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                any_relaxed = true;
            }
        }
        if (!any_relaxed) break;
    }
    // Check for negative cycles
    for (const auto& e : edges) {
        int u = e[0], v = e[1], w = e[2];
        if (dist[u] + w < dist[v]) {
            // Negative cycle detected
            return {};
        }
    }
    return dist;
}
```

## Bellman ford algorithm for single source shortest path problem with negative cycles


```py
def bellmanFord(n: int, src: int, edges: List[List[int]]) -> List[int]:
    dist = [math.inf]*n
    parents = [-1]*n
    x = None
    dist[src] = 0
    for _ in range(n):
        any_relaxed = False
        x = None
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                any_relaxed = True
                dist[v] = dist[u] + w
                parents[v] = u
                x = v
        if not any_relaxed: break
    # check for any negative cycles
    for u, v, w in edges:
        if dist[v] > dist[u] + w: return []
    return dist
```