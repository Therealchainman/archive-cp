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