# Articulation Points and Bridges

Tarjan's algorithm allows you to find articulation points and bridges in undirected graphs in O(V + E) time complexity.

An articulation point is a vertex whose removal increases the number of connected components in the graph. A bridge is an edge whose removal increases the number of connected components in the graph.

The detection of these is done by discovery time and low time of each vertex in a dfs traversal of the graph.

The only difference needed in the algorithm is that for bridges disc(u) < low(v), while for articulation points disc(u) <= low(v).

But there is also another edge case for articulation points for when the root of the dfs traversal has more than one child. In this case, the root is an articulation point.

## cpp implementation for Bridges

```cpp
int timer;
vector<int> disc, low;

void dfs(int u, int p) {
    disc[u] = low[u] = ++timer;
    for (int v : adj[u]) {
        if (v == p) continue;
        if (!disc[v]) {
            dfs(v, u);
            if (disc[u] < low[v]) {
                bridge_count++;
            }
            low[u] = min(low[u], low[v]);
        } else {
            low[u] = min(low[u], disc[v]); // back edge, disc[v] because of ap of cycle
        }
    }
}

// initialization of values
timer = 0;
disc.assign(n, 0);
low.assign(n, 0);
```

## With additional ability to track the size of each subtree, if you were converting the graph to a bridge tree

```cpp
int dfs(int u, int p) {
    int sz = 0;
    disc[u] = low[u] = ++timer;
    for (int v : adj[u]) {
        if (v == p) continue;
        if (!disc[v]) {
            int csz = dfs(v, u);
            if (disc[u] < low[v]) {
                bridges.push_back(csz);
            }
            sz += csz;
            low[u] = min(low[u], low[v]);
        } else {
            low[u] = min(low[u], disc[v]); // back edge, disc[v] because of ap of cycle
        }
    }
    return ++sz;
}


```


need to rewrite this, but easy just do as done above in python.
```py
from math import inf
from typing import Optional, List

class Tarjans:
    def __init__(self, n: int, edges: List[int]):
        self.bridges = []
        self.unvisited = inf
        self.disc = [self.unvisited]*n 
        self.low = [self.unvisited]*n
        self.cnt = 0
        self.adjList = [[] for _ in range(n)]
        for u, v in edges:
            self.adjList[u].append(v)
            self.adjList[v].append(u)
    def dfs(self, node: int, parent_node: Optional[int] = None) -> None:
        if self.disc[node] != self.unvisited:
            return
        self.disc[node] = self.low[node] = self.cnt
        self.cnt += 1
        for nei_node in self.adjList[node]:
            if nei_node == parent_node: continue
            self.dfs(nei_node, node)
            if self.disc[node] < self.low[nei_node]:
                self.bridges.append([node, nei_node])
            self.low[node] = min(self.low[node], self.low[nei_node])
    def get_bridges(self) -> List[int]:
        return self.bridges
```py