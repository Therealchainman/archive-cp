# Cycle Detection in graphs

## cycle detection in directed graph

### cycle detection with three color mechanism and DFS

```cpp
enum Color {
    WHITE,
    GREY,
    BLACK
};

vector<vector<int>> adj;
vector<int> color;

bool hasCycle(int u) {
    if (color[u] == BLACK) return false;
    if (color[u] == GREY) return true;
    color[u] = GREY;
    bool res = false;
    for (int v : adj[u]) {
        res |= hasCycle(v);
    }
    color[u] = BLACK;
    return res;
}
```

Why GREY→GREY is a cycle
In a DFS, an edge from the current node back to a GREY node means you’ve found a back‐edge into your own active recursion chain—by definition that’s a cycle in a directed graph.

### cycle detection with recursive dfs

```py
visited = set()
in_path = set()
def detect_cycle(node: str) -> bool:
    visited.add(node)
    in_path.add(node)
    nei = adj_list.get(node, None)
    if nei is not None:
        if nei in in_path:
            return True
        if nei not in visited and detect_cycle(nei):
            return True
    in_path.remove(node)
    return False
```

### cycle detection with arrays and save the path for recreate the cycle path

Just need to make sure when recreate cycle path to reverse it cause it is in reverse order

```py
visited = [0] * (n + 1)
in_path = [0] * (n + 1)
path = []
def detect_cycle(node) -> bool:
    path.append(node)
    visited[node] = 1
    in_path[node] = 1
    for nei in adj_list[node]:
        if in_path[nei]: return nei
        if visited[nei]: continue
        res = detect_cycle(nei)
        if res: return res
    in_path[node] = 0
    path.pop()
    return 0

node = detect_cycle(i)
cur = node
cycle = [node]
if cur:
    while path[-1] != node:
        cur = path.pop()
        cycle.append(cur)
    print(len(cycle))
    print(*cycle[::-1])
```

## cycle detection in undirected graph

### cycle detection with recursive dfs

```py
visited = [False]*n
def is_cycle(node, parent_node):
    visited[node] = True
    for nei in adj_list[node]:
        if not visited[nei]:
            if is_cycle(nei, node):
                return True
        elif nei != parent_node:
            return True
    return False
```
