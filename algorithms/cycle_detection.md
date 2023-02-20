# Cycle Detection in graphs

## cycle detection in directed graph

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
