# Path Graph

A tree where the root node and leaf node have a degree of 1 and all other nodes have a degree of 2.

all vertices and edges lie on a straight line.

Related to a path in graph theory. 
path: a finite or infinite sequence of edges which joins a sequence of vertices which, by most definitions, are all distinct 

```py
def is_path_graph(n: int, edges: List[List[int]]) -> bool:
    visited = [False]*n
    adj_list = defaultdict(list)
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)
    def is_cycle(node, parent_node):
        visited[node] = True
        for nei in adj_list[node]:
            if not visited[nei]:
                if is_cycle(nei, node):
                    return True
            elif nei != parent_node:
                return True
        return False
    cycle = is_cycle(0, None)
    return True if not cycle and sum(visited) == n else False
```