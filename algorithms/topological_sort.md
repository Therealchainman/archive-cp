# topological sorting


topological sorting/ordering of nodes in directed graph

## BFS Kahn's Algorithm for topological sort

```py
from collections import defaultdict, deque
def KahnsAlgorithm(n, relations)
    visited = [0]*(n+1)
    graph = defaultdict(list)
    indegrees = [0]*(n+1)
    for u, v in relations:
        graph[u].append(v)
        indegrees[v] += 1
    num_semesters = studied_count = 0
    queue = deque()
    for node in range(1,n+1):
        if indegrees[node] == 0:
            queue.append(node)
            studied_count += 1
    while queue:
        num_semesters += 1
        sz = len(queue)
        for _ in range(sz):
            node = queue.popleft()
            for nei in graph[node]:
                indegrees[nei] -= 1
                if indegrees[nei] == 0 and not visited[nei]:
                    queue.append(nei)
                    studied_count += 1
    return num_semesters if studied_count == n else -1
```

## Simple generic solution using bfs

Creates a topological ordering of nodes
n = number of nodes
nodes = iterable of nodes in directed graph
adj_list = dictionary of lists adjacency list representation of directed graph for the nodes given

returns: topological ordering of nodes if possible

```py
def topological_ordering(n, nodes, adj_list):
    indegrees = Counter()
    queue = deque()
    for neis in adj_list.values():
        for nei in neis: indegrees[nei] += 1
    for node in nodes:
        if indegrees[node] == 0: queue.append(node)
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for nei in adj_list[node]:
            indegrees[nei] -= 1
            if indegrees[nei] == 0: queue.append(nei)
    return topo_order if len(topo_order) == n else []
```

### C++ implementation of general topological ordering

Given the adjacency list, and the number of nodes, and nodes are numbered from 0 to n - 1.

If there is a cycle or something and it cannot find a valid topological ordering it will return an empty vector.

```cpp
vector<int> topologicalOrdering(const int N, const vector<vector<int>>& adj) {
    vector<int> indegrees(N, 0);
    for (int u = 0; u < N; u++) {
        for (int v : adj[u]) {
            indegrees[v]++;
        }
    }
    vector<int> order;
    deque<int> queue;
    for (int i = 0; i < N; i++) {
        if (indegrees[i] == 0) queue.push_back(i);
    }
    while (!queue.empty()) {
        int u = queue.front();
        order.push_back(u);
        queue.pop_front();
        for (int v : adj[u]) {
            indegrees[v]--;
            if (indegrees[v] == 0) queue.push_back(v);
        }
    }
    return order.size() == N ? order : vector<int>();
}
```