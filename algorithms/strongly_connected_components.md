# STRONGLY CONNECT COMPONENTS

- every non-trivial strongly connected component contains at least one directed cycle.

## TARJAN'S ALGORITHM

A single dfs algorithm that use lowlinke and index to find strongly connected components.

construct an  adjacency list for the edges, not shown in code below. 

This particular implementation is computing the largest strongly connected component.

```py
res = time = 0
disc, low, on_stack = [0]*n, [0]*n, [0]*n
stack = []
def dfs(node):
    nonlocal res, time
    time += 1
    disc[node] = time
    low[node] = disc[node]
    on_stack[node] = 1
    stack.append(node)
    for nei in adj_list[node]:
        if not disc[nei]: dfs(nei)
        if on_stack[nei]: low[node] = min(low[node], low[nei])
    # found scc
    if disc[node] == low[node]:
        size = 0
        while stack:
            snode = stack.pop()
            size += 1
            on_stack[snode] = 0
            low[snode] = low[node]
            if snode == node: break
        res = max(res, size)
for i in range(n):
    if disc[i]: continue
    dfs(i)
```

## Computing the number of SCCs and determining which component each node is belongs

This implementation is computing the number of strongly connected components. And also the id of each node in the scc.

```py
time = num_scc = 0
scc_ids = [0]*(n + 1)
disc, low, on_stack = [0]*(n + 1), [0]*(n + 1), [0]*(n + 1)
stack = []
def dfs(node):
    nonlocal time, num_scc
    time += 1
    disc[node] = time
    low[node] = disc[node]
    on_stack[node] = 1
    stack.append(node)
    for nei in adj_list[node]:
        if not disc[nei]: dfs(nei)
        if on_stack[nei]: low[node] = min(low[node], low[nei])
    # found scc
    if disc[node] == low[node]:
        num_scc += 1
        while stack:
            snode = stack.pop()
            on_stack[snode] = 0
            low[snode] = low[node]
            scc_ids[snode] = num_scc
            if snode == node: break
for i in range(1, n + 1):
    if disc[i]: continue
    dfs(i)
```

## CPP variant

```cpp
int N, M, timer, scc_count;
vector<vector<int>> adj;
vector<int> disc, low, comp;
stack<int> stk;
vector<bool> on_stack;

void dfs(int u) {
    disc[u] = low[u] = ++timer;
    stk.push(u);
    on_stack[u] = true;
    for (int v : adj[u]) {
        if (not disc[v]) dfs(v);
        if (on_stack[v]) low[u] = min(low[u], low[v]);
    }
    if (disc[u] == low[u]) { // found scc
        scc_count++;
        while (!stk.empty()) {
            int v = stk.top();
            stk.pop();
            on_stack[v] = false;
            low[v] = low[u];
            comp[v] = scc_count;
            if (v == u) break;
        }
    }
}
```

## CONDENSATION GRAPH

If each strongly connected component is contracted to a single vertex, the resulting graph is a directed acyclic graph, the condensation of G. A directed graph is acyclic if and only if it has no strongly connected subgraphs with more than one vertex, because a directed cycle is strongly connected and every non-trivial strongly connected component contains at least one directed cycle.

It is obvious, that strongly connected components do not intersect each other, i.e. this is a partition of all graph vertices. Thus we can give a definition of condensation graph  
as a graph containing every strongly connected component as one vertex. Each vertex of the condensation graph corresponds to the strongly connected component of graph.

Condensation graph can be used for dynamic programming on a directed graph and so on.

Example of using condensation graph + topological sort to use dynamic programming on the resulting condensation graph.  this is basically just collecting maximum coin value in the graph starting at any vertex in graph.  

```py
coins = [0] + list(map(int, input().split()))
# PHASE 0: CONSTRUCT ADJACENCY LIST REPRESENTATION OF GRAPH
adj_list = [[] for _ in range(n + 1)]
for _ in range(m):
    a, b = map(int, input().split())
    adj_list[a].append(b)
# PHASE 1: FIND STRONGLY CONNECTED COMPONENTS
time = num_scc = 0
scc_ids = [0]*(n + 1)
disc, low, on_stack = [0]*(n + 1), [0]*(n + 1), [0]*(n + 1)
stack = []
def dfs(node):
    nonlocal time, num_scc
    time += 1
    disc[node] = time
    low[node] = disc[node]
    on_stack[node] = 1
    stack.append(node)
    for nei in adj_list[node]:
        if not disc[nei]: dfs(nei)
        if on_stack[nei]: low[node] = min(low[node], low[nei])
    # found scc
    if disc[node] == low[node]:
        num_scc += 1
        while stack:
            snode = stack.pop()
            on_stack[snode] = 0
            low[snode] = low[node]
            scc_ids[snode] = num_scc
            if snode == node: break
for i in range(1, n + 1):
    if disc[i]: continue
    dfs(i)
# PHASE 2: CONSTRUCT CONDENSATION GRAPH
scc_adj_list = [[] for _ in range(num_scc + 1)]
indegrees = [0]*(num_scc + 1)
# condensing the values of the coins into it's scc
val_scc = [0]*(num_scc + 1)
for i in range(1, n + 1):
    val_scc[scc_ids[i]] += coins[i]
    for nei in adj_list[i]:
        if scc_ids[i] != scc_ids[nei]:
            indegrees[scc_ids[nei]] += 1
            scc_adj_list[scc_ids[i]].append(scc_ids[nei])
# PHASE 3: DO TOPOLOGICAL SORT ON CONDENSATION GRAPH WITH MEMOIZATION FOR MOST COINS COLLECTED IN EACH NODE IN CONDENSATION GRAPH
stack = []
memo = [0]*(num_scc + 1)
for i in range(1, num_scc + 1):
    if indegrees[i] == 0:
        stack.append(i)
        memo[i] = val_scc[i]
while stack:
    node = stack.pop()
    for nei in scc_adj_list[node]:
        indegrees[nei] -= 1
        memo[nei] = max(memo[nei], memo[node] + val_scc[nei])
        if indegrees[nei] == 0: stack.append(nei)
print(max(memo))
```
