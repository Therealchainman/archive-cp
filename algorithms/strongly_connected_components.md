# STRONGLY CONNECT COMPONENTS

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