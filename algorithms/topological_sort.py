"""
BFS Kahn's Algorithm for topological sort
"""
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