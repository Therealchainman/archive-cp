"""
All Pairs Shortest Paths Problem

Solved with Floyd Warshall algorithm O(v^3)
Shortest path between all pairs of vertices in a graph

Can also use BFS O(V^2 * (V+E))
"""
from collections import deque, defaultdict
from typing import List, Dict, Tuple

"""
shortest path by bfs algorithm with O(V * (V+E)) time complexity with adjacency list worse case is when dense graph and E = V^2 then O(V^3) is the time complexity
- non-negative edge weights
"""

def bfs(src: int, dst: int, adj_list: Dict[int, List[int]]) -> int:
    dist = 0
    queue = deque([src])
    vis = set([src])
    while queue:
        sz = len(queue)
        for _ in range(sz):
            node = queue.popleft()
            if node == dst: return dist
            for nei in adj_list[node]:
                if nei in vis: continue
                queue.append(nei)
                vis.add(nei)
        dist += 1
    return -1

def shortest_path_bfs(edges: List[List[int]]) -> Dict[Tuple[int, int], int]:
    adj_list = defaultdict(list)
    nodes = set()
    for src, dst in edges:
        adj_list[src].append(dst)
        adj_list[dst].append(src)
        nodes.update([src, dst])
    shortestPath = {}
    for src in nodes:
        for dst in nodes:
            if src == dst: continue
            shortestPath[(src, dst)] = bfs(src, dst, adj_list)
    return shortestPath

"""
shortest path by floyd warshall algorithm with O(V^3) time complexity
"""

def shortest_path_floyd_warshall(edges: List[List[int]]) -> Dict[Tuple[int, int], int]:
    adj_list = defaultdict(list)
    nodes = set()
    for src, dst in edges:
        adj_list[src].append(dst)
        adj_list[dst].append(src)
        nodes.update([src, dst])
    shortestPath = {}
    for src in nodes:
        for dst in nodes:
            if src == dst: continue
            shortestPath[(src, dst)] = bfs(src, dst, adj_list)
    for k in nodes:
        for i in nodes:
            for j in nodes:
                if shortestPath[(i, k)] == -1 or shortestPath[(k, j)] == -1: continue
                shortestPath[(i, j)] = min(shortestPath[(i, j)], shortestPath[(i, k)] + shortestPath[(k, j)])
    return shortestPath

"""
dijkstra algorithm with O(VElogV) time complexity which in worse case is O(V^3 * log(V)) for a highly dense graph
So dijkstra is faster than floyd warshall for sparse graphs
- non-negative edge weights
"""