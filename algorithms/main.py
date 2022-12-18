from typing import List
import math
import heapq
from itertools import product

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

def dijkstra(n: int, src: int, adj_list: List[List[int]]) -> List[int]:
    dist = [math.inf]*n
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: continue
        for v, w in adj_list[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    return dist

def johnsons(n: int, edges: List[List[int]]) -> List[List[int]]:
    # create a new vertex that is connected to all other vertices with weight 0
    # new vertex that will be the source for bellman fourd is going to be n
    # run bellman ford to find shortest paths from the new vertex to all other vertices
    dist = bellmanFord(n+1, n, edges + [[n, i, 0] for i in range(n)])
    if not dist: return [] # if it has negative cycle
    # reweight the edges
    for i in range(len(edges)):
        u, v, w = edges[i]
        edges[i][2] = w + dist[u] - dist[v]
    # run dijkstra for each vertex
    adj_list = [[] for _ in range(n)]
    for u, v, w in edges:
        adj_list[u].append((v, w))
    shortest_paths = [dijkstra(n, i, adj_list) for i in range(n)]
    # undo the reweighting
    for u, v in product(range(n), repeat = 2):
        if shortest_paths == math.inf: continue
        shortest_paths[u][v] = shortest_paths[u][v] + dist[v] - dist[u]
    return shortest_paths