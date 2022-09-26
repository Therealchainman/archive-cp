"""
Tarjan's algorithm implemented to find the bridges in a connected undirected graph

requires the edges which can be represented by integers from 0 to n-1, 
"""
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