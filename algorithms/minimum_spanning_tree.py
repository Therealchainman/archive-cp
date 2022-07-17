"""
Example of minimum spanning tree using Kruskal's algorithm 
"""
class UnionFind:
    def __init__(self,n):
        self.size = [1]*n
        self.parent = list(range(n))
    
    def find(self,i):
        if i==self.parent[i]:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self,i,j):
        i, j = self.find(i), self.find(j)
        if i!=j:
            if self.size[i] < self.size[j]:
                i,j=j,i
            self.parent[j] = i
            self.size[i] += self.size[j]
            return True
        return False
class Solution:
    def minCostToSupplyWater(self, n: int, wells: List[int], pipes: List[List[int]]) -> int:
        for i, weight in enumerate(wells, start=1):
            pipes.append((0,i,weight))
        pipes.sort(key=lambda edge: edge[2])
        dsu = UnionFind(n+1)
        return sum(w for u,v,w in pipes if dsu.union(u,v))