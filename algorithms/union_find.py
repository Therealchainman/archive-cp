"""
Union Find algorithm with path compression and size for optimization
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
    def single_connected_component(self):
        return self.size[self.find(0)] == len(self.parent)
    def is_same_connected_components(self, i, j):
        return self.find(i) == self.find(j)
    def __repr__(self):
        return f'parents: {[(i, parent) for i, parent in enumerate(self.parent)]}, sizes: {self.size}'