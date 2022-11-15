"""
Union Find algorithm with path compression and size for optimization
"""
class UnionFind:
    def __init__(self,n: int):
        self.size = [1]*n
        self.parent = list(range(n))
    
    def find(self,i: int) -> int:
        if i==self.parent[i]:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self,i: int,j: int) -> bool:
        i, j = self.find(i), self.find(j)
        if i!=j:
            if self.size[i] < self.size[j]:
                i,j=j,i
            self.parent[j] = i
            self.size[i] += self.size[j]
            return True
        return False
    def single_connected_component(self) -> bool:
        return self.size[self.find(0)] == len(self.parent)
    def is_same_connected_components(self, i: int, j: int) -> bool:
        return self.find(i) == self.find(j)
    def num_connected_components(self) -> int:
        return len(set(map(self.find, self.parent)))
    def __repr__(self) -> str:
        return f'parents: {[(i, parent) for i, parent in enumerate(self.parent)]}, sizes: {self.size}'

"""
This is a space optimized union find algorithm, where don't have to initialize with the size of the elements, but 
add elements to dictionary and set when they are processed.
"""
class CompactUnionFind:
    def __init__(self):
        self.size = dict()
        self.parent = dict()
        self.connected_components = set()
        self.count_connected_components = 0
        
    def add(self, i: int) -> None:
        if i not in self.connected_components:
            self.connected_components.add(i)
            self.parent[i] = i
            self.size[i] = 1
        self.count_connected_components += 1
    
    def find(self,i: int) -> int:
        if i==self.parent[i]:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self,i: int,j: int) -> bool:
        # FIND THE REPRESENTATIVE NODE FOR THESE NODES IF THEY ARE ALREADY BELONGING TO CONNECTED COMPONENTS
        i, j = self.find(i), self.find(j)
        if i!=j:
            if self.size[i] < self.size[j]:
                i,j=j,i
            self.parent[j] = i
            self.size[i] += self.size[j]
            self.count_connected_components -= 1
            return True
        return False
    
    def __repr__(self) -> str:
        return f'parents: {self.parent}, sizes: {self.size}, connected_components: {self.connected_components}'

"""
This union find implementation is using iterative approach to find the representative node for a given node.
It also uses dictionary for size and parent, so it is space optimized and compact
"""
class UnionFind:
    def __init__(self):
        self.size = dict()
        self.parent = dict()
    
    def find(self,i: int) -> int:
        if i not in self.parent:
            self.size[i] = 1
            self.parent[i] = i
        while i != self.parent[i]:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self,i: int,j: int) -> bool:
        i, j = self.find(i), self.find(j)
        if i!=j:
            if self.size[i] < self.size[j]:
                i,j=j,i
            self.parent[j] = i
            self.size[i] += self.size[j]
            return True
        return False
    
    @property
    def root_count(self):
        return sum(node == self.find(node) for node in self.parent)

    def __repr__(self) -> str:
        return f'parents: {[(i, parent) for i, parent in enumerate(self.parent)]}, sizes: {self.size}'