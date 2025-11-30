
"""
Binary Lifting algorithm to the find the kth ancestor in a tree
"""
import math
from typing import List
from collections import defaultdict
class TreeAncestor:

    def __init__(self, n: int, parent: List[int]):
        C = 1+int(math.log2(n))
        self.table = [[-1 for _ in range(C)] for _ in range(n)]
        for c in range(C):
            for r in range(n):
                if c==0: self.table[r][c] = parent[r]
                elif self.table[r][c-1] != -1:
                    self.table[r][c] = self.table[self.table[r][c-1]][c-1]
    
    def getKthAncestor(self, node: int, k: int) -> int:
        while node != -1 and k>0:
            i = int(math.log2(k))
            node = self.table[node][i]
            k-=(1<<i)
        return node

"""
A better iteration on binary lifting algorithm that can compute lca and kth ancestor of tree in
logn time complexity
"""
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class BinaryLift:

    def __init__(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode'):
        self.root = root
        self.p = p
        self.q = q
        self.nodesIndex = defaultdict(int) # node to index
        self.nodes = [] # index to node
        self.parents = defaultdict(lambda: -1) # index of node to parent index
        self.levels = defaultdict(int) # index of node to level
        self.i = -1
        self.dfs(root)
        self.build_table(self.i+1)
        
    def build_table(self, n: int):
        self.C = 1+int(math.log2(n))
        self.table = [[-1 for _ in range(self.C)] for _ in range(n)]
        for c in range(self.C):
            for r in range(n):
                if c==0: self.table[r][c] = self.parents[r]
                elif self.table[r][c-1] != -1:
                    self.table[r][c] = self.table[self.table[r][c-1]][c-1]
    
    """
    Finding the parents and level nodes
    """
    def dfs(self, node, level=0, parent=-1):
        if not node: return
        self.i += 1
        i = self.i
        self.nodesIndex[node] = self.i
        self.nodes.append(node)
        self.levels[self.i] = level
        self.parents[self.i] = parent
        self.dfs(node.left, level+1, i)
        self.dfs(node.right, level+1, i)
        
        
    def getKthAncestor(self, node: int, k: int) -> int:
        while node != -1 and k>0:
            i = int(math.log2(k))
            node = self.table[node][i]
            k-=(1<<i)
        return node
    
    def find_lca(self):
        # p is at deeper level
        p, q = self.nodesIndex[self.p], self.nodesIndex[self.q]
        if self.levels[p]<self.levels[q]:
            p, q = q, p
        # put on same level by finding the kth ancestor
        k = self.levels[p] - self.levels[q]
        p = self.getKthAncestor(p, k)
        if p == q:
            return self.nodes[p]
        for j in range(self.C)[::-1]:
            if self.table[p][j] != self.table[q][j]:
                p = self.table[p][j]
                q = self.table[q][j]
        return self.nodes[self.table[p][0]]

"""
Binary Lift that was used on a forest datastructure (multiple trees).  It also computes the maximum edge weight till reaches the lca.
That is assumed to be the maximum edge weight in the path.  

requires 0-indexed nodes, so you may need to perform a coordinate compression
"""

class maxBinaryLift:
    """
    This binary lift function works on any undirected graph that is composed of
    an adjacency list defined by graph
    """
    def __init__(self, node_count: int, graph: List[List[int]]):
        self.size = node_count
        self.graph = graph # pass in an adjacency list to represent the graph
        self.depth = [0]*node_count
        self.parents = [-1]*node_count
        self.parentsWeight = [0]*node_count
        self.visited = [False]*node_count
        # ITERATE THROUGH EACH POSSIBLE TREE
        for node in range(node_count):
            if self.visited[node]: continue
            self.visited[node] = True
            self.get_parent_depth(node)
        self.maxAncestor = 18 # set it so that only up to 2^18th ancestor can exist for this example
        self.jump = [[-1]*self.maxAncestor for _ in range(self.size)]
        self.maxJumpWeight = [[0]*self.maxAncestor for _ in range(self.size)]
        self.build_sparse_table()
        
    def build_sparse_table(self) -> None:
        """
        builds the jump and maxWeightJump sparse arrays for computing the 2^jth ancestor of ith node in any given query
        """
        for j in range(self.maxAncestor):
            for i in range(self.size):
                if j == 0:
                    self.jump[i][j] = self.parents[i]
                    self.maxJumpWeight[i][j] = self.parentsWeight[i]
                elif self.jump[i][j-1] != -1:
                    prev_ancestor = self.jump[i][j-1]
                    self.jump[i][j] = self.jump[prev_ancestor][j-1]
                    current_jump_weight = self.maxJumpWeight[i][j-1]
                    prev_max_weight = self.maxJumpWeight[prev_ancestor][j-1]
                    if prev_max_weight == 0: continue 
                    self.maxJumpWeight[i][j] = max(current_jump_weight, prev_max_weight)
                    
    def get_parent_depth(self, node: int, parent_node: int = -1, weight: int = 0, depth: int = 0) -> None:
        """
        Fills out the depth array for each node and the parent array for each node
        """
        self.parents[node] = parent_node
        self.parentsWeight[node] = weight
        self.depth[node] = depth
        for nei_node, wei in self.graph[node]:
            if self.visited[nei_node]: continue
            self.visited[nei_node] = True
            self.get_parent_depth(nei_node, node, wei, depth+1)
            
    def max_weight_lca(self, p: int, q: int) -> int:
        self.maxWeight = 0
        # ASSUME NODE P IS DEEPER THAN NODE Q   
        if self.depth[p] < self.depth[q]:
            p, q = q, p
        k = self.depth[p] - self.depth[q]
        p = self.kthAncestor(p, k)
        if p == q: return self.maxWeight
        for j in range(self.maxAncestor)[::-1]:
            if self.jump[p][j] != self.jump[q][j]:
                self.maxWeight = max(self.maxWeight, self.maxJumpWeight[p][j], self.maxJumpWeight[q][j])
                p, q = self.jump[p][j], self.jump[q][j] # jump to 2^jth ancestor nodes
        self.maxWeight = max(self.maxWeight, self.maxJumpWeight[p][0], self.maxJumpWeight[q][0])
        return self.maxWeight
    
    def kthAncestor(self, node: int, k: int) -> int:
        while node != -1 and k > 0:
            j = int(math.log2(k))
            self.maxWeight = max(self.maxWeight, self.maxJumpWeight[node][j])
            node = self.jump[node][j]
            k -= (1<<j)
        return node