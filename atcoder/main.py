import os,sys
from io import BytesIO, IOBase
sys.setrecursionlimit(10**6)
from typing import *
import pypyjit
pypyjit.set_param('max_unroll_recursion=-1')


# Fast IO Region
BUFSIZE = 8192
class FastIO(IOBase):
    newlines = 0
    def __init__(self, file):
        self._fd = file.fileno()
        self.buffer = BytesIO()
        self.writable = "x" in file.mode or "r" not in file.mode
        self.write = self.buffer.write if self.writable else None
    def read(self):
        while True:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            if not b:
                break
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines = 0
        return self.buffer.read()
    def readline(self):
        while self.newlines == 0:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            self.newlines = b.count(b"\n") + (not b)
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines -= 1
        return self.buffer.readline()
    def flush(self):
        if self.writable:
            os.write(self._fd, self.buffer.getvalue())
            self.buffer.truncate(0), self.buffer.seek(0)
class IOWrapper(IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        self.write = lambda s: self.buffer.write(s.encode("ascii"))
        self.read = lambda: self.buffer.read().decode("ascii")
        self.readline = lambda: self.buffer.readline().decode("ascii")
sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)
input = lambda: sys.stdin.readline().rstrip("\r\n")

import math

"""
This is an euler tour for weights on edges on the tree

it starts the counter at 1, so it is 1-indexed flattened tree so it works well with fenwick tree.
Fenwick tree requires 1-indexed arrays. 
"""
class EulerTourPathQueries:
    def __init__(self, num_nodes, adj_list):
        num_edges = num_nodes - 1
        self.edge_to_child_node = [0]*num_edges
        self.num_edges = num_edges
        self.adj_list = adj_list
        self.root_node = 0 # root of the tree
        self.enter_counter, self.exit_counter = [0]*num_nodes, [0]*num_nodes
        self.counter = 1
        self.euler_tour(self.root_node, -1)

    def euler_tour(self, node: int, parent_node: int):
        self.enter_counter[node] = self.counter
        self.counter += 1
        for child_node, _, edge_index in self.adj_list[node]:
            if child_node != parent_node:
                self.edge_to_child_node[edge_index] = child_node
                self.euler_tour(child_node, node)
        self.counter += 1
        self.exit_counter[node] = self.counter

    def __repr__(self):
        return f"enter_counter: {self.enter_counter}, exit_counter: {self.exit_counter}"

class FenwickTree:
    def __init__(self, N):
        self.sums = [0 for _ in range(N+1)]

    def update(self, i, delta):
        while i < len(self.sums):
            self.sums[i] += delta
            i += i & (-i)

    def query(self, i):
        res = 0
        while i > 0:
            res += self.sums[i]
            i -= i & (-i)
        return res

    def __repr__(self):
        return f"array: {self.sums}"
    

class BinaryLift:
    """
    This binary lift function works on any undirected graph that is composed of
    an adjacency list defined by graph
    """
    def __init__(self, node_count: int, graph: List[List[int]]):
        self.size = node_count
        self.graph = graph # pass in an adjacency list to represent the graph
        self.depth = [0]*node_count
        self.parents = [-1]*node_count
        self.visited = [False]*node_count
        # ITERATE THROUGH EACH POSSIBLE TREE
        for node in range(node_count):
            if self.visited[node]: continue
            self.visited[node] = True
            self.get_parent_depth(node)
        self.maxAncestor = 18 # set it so that only up to 2^18th ancestor can exist for this example
        self.jump = [[-1]*self.maxAncestor for _ in range(self.size)]
        self.build_sparse_table()
        
    def build_sparse_table(self) -> None:
        """
        builds the jump sparse arrays for computing the 2^jth ancestor of ith node in any given query
        """
        for j in range(self.maxAncestor):
            for i in range(self.size):
                if j == 0:
                    self.jump[i][j] = self.parents[i]
                elif self.jump[i][j-1] != -1:
                    prev_ancestor = self.jump[i][j-1]
                    self.jump[i][j] = self.jump[prev_ancestor][j-1]
                    
    def get_parent_depth(self, node: int, parent_node: int = -1, depth: int = 0) -> None:
        """
        Fills out the depth array for each node and the parent array for each node
        """
        self.parents[node] = parent_node
        self.depth[node] = depth
        for nei_node, _, _ in self.graph[node]:
            if self.visited[nei_node]: continue
            self.visited[nei_node] = True
            self.get_parent_depth(nei_node, node, depth+1)

    def distance(self, p: int, q: int) -> int:
        """
        Computes the distance between two nodes
        """
        lca = self.find_lca(p, q)
        return self.depth[p] + self.depth[q] - 2*self.depth[lca]

    def find_lca(self, p: int, q: int) -> int:
        # ASSUME NODE P IS DEEPER THAN NODE Q   
        if self.depth[p] < self.depth[q]:
            p, q = q, p
        # PUT ON SAME DEPTH BY FINDING THE KTH ANCESTOR
        k = self.depth[p] - self.depth[q]
        p = self.kthAncestor(p, k)
        if p == q: return p
        for j in range(self.maxAncestor)[::-1]:
            if self.jump[p][j] != self.jump[q][j]:
                p, q = self.jump[p][j], self.jump[q][j] # jump to 2^jth ancestor nodes
        return self.jump[p][0]
    
    def kthAncestor(self, node: int, k: int) -> int:
        while node != -1 and k>0:
            i = int(math.log2(k))
            node = self.jump[node][i]
            k-=(1<<i)
        return node

def main():
    n = int(input())
    adj_list = [[] for _ in range(n)]
    values = [0]*n
    for i in range(n - 1):
        u, v, w = map(int, input().split())
        u -= 1
        v -= 1
        adj_list[u].append((v, w, i))
        adj_list[v].append((u, w, i))
    euler_tour = EulerTourPathQueries(n, adj_list)
    stack = [0]
    vis = [0]*n
    vis[0] = 1
    fenwick = FenwickTree(2*n + 2)
    while stack:
        node = stack.pop()
        for nei, wei, _ in adj_list[node]:
            if vis[nei]: continue
            enter_counter, exit_counter = euler_tour.enter_counter[nei], euler_tour.exit_counter[nei]
            values[nei] = wei
            fenwick.update(enter_counter, wei)
            fenwick.update(exit_counter, -wei)
            vis[nei] = 1
            stack.append(nei)
    binary_lifting = BinaryLift(n, adj_list)
    q = int(input())
    for _ in range(q):
        t, u, v = map(int, input().split())
        if t == 1:
            node = euler_tour.edge_to_child_node[u - 1]
            enter_counter, exit_counter = euler_tour.enter_counter[node], euler_tour.exit_counter[node]
            delta = v - values[node]
            values[node] = v
            fenwick.update(enter_counter, delta)
            fenwick.update(exit_counter, -delta)
        else:
            u -= 1
            v -= 1
            lca_uv = binary_lifting.find_lca(u, v)
            dist_u, dist_v, dist_lca = map(fenwick.query, (euler_tour.enter_counter[u], euler_tour.enter_counter[v], euler_tour.enter_counter[lca_uv]))
            print(dist_u + dist_v - 2*dist_lca)

if __name__ == '__main__':
    main()