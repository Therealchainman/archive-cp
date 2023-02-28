import os,sys
from io import BytesIO, IOBase
from typing import *

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

sys.setrecursionlimit(1_000_000)

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
    
class EulerTour:
    def __init__(self, num_nodes: int, edges: List[List[int]]):
        self.num_nodes = num_nodes
        self.edges = edges
        self.adj_list = [[] for _ in range(num_nodes + 1)]
        self.root_node = 1 # root of the tree
        self.enter_counter, self.exit_counter = [0]*(num_nodes + 1), [0]*(num_nodes + 1)
        self.counter = 1
        self.build_adj_list() # adjacency list representation of the tree
        self.euler_tour(self.root_node, -1)
    
    def build_adj_list(self) -> None:
        for u, v in self.edges:
            self.adj_list[u].append(v)
            self.adj_list[v].append(u)

    def euler_tour(self, node: int, parent_node: int):
        self.enter_counter[node] = self.counter
        self.counter += 1
        for child_node in self.adj_list[node]:
            if child_node != parent_node:
                self.euler_tour(child_node, node)
        self.exit_counter[node] = self.counter - 1

def main():
    n, q = map(int, input().split())
    arr = [0] + list(map(int, input().split()))
    edges = []
    for _ in range(n - 1):
        u, v = map(int, input().split())
        edges.append((u, v))
    euler_tour = EulerTour(n, edges)
    fenwick_tree = FenwickTree(n + 1)
    for node, enter_counter in enumerate(euler_tour.enter_counter[1:], start = 1):
        fenwick_tree.update(enter_counter, arr[node])
    result = []
    for _ in range(q):
        query = list(map(int, input().split()))
        if query[0] == 1:
            u, x = query[1:]
            node_index_in_flatten_tree = euler_tour.enter_counter[u]
            delta = x - arr[u]
            arr[u] = x
            fenwick_tree.update(node_index_in_flatten_tree, delta) # update the fenwick tree
        else:
            s = query[1]
            subtree_sum = fenwick_tree.query(euler_tour.exit_counter[s]) - fenwick_tree.query(euler_tour.enter_counter[s] - 1)
            result.append(subtree_sum)
    return '\n'.join(map(str, result))

if __name__ == '__main__':
    print(main())
