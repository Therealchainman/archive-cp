import os,sys
from io import BytesIO, IOBase
from typing import *
# only use pypyjit when needed, it usese more memory, but speeds up recursion in pypy
sys.setrecursionlimit(1_000_000)
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')
 
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

def main():
    n = int(input())
    adj_list = [[] for _ in range(n + 1)]
    for _ in range(n - 1):
        u, v = map(int, input().split())
        adj_list[u].append(v)
        adj_list[v].append(u)
    ans = [n] * (n + 1)
    freq = [0] * (n + 1)
    max_dist_subtree1 = [0] * (n + 1)
    max_dist_subtree2 = [0] * (n + 1)
    child1 = [0] * (n + 1)
    child2 = [0] * (n + 1)
    # PHASE 1: DFS to find the max distance from each node to the leaf in it's subtree but find first and second max distance
    def dfs1(node, parent):
        for child in adj_list[node]:
            if child == parent: continue
            max_dist_subtree = dfs1(child, node)
            if max_dist_subtree > max_dist_subtree1[node]:
                max_dist_subtree2[node] = max_dist_subtree1[node]
                child2[node] = child1[node]
                max_dist_subtree1[node] = max_dist_subtree
                child1[node] = child
            elif max_dist_subtree > max_dist_subtree2[node]:
                max_dist_subtree2[node] = max_dist_subtree
                child2[node] = child
        return max_dist_subtree1[node] + 1
    dfs1(1, 0)
    parent_max_dist = [-1] * (n + 1)
    # PHASE 2: 
    def dfs2(node, parent):
        parent_max_dist[node] = parent_max_dist[parent] + 1
        if parent != 0:
            parent_max_dist[node] = max(parent_max_dist[node], max_dist_subtree1[parent] + 1) if node != child1[parent] else max(parent_max_dist[node], max_dist_subtree2[parent] + 1)
        for child in adj_list[node]:
            if child == parent: continue
            dfs2(child, node)
    dfs2(1, 0)
    # PHASE 3: compute the frequency for each max distance for each node
    for i in range(1, n + 1):
        freq[max(max_dist_subtree1[i], parent_max_dist[i])] += 1
    suffix_freq = 0
    for i in range(n, 0, -1):
        suffix_freq += freq[i]
        if suffix_freq > 0:
            ans[i] = n - suffix_freq + 1
    print(*ans[1:])

if __name__ == '__main__':
    main()