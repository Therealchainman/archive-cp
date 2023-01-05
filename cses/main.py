import os,sys
from io import BytesIO, IOBase
from typing import *
import math
from collections import deque, defaultdict


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

def bellmanFord(n: int, source: int, edges: List[List[int]]) -> List[int]:
    dist = [int(1e18)]*n
    parents = [-1]*n
    dist[source] = 0
    last_node_updated = None
    for _ in range(n):
        last_node_updated = None
        for src, dst, wei in edges:
            if dist[src] + wei < dist[dst]:
                dist[dst] = dist[src] + wei
                parents[dst] = src
                last_node_updated = dst
    if last_node_updated is None: 
        print('NO')
        return
    cycle = []
    for _ in range(n):
        last_node_updated = parents[last_node_updated]
    node = last_node_updated
    while True:
        cycle.append(node + 1)
        if node == last_node_updated and len(cycle) > 1: break
        node = parents[node]
    print('YES')
    print(*reversed(cycle))

def main():
    n, m = map(int, input().split())
    edges = [None] * m
    for i in range(m):
        u, v, w = map(int, input().split())
        edges[i] = (u - 1, v - 1, w)
    bellmanFord(n, 0, edges)
    
if __name__ == '__main__':
    main()