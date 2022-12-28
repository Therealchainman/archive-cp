import os,sys
from io import BytesIO, IOBase
from typing import *
import math


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

class MaxFlow:
    def __init__(self, n: int, edges: List[Tuple[int, int, int]]):
        self.size = n
        self.edges = edges

    def build(self, n: int, edges: List[Tuple[int, int, int]]) -> None:
        self.adj_list = {}
        for u, v, cap in edges:
            if u not in self.adj_list:
                self.adj_list[u] = Counter()
            self.adj_list[u][v] += cap
            if v not in self.adj_list:
                self.adj_list[v] = Counter()

    def main(self, source: int, sink: int) -> int:
        self.build(self.size, self.edges)
        maxflow = 0
        while True:
            self.reset()
            cur_flow = self.dfs(source, sink, math.inf)
            if cur_flow == 0:
                break
            maxflow += cur_flow
        return maxflow

    def reset(self) -> None:
        self.vis = [False] * self.size

    def dfs(self, node: int, sink: int, flow: int) -> int:
        if node == sink:
            return flow
        self.vis[node] = True
        cap = self.adj_list[node]
        for nei, cap in cap.items():
            if not self.vis[nei] and cap > 0:
                cur_flow = self.dfs(nei, sink, min(flow, cap))
                if cur_flow > 0:
                    self.adj_list[node][nei] -= cur_flow
                    self.adj_list[nei][node] += cur_flow
                    return cur_flow
        return 0

def main():
    n, m = map(int, input().split())
    edges = [None] * m
    mx = 0
    for i in range(m):
        u, v, cap = map(int, input().split())
        edges[i] = (u - 1, v - 1, cap)
        mx = max(mx, cap)
    source, sink = 0, n - 1
    maxflow = MaxFlow(n, edges)
    return maxflow.main(source, sink)

if __name__ == '__main__':
    print(main())