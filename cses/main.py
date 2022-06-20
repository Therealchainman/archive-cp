import os,sys
from io import BytesIO, IOBase

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

# Source: https://usaco.guide/general/io

from math import log2
from collections import deque

class LCA:
    def main(self):
        n, q = map(int,input().split())
        parents_gen = map(int, input().split())
        self.LOG = int(log2(n))+1
        self.up = [[-1]*self.LOG for _ in range(n+1)]
        graph = [[] for _ in range(n+1)] # travel from parent to child node
        for i, par in zip(range(2,n+1), parents_gen):
            self.up[i][0] = par
            graph[par].append(i)
        self.depth = [0]*(n+1)
        queue = deque([(1, 0)]) # (node, depth)
        while queue:
            node, dep = queue.popleft()
            self.depth[node] = dep
            for child in graph[node]:
                for j in range(1,self.LOG):
                    if self.up[child][j-1] == -1: break
                    self.up[child][j] = self.up[self.up[child][j-1]][j-1]
                queue.append((child, dep+1))
        result = []
        for _ in range(q):
            u, v = map(int,input().split())
            result.append(self.lca(u,v))
        return '\n'.join(map(str,result))

    def lca(self, u, v):
        # always depth[u] < depth[v], v is deeper node
        if self.depth[u] > self.depth[v]:
            u, v = v, u # swap the nodes
        k = self.depth[v] - self.depth[u]
        while k > 0:
            i = int(log2(k))
            v = self.up[v][i]
            k-=(1<<i)
        if u == v: return u
        for j in range(self.LOG)[::-1]:
            if self.up[u][j]==-1 or self.up[v][j]==-1 or self.up[u][j] == self.up[v][j]: continue
            u = self.up[u][j]
            v = self.up[v][j]
        return self.up[u][0]


if __name__ == '__main__':
    print(LCA().main())