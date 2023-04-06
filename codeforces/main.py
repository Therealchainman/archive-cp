import os,sys
from io import BytesIO, IOBase
from typing import *
# only use pypyjit when needed, it usese more memory, but speeds up recursion in pypy
# sys.setrecursionlimit(1_000_000)
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

from collections import deque

def main():
    input()
    n, m = map(int, input().split())
    adj_list = [[] for _ in range(n + 1)]
    degrees = [0] * (n + 1)
    for _ in range(m):
        u, v = map(int, input().split())
        adj_list[u].append(v)
        degrees[v] += 1
        adj_list[v].append(u)
        degrees[u] += 1
    k = m - n
    # count check
    if k*k + k != m or k*k != n:
        return "No"
    # degree check
    if any(deg not in (2, 4) for deg in degrees[1:]):
        return "No"
    # check all are part of a single connected component
    def bfs1(node):
        cnt = 0
        queue = deque([node])
        vis = [0]*(n + 1)
        vis[node] = 1
        while queue:
            node = queue.popleft()
            cnt += 1
            for nei in adj_list[node]:
                if vis[nei]: continue
                vis[nei] = 1
                queue.append(nei)
        return cnt == n
    if not bfs1(1):
        return "No"
    # connectivity check
    visited = [0]*(n + 1)
    def bfs2(node):
        size = 0
        queue = deque([node])
        visited[node] = 1
        while queue:
            node = queue.popleft()
            size += 1
            for nei in adj_list[node]:
                if visited[nei] or degrees[node] == degrees[nei] == 4: continue
                visited[nei] = 1
                queue.append(nei)
        return size == k
    num_components = 0
    for i in range(1, n + 1):
        if visited[i]: continue
        num_components += 1
        if not bfs2(i):
            return "No"
    return "Yes" if num_components == k else "No"
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        print(main())