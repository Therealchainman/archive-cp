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

# sys.setrecursionlimit(1_000_000)
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')

def is_eulerian_path(n, adj_list, indegrees, outdegrees):
    # start node is 1 in this instance
    start_node = 1
    end_node = n
    stack = [start_node]
    vis = [0] * (n + 1)
    vis[start_node] = 1
    while stack:
        node = stack.pop()
        for nei in adj_list[node]:
            if vis[nei]: continue
            vis[nei] = 1
            stack.append(nei)
    if outdegrees[start_node] - indegrees[start_node] != 1 or indegrees[end_node] - outdegrees[end_node] != 1: return False
    for i in range(1, n + 1):
        if ((outdegrees[i] > 0 or indegrees[i] > 0) and not vis[i]): return False
        if (indegrees[i] != outdegrees[i] and i not in (start_node, end_node)): return False
    return True

def hierholzers_directed(n, adj_list):
    start_node = 1
    end_node = n
    stack = [start_node]
    euler_path = []
    while stack:
        node = stack[-1]
        if len(adj_list[node]) == 0:
            euler_path.append(stack.pop())
        else:
            nei = adj_list[node].pop()
            stack.append(nei)
    return euler_path[::-1]

def main():
    n, m = map(int, input().split())
    adj_list = [set() for _ in range(n + 1)]
    indegrees, outdegrees = [0] * (n + 1), [0] * (n + 1)
    for _ in range(m):
        u, v = map(int, input().split())
        adj_list[u].add(v)
        indegrees[v] += 1
        outdegrees[u] += 1
    # all degrees are even and one connected component with edge (nonzero degrees)
    if not is_eulerian_path(n, adj_list, indegrees, outdegrees):
        return "IMPOSSIBLE"
    # hierholzer's algorithm to reconstruct the eulerian circuit
    eulerian_path = hierholzers_directed(n, adj_list)
    return ' '.join(map(str, eulerian_path))

if __name__ == '__main__':
    print(main())
    # T = int(input())
    # for _ in range(T):
    #     print(main())
