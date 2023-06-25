import os,sys
from io import BytesIO, IOBase
sys.setrecursionlimit(10**6)
from typing import *
# only use pypyjit when needed, it usese more memory, but speeds up recursion in pypy
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')
# sys.stdout = open('output.txt', 'w')

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

from heapq import heappush, heappop

def main():
    n, m = map(int, input().split())
    adj_list = [[] for _ in range(n + 1)]
    for _ in range(m):
        u, v, w = map(int, input().split())
        adj_list[u].append((v, w))
        adj_list[v].append((u, w))
    K = int(input())
    start_nodes = map(int, input().split())
    D = int(input())
    dist = [0] + list(map(int, input().split()))
    res = [-1] * (n + 1)
    min_heap = []
    for node in start_nodes:
        res[node] = 0
        for nei, wei in adj_list[node]:
            heappush(min_heap, (wei, nei))
    def dfs(node, rem_dist):
        neighbors = []
        rem_heap = [(rem_dist, node)]
        while rem_heap:
            rem_dist, node = heappop(rem_heap)
            for nei, wei in adj_list[node]:
                if res[nei] != -1: continue
                if wei <= rem_dist:
                    res[nei] = day
                    heappush(rem_heap, (rem_dist - wei, nei))
                else:
                    neighbors.append((wei, nei))
        return neighbors
    for day in range(1, D + 1):
        tomorrow = []
        while min_heap and min_heap[0][0] <= dist[day]:
            wei, node = heappop(min_heap)
            if res[node] != -1: continue
            res[node] = day
            tomorrow.extend(dfs(node, dist[day] - wei))
        for wei, node in tomorrow:
            heappush(min_heap, (wei, node))
    print('\n'.join(map(str, res[1:])))

if __name__ == '__main__':
    main()