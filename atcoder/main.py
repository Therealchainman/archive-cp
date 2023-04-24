import os,sys
from io import BytesIO, IOBase
sys.setrecursionlimit(10**6)
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

from collections import deque
import heapq
import math

def main():
    n, m = map(int, input().split())
    adj_list = [[] for _ in range(n)]
    for _ in range(m):
        u, v = map(int, input().split())
        adj_list[u-1].append(v-1)
        adj_list[v-1].append(u-1)
    k = int(input())
    max_heap = []
    max_dist = [0]*n
    min_dist = [math.inf]*n
    for _ in range(k):
        p, d = map(int, input().split())
        p -= 1
        min_dist[p] = d
        if d > 0:
            heapq.heappush(max_heap, (-d, p))
            max_dist[p] = d
    result = [1]*n
    while max_heap:
        dist, node = heapq.heappop(max_heap)
        result[node] = 0
        dist = -dist
        if dist < max_dist[node]: continue
        for nei in adj_list[node]:
            if dist - 1 <= max_dist[nei]: continue
            max_dist[nei] = dist - 1
            heapq.heappush(max_heap, (-(dist-1), nei))
    def bfs():
        vis = [0]*n
        queue = deque()
        for i in range(n):
            if result[i] == 1:
                queue.append(i)
                vis[i] = 1
        dist = 0
        while queue:
            for _ in range(len(queue)):
                node = queue.popleft()
                if dist > min_dist[node]: return False
                for nei in adj_list[node]:
                    if vis[nei]: continue
                    vis[nei] = 1
                    queue.append(nei)
            dist += 1
        return True
    if sum(result) == 0 or not bfs():
        print('No')
        return
    print('Yes')
    print(''.join(map(str, result)))

if __name__ == '__main__':
    main()
