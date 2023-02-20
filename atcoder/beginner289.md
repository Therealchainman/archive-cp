# Atcoder Beginner Contest 289

## What is used at the top of each submission

```py
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
```

## A - flip

### Solution 1: str + xor + bit operations

```py
def main():
    s = input()
    return ''.join(map(str, [x^1 for x in map(int, s)]))
 
if __name__ == '__main__':
    print(main())
```

## B - V

### Solution 1:  dfs for connected components + undirected graph

```py
def main():
    n, m = map(int, input().split())
    arr = list(map(int, input().split()))
    adj_list = [[] for _ in range(n + 1)]
    visited = [0]*(n + 1)
    def bfs(node: int, adj_list: List[int]) -> List[int]:
        stack = [node]
        visited[node] = 1
        cc = []
        while stack:
            cur_node = stack.pop()
            cc.append(cur_node)
            for neighbor in adj_list[cur_node]:
                if visited[neighbor] == 0:
                    visited[neighbor] = 1
                    stack.append(neighbor)
        return cc
    for u in arr:
        v = u + 1
        adj_list[u].append(v)
        adj_list[v].append(u)
    result = []
    for node in range(1, n + 1):
        if visited[node]: continue
        connected_component = bfs(node, adj_list)
        result.extend(sorted(connected_component, reverse = True))
    return ' '.join(map(str, result))
 
if __name__ == '__main__':
    print(main())
```

## C - Coverage

### Solution 1:  bitmask + set operations + union of sets

```py
def main():
    n, m = map(int, input().split())
    s = [None]*m
    for i in range(m):
        _ = int(input())
        s[i] = set(map(int, input().split()))
    res = 0
    for mask in range(1, 1<<m):
        cur_set = set()
        for i in range(m):
            if (mask>>i)&1:
                cur_set |= s[i]
        res += len(cur_set) == n
    return res
 
if __name__ == '__main__':
    print(main())
```

## D - Step Up Robot

### Solution 1:  stack + iterative dfs + visited array

```py
def main():
    n = int(input())
    moves = list(map(int, input().split()))
    m = int(input())
    traps = set(map(int, input().split()))
    x = int(input())
    stack = [0]
    visited = [0]*(x + 1)
    visited[0] = 1
    while stack:
        node = stack.pop()
        for move in moves:
            new_node = node + move
            if new_node == x: return 'Yes'
            if new_node < x and not visited[new_node] and new_node not in traps:
                visited[new_node] = 1
                stack.append(new_node)
    return 'No'
 
if __name__ == '__main__':
    print(main())
```

## E - Swap Places

### Solution 1:  memoization for states + bfs + deque + shortest path

```py
import math
from collections import deque
 
def main():
    n, m = map(int, input().split())
    colors = [0] + list(map(int, input().split()))
    adj_list = [[] for _ in range(n + 1)]
    for _ in range(m):
        u, v = map(int, input().split())
        adj_list[u].append(v)
        adj_list[v].append(u)
    dist = [[math.inf]*(n + 1) for _ in range(n + 1)]
    dist[1][n] = 0
    queue = deque([(1, n, 0)])
    while queue:
        u, v, d = queue.popleft()
        if (u, v) == (n, 1): return d
        for nei_u in adj_list[u]:
            for nei_v in adj_list[v]:
                if colors[nei_u] == colors[nei_v]: continue
                if d + 1 < dist[nei_u][nei_v]:
                    dist[nei_u][nei_v] = d + 1
                    queue.append((nei_u, nei_v, d + 1))
    return -1
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        print(main())
```

## F - Teleporter Takahashi

### Solution 1:  a trivial way to change the current x, y based on two moves in the rectangle leads to solution + the cases that will guarantee that it cannot reach the answer + x, y needs to always have the same parity as tx, ty + you need x_min != x_max if you are going to be able to change that x to a target value.  + all moves will always be less than 1 million, think about it worst case is probably 400,000 moves to get to a target point.

```py
def main():
    sx, sy = map(int, input().split())
    tx, ty = map(int, input().split())
    x_min, x_max, y_min, y_max = map(int, input().split())
    # (rx, ry) is point in the the rectangle
    def move(rx: int, ry: int) -> Tuple[int, int]:
        path.append(f'{rx} {ry}')
        return 2*rx - x, 2*ry - y
    for i in range(2):
        x, y = sx, sy
        path = ['Yes']
        if i:
            x, y = move(x_min, y_min)
        if x%2 != tx%2 or y%2 != ty%2 or (x_min == x_max and x != tx) or (y_min == y_max and y != ty):
            continue
        while (x < tx): 
            x, y = move(x_min, y_min)
            x, y = move(x_min + 1, y_min)
        while (x > tx):
            x, y = move(x_min + 1, y_min)
            x, y = move(x_min, y_min)
        while (y < ty):
            x, y = move(x_min, y_min)
            x, y = move(x_min, y_min + 1)
        while (y > ty):
            x, y = move(x_min, y_min + 1)
            x, y = move(x_min, y_min)
        return '\n'.join(path)
    return 'No'

if __name__ == '__main__':
    print(main())
```