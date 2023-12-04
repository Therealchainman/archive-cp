# Atcoder Beginner Contest 311

## What is used at the top of each submission

```py
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
                    
if __name__ == '__main__':
    print(main())
    # main()
    # sys.stdout.close()
```

## A - First ABC 

### Solution 1:  set

```py
def main():
    n = int(input())
    s = input()
    seen = set()
    for i, ch in enumerate(s):
        seen.add(ch)
        if len(seen) == 3: return print(i + 1)
    print(n)

if __name__ == '__main__':
    main()
```

## B - Vacation Together 

### Solution 1:  sliding window + any

```py
def main():
    n, d = map(int, input().split())
    arr = [list(input()) for _ in range(n)]
    res = window = 0
    for i in range(d):
        if any(person[i] == 'x' for person in arr):
            window = 0
        else:
            window += 1
        res = max(res, window)
    print(res)

if __name__ == '__main__':
    main()
```

## C - Find it! 

### Solution 1:  dfs + detect cycle in directed graph + backtrack to recreate directed cycle

```py
def main():
    n = int(input())
    adj_list = [[] for _ in range(n + 1)]
    parent = [0] + list(map(int, input().split()))
    for i in range(1, n + 1):
        adj_list[i].append(parent[i])
    visited = [0] * (n + 1)
    in_path = [0] * (n + 1)
    path = []
    def detect_cycle(node) -> bool:
        path.append(node)
        visited[node] = 1
        in_path[node] = 1
        for nei in adj_list[node]:
            if in_path[nei]: return nei
            if visited[nei]: continue
            res = detect_cycle(nei)
            if res: return res
        in_path[node] = 0
        path.pop()
        return 0
    for i in range(1, n + 1):
        if visited[i]: continue
        node = detect_cycle(i)
        cur = node
        cycle = [node]
        if cur:
            while path[-1] != node:
                cur = path.pop()
                cycle.append(cur)
            print(len(cycle))
            print(*cycle[::-1])
            return
    print(-1)

if __name__ == '__main__':
    main()
```

## D - Grid Ice Floor 

### Solution 1:  bfs + modified neighborhood

```py
from collections import deque

def main():
    R, C = map(int, input().split())
    grid = [list(input()) for _ in range(R)]
    vis = set()
    vis2 = set([(1, 1)])
    queue = deque([(1, 1)])
    def neighborhood(r, c):
        for dr, dc in ((-1, 0), (0, -1), (1, 0), (0, 1)):
            nr, nc = r, c
            while grid[nr][nc] == '.':
                vis.add((nr, nc))
                nr += dr
                nc += dc
            yield nr - dr, nc - dc
    while queue:
        r, c = queue.popleft()
        for nr, nc in neighborhood(r, c):
            if (nr, nc) in vis2: continue
            vis2.add((nr, nc))
            queue.append((nr, nc))
    print(len(vis))

if __name__ == '__main__':
    main()
```

## E - Defect-free Squares 

### Solution 1:  dynamic programming + size of squares

```py
from itertools import product

def main():
    R, C, N = map(int, input().split())
    dp = [[1] * C for _ in range(R)]
    for _ in range(N):
        r, c = map(int, input().split())
        dp[r - 1][c - 1] = 0
    for r, c in product(range(R), range(C)):
        if r == 0 or c == 0: continue
        if dp[r][c] == 0: continue
        dp[r][c] = min(dp[r - 1][c], dp[r][c - 1], dp[r - 1][c - 1]) + 1
    res = sum(sum(row) for row in dp)
    print(res)

if __name__ == '__main__':
    main()
```

## F - Yet Another Grid Task 

### Solution 1: 

```py

```

## G - One More Grid Task 

### Solution 1: 

```py

```