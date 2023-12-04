# Atcoder Beginner Contest 291

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

## A - camel Case 

### Solution 1:  loop

```py
def main():
    s = input()
    for i, ch in enumerate(s, start = 1):
        if ch == ch.upper():
            return i
    return -1
 
if __name__ == '__main__':
    print(main())
```

## B - Trimmed Mean 

### Solution 1:  sort + sum

```py
def main():
    n = int(input())
    arr = sorted(list(map(int, input().split())))
    mean = sum(arr[n:4*n]) / (3*n)
    return mean
 
if __name__ == '__main__':
    print(main())
```

## C - LRUD Instructions 2 

### Solution 1:  set + position in plane

```py
def main():
    n = int(input())
    s = input()
    vis = set([(0, 0)])
    x = y = 0
    for ch in s:
        if ch == 'L':
            x -= 1
        elif ch == 'R':
            x += 1
        elif ch == 'U':
            y += 1
        else: # 'D'
            y -= 1
        if (x, y) in vis: return 'Yes'
        vis.add((x, y))
    return 'No'
 
if __name__ == '__main__':
    print(main())
```

## D - Flip Cards 

### Solution 1:  iterative dp + store the number of ways to satisfy condition with the current card face up, front or back card + keep a count for both

```py
def main():
    n = int(input())
    front, back = [0]*n, [0]*n
    for i in range(n):
        front_card, back_card = map(int, input().split())
        front[i], back[i] = front_card, back_card
    mod = 998_244_353
    front_count, back_count = [0]*(n), [0]*n
    front_count[0] = back_count[0] = 1
    for i in range(1, n):
        if front[i] != front[i - 1]:
            front_count[i] += front_count[i - 1]
        if front[i] != back[i - 1]:
            front_count[i] += back_count[i - 1]
        if back[i] != front[i - 1]:
            back_count[i] += front_count[i - 1]
        if back[i] != back[i - 1]:
            back_count[i] += back_count[i - 1]
        front_count[i] %= mod
        back_count[i] %= mod
    return (front_count[-1] + back_count[-1])%mod
 
if __name__ == '__main__':
    print(main())
```

## E - Find Permutation 

### Solution 1:  topological sort + if at any point there are multiple neighbors that have indegree = 0 that would mean it does not have a unique solution + if can't reach the end

```py
from collections import deque
 
def main():
    n, m = map(int, input().split())
    adj_list = [[] for _ in range(n + 1)]
    indegrees = [0]*(n + 1)
    for _ in range(m):
        u, v = map(int, input().split())
        indegrees[v] += 1
        adj_list[u].append(v)
    queue = deque()
    for i in range(1, n + 1):
        if indegrees[i] == 0:
            queue.append(i)
    indices = []
    while queue:
        if len(queue) > 1: return 'No'
        idx = queue.popleft()
        indices.append(idx)
        for nei in adj_list[idx]:
            indegrees[nei] -= 1
            if indegrees[nei] == 0:
                queue.append(nei)
    if len(indices) != n: return 'No'
    res = [0]*(n + 1)
    for i, idx in enumerate(indices, start = 1):
        res[idx] = i
    return f"Yes\n{' '.join(map(str, res[1:]))}"
 
if __name__ == '__main__':
    print(main())
```

## F - Teleporter and Closed off 

### Solution 1:  shortest path with dp + store the shortest distance from 1 to ith node and from n to ith node + do with dp + O(nm) time + for k there are these transitions possible k - m < i < k < j < k + m + if can teleport from i -> j while skipping k then take the distance from 1 and n to get the distance to n by skipping kth node + O(nm^2) time

```py
import math
 
def main():
    n, m = map(int, input().split())
    teleporters = [''] + [input() for _ in range(n)]
    min_teleports = [math.inf]*n
    dist_from_start, dist_from_end = [math.inf]*(n + 1), [math.inf]*(n + 1)
    dist_from_start[1] = dist_from_end[n] = 0    for i in range(2, n + 1):
        for j in range(max(1, i - m), i):
            if teleporters[j][i - j - 1] == '1':
                dist_from_start[i] = min(dist_from_start[i], dist_from_start[j] + 1)
    for i in range(n - 1, 0, -1):
        for j in range(i + 1, min(n + 1, i + m + 1)):
            if teleporters[i][j - i - 1] == '1':
                dist_from_end[i] = min(dist_from_end[i], dist_from_end[j] + 1)
    for k in range(2, n):
        for i in range(max(1, k - m), k):
            for j in range(k + 1, min(n + 1, i + m + 1)):
                if teleporters[i][j - i - 1] == '1':
                    min_teleports[k] = min(min_teleports[k], dist_from_start[i] + dist_from_end[j] + 1)
    min_teleports = [t if t < math.inf else -1 for t in min_teleports]
    return ' '.join(map(str, min_teleports[2:]))
 
if __name__ == '__main__':
    print(main())
```

## G - OR Sum 

### Solution 1:

```py

```

