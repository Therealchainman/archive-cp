# Atcoder Beginner Contest 301

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

##

### Solution 1:  

```py

```

##

### Solution 1:  

```py

```

## C - AtCoder Cards

### Solution 1:  frequency arrays + counters + anagram

```py
from itertools import product
import math
import heapq
from collections import Counter
import string
 
def main():
    s = input()
    t = input()
    freq_s, freq_t = Counter(s), Counter(t)
    free_s, free_t = s.count('@'), t.count('@')
    chars = 'atcoder'
    for ch in string.ascii_lowercase:
        if freq_s[ch] != freq_t[ch] and ch not in chars: return 'No'
        if freq_s[ch] < freq_t[ch]:
            delta = freq_t[ch] - freq_s[ch]
            if free_s < delta: return 'No'
            free_s -= delta
        elif freq_s[ch] > freq_t[ch]:
            delta = freq_s[ch] - freq_t[ch]
            if free_t < delta: return 'No'
            free_t -= delta
    return 'Yes'
 
                    
if __name__ == '__main__':
    print(main())
```

## D - Bitmask 

### Solution 1:  bit manipulation + greedy

```py
def main():
    s = list(reversed(input()))
    n = int(input())
    res = 0
    for i in range(len(s)):
        if s[i] == '1':
            res |= (1 << i)
    if res > n: return -1
    for i in reversed(range(len(s))):
        if s[i] == '?' and (res | (1 << i)) <= n:
            res |= (1 << i)
    return res
if __name__ == '__main__':
    print(main())
```

## E - Pac-Takahashi 

### Solution 1:  traveling salesman problem + dp bitmask + (n^2 * 2^n) time

Given a list of cities and the distances between each pair of cities, what is the shortest possible route that visits each city exactly once and returns to the origin city?

minimize number of moves for the dp[i][mask], where i is the current node you are visiting and mask contains the set of all nodes that have been visited, which in this case will tell how many candies have been collected. 

Find the shortest distance between all pairs of vertices with bfs with the adjacency matrix

loop through each mask or set of visited vertices, then loop through the src and dst vertex and consider this. 

```py
from itertools import product
import math
from collections import deque

def main():
    H, W, T = map(int, input().split())
    A = [input() for _ in range(H)]
    start, goal, empty, wall, candy = ['S', 'G', '.', '#', 'o']
    in_bounds = lambda r, c: 0 <= r < H and 0 <= c < W
    neighborhood = lambda r, c: [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
    # CONSTRUCT THE VERTICES
    start_pos = goal_pos = None
    nodes = []
    index = {}
    for i, j in product(range(H), range(W)):
        if A[i][j] == start:
            start_pos = (i, j)
        elif A[i][j] == goal:
            goal_pos = (i, j)
        elif A[i][j] == candy:
            nodes.append((i, j))
    nodes.extend([goal_pos, start_pos])
    for i, (r, c) in enumerate(nodes):
        index[(r, c)] = i
    V = len(nodes)
    adj_matrix = [[math.inf]*V for _ in range(V)] # complete graph so don't need adjacency list
    # CONSTRUCT THE EDGES WITH BFS
    for i, (r, c) in enumerate(nodes):
        queue = deque([(r, c)])
        dist = 0
        vis = set([(r, c)])
        while queue:
            dist += 1
            for _ in range(len(queue)):
                row, col = queue.popleft()
                for nr, nc in neighborhood(row, col):
                    if not in_bounds(nr, nc) or A[nr][nc] == wall or (nr, nc) in vis:
                        continue
                    if (nr, nc) in index:
                        adj_matrix[i][index[(nr, nc)]] = dist
                    vis.add((nr, nc))
                    queue.append((nr, nc))
    # TRAVELING SALESMAN PROBLEM
    dp = [[math.inf]*(1 << (V - 1)) for _ in range(V)]
    dp[V - 1][0] = 0 # start at node 0 with no candy
    for mask in range(1 << (V - 1)):
        for i in range(V):
            if dp[i][mask] == math.inf: continue
            for j in range(V - 1):
                nmask = mask | (1 << j)
                dp[j][nmask] = min(dp[j][nmask], dp[i][mask] + adj_matrix[i][j])
    res = -1
    for mask in range(1 << (V - 1)):
        if dp[V - 2][mask] > T: continue
        res = max(res, bin(mask).count('1') - 1)
    return res
                    
if __name__ == '__main__':
    print(main())

```

##

### Solution 1:  

```py

```

##

### Solution 1:  

```py

```