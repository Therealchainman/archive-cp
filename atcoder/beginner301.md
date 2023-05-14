# Atcoder Beginner Contest 301

## What is used at the top of each submission

```py
import os,sys
from io import BytesIO, IOBase
sys.setrecursionlimit(10**6)
from typing import *
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
```

##

### Solution 1:  

```py

```

##

### Solution 1:  

```py

```

##

### Solution 1:  

```py

```

##

### Solution 1:  

```py

```

##

### Solution 1:  

```py
from itertools import product
import math
import heapq
from collections import defaultdict

def main():
    H, W, T = map(int, input().split())
    A = [input() for _ in range(H)]
    start, goal, empty, wall, candy = ['S', 'G', '.', '#', 'o']
    start_pos = goal_pos = None
    candy_arr = []
    index = {}
    adj_list = 
    for i, j in product(range(H), range(W)):
        if A[i][j] == start:
            start_pos = (i, j)
        elif A[i][j] == goal:
            goal_pos = (i, j)
        elif A[i][j] == candy:
            index[(i, j)] = len(candy_arr)
            candy_arr.append((i, j))
    in_bounds = lambda r, c: 0 <= r < H and 0 <= c < W
    minheap = [(0, 0, *start_pos, 0)]
    max_candies = defaultdict(lambda: -math.inf)
    max_candies[(*start_pos, 0)] = 0
    res = -1
    while minheap:
        # print('heap', minheap)
        candies, moves, r, c, candy_mask = heapq.heappop(minheap)
        if moves > T: continue
        candies = abs(candies)
        # print(r, c, candies, moves)
        if (r, c) == goal_pos:
            res = max(res, candies)
            if res == len(candy_arr): break
        for nr, nc in [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]:
            if not in_bounds(nr, nc) or A[nr][nc] == wall: continue
            if A[nr][nc] == candy and (candy_mask>>index[(nr, nc)])&1: continue
            ncandies = candies + (A[nr][nc] == candy)
            nmoves = moves + 1
            if A[nr][nc] == candy:
                ncandy_mask = candy_mask | ((A[nr][nc] == candy) << index[(nr, nc)])
            else:
                ncandy_mask = candy_mask
            if max_candies[(nr, nc, nmoves)] < ncandies:
                max_candies[(nr, nc, nmoves)] = ncandies
                heapq.heappush(minheap, (-ncandies, nmoves, nr, nc, ncandy_mask))
    return res
                    
if __name__ == '__main__':
    print(main())
    # main()

```

##

### Solution 1:  

```py

```