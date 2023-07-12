# Atcoder Regular Contest 164

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

## A - Ternary Decomposition 

### Solution 1: 

3^37 < 10^18

and use 3^0 10^18 times 
could use about 3^1 on order 10^17 times (binary search)
use 3^2 also about 10^17 times (binary search)
3^3 is on order 10^16 (binary search)

```py
for i in range(1_000):
    v = pow(3, i)
    if v > 10**18:
        print('i', i, f"{v:,}")
        break
```

```py

```

## B - Switching Travel 

### Solution 1:  spanning tree + union find

Create connected components with edges that are two colors.  Then for the edges the single color, check if both the nodes of that edge are already part of connected component, cause that means there is a path going from one color to the other color all the way to this single color edge.  One of the nodes on the single color edge will be the starting point of the path, and capable to return to it. 

```py
class UnionFind:
    def __init__(self, n: int):
        self.size = [1]*n
        self.parent = list(range(n))
    
    def find(self,i: int) -> int:
        while i != self.parent[i]:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    """
    returns true if the nodes were not union prior. 
    """
    def union(self,i: int,j: int) -> bool:
        i, j = self.find(i), self.find(j)
        if i!=j:
            if self.size[i] < self.size[j]:
                i,j=j,i
            self.parent[j] = i
            self.size[i] += self.size[j]
            return True
        return False
    
    def __repr__(self) -> str:
        return f'parents: {[(i, parent) for i, parent in enumerate(self.parent)]}, sizes: {self.size}'

def main():
    n, m = map(int, input().split())
    edges = [None] * m
    for i in range(m):
        u, v = map(int, input().split())
        edges[i] = (u, v)
    # 0 or 1
    colors = [0] + list(map(int, input().split()))
    dsu = UnionFind(n + 1)
    for u, v in edges:
        if colors[u] ^ colors[v]:
            dsu.union(u, v)
    for u, v in edges:
        if colors[u] == colors[v] and dsu.find(u) == dsu.find(v):
            return print("Yes")
    print("No")

if __name__ == '__main__':
    main()
```

### Solution 2:  dfs + undirected graph + find cycles + backtrack on cycles

This doesn't work, but for reasons you wouldn't expect.

This code seems correct, but it just doesn't appear to finish.  

It is said the graph is connected, so starting with node 1 is fine. for a graph with over 130,000 nodes, it runs in the dfs for only over 35,000 calls, which means it is not traversing all the edges and some reason stopping.  But there is no error output, and nothing can be printed after the dfs(1, 0)

```py
def main():
    n, m = map(int, input().split())
    adj_list = [[] for _ in range(n + 1)]
    for _ in range(m):
        u, v = map(int, input().split())
        adj_list[u].append(v)
        adj_list[v].append(u)
    # 0 or 1
    colors = [0] + list(map(int, input().split()))
    vis_color = [0] * (n + 1)
    parent_arr = [0] * (n + 1)
    def dfs(node, parent):
        # found a cycle
        if vis_color[node] == 1:
            # backtrack through parent array
            cnt = int(colors[node] == colors[parent])
            cur = parent
            color = colors[cur]
            while cur != node:
                cur = parent_arr[cur]
                cnt += colors[cur] == color
                color = colors[cur]
            # return true if it is a valid cycle
            # the only valid cycle is when there is only two adjacent nodes with same color in cycle
            return cnt == 1
        parent_arr[node] = parent
        vis_color[node] = 1 # exploring
        for nei in adj_list[node]:
            if nei == parent or vis_color[nei] == 2: continue
            if dfs(nei, node): return True
        vis_color[node] = 2 # finished exploring from this node
        return False
    possible = dfs(1, 0)
    res = "Yes" if possible else "No"
    print(res)

if __name__ == '__main__':
    main()
```

## C - Reversible Card Game

### Solution 1:  math + dynamic programming

The observation that needs to be made is that, Bob can always take the card on the side he wants which is the higher valued side if there is an even number of cards in the desired direction.  The reason is because Alice will have to flip one of the cards and thus it will be odd number in desired direction.  Which means it must be 1 or more, and cannot be zero.  So there is always a card in the desired direction.  So Bob can always take the card on the side he wants.

However if there is only an odd number of cards in the desired direction, then When alice flips it will be come even, and could be zero.  The best option is for Bob to take one card not in desired direction, and pick the remaining even cards in desired direction so he can get them all.  The best card to do this with will be card with the minimum difference between it's red color and blue colored side values. 

```py
import math

def main():
    n = int(input())
    res = cnt = 0
    min_delta = math.inf
    for _ in range(n):
        a, b = map(int, input().split())
        if a > b:
            res += a
            cnt += 1
            min_delta = min(min_delta, a - b)
        else:
            res += b
            min_delta = min(min_delta, b - a)
    if cnt & 1:
        res -= min_delta
    print(res)

if __name__ == '__main__':
    main()
```

## 

### Solution 1: 

```py

```