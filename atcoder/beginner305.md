# Atcoder Beginner Contest 305

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

## A - Water Station 

### Solution 1:  math

```py
def main():
    n = int(input())
    quotient = n // 5
    n1, n2 = quotient * 5, (quotient + 1) * 5
    if abs(n - n1) < abs(n - n2):
        print(n1)
    else:
        print(n2)

if __name__ == '__main__':
    main()
```

## B - ABCDEFG 

### Solution 1:  successor graph + search

```py
def main():
    p, q = map(int, input().split())
    adj_list = {"A": ("B", 3), "B": ("C", 1), "C": ("D", 4), "D": ("E", 1), "E": ("F", 5), "F": ("G", 9)}
    p, q = min(p, q), max(p, q)
    node = p
    res = 0
    while node != q:
        node, cost = adj_list[node]
        res += cost
    print(res)

if __name__ == '__main__':
    main()
```

## C - Snuke the Cookie Picker 

### Solution 1:  matrix + count adjacent cells with cookies, if greater than or equal to 2 than it is cookie taken

```py
def main():
    h, w = map(int, input().split())
    grid = [list(input()) for _ in range(h)]
    in_bounds = lambda r, c: 0 <= r < h and 0 <= c < w
    neighborhood = lambda r, c: [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]
    for r, c in product(range(h), range(w)):
        if grid[r][c] == '#': continue
        cnt = 0
        for nr, nc in neighborhood(r, c):
            if not in_bounds(nr, nc): continue
            cnt += grid[nr][nc] == '#'
        if cnt > 1: return print(f"{r + 1} {c + 1}")

if __name__ == '__main__':
    main()
```

## D - Sleep Log 

### Solution 1:  binary search for prefix and suffix endpoints + count middle with fenwick tree queries

```py
class FenwickTree:
    def __init__(self, N):
        self.sums = [0 for _ in range(N+1)]

    def update(self, i, delta):
        while i < len(self.sums):
            self.sums[i] += delta
            i += i & (-i)

    def query(self, i):
        res = 0
        while i > 0:
            res += self.sums[i]
            i -= i & (-i)
        return res

    def __repr__(self):
        return f"array: {self.sums}"
    
import bisect

def main():
    n = int(input())
    arr = list(map(int, input().split()))
    q = int(input())
    fenwick_tree = FenwickTree(n + 1)
    for i in range(2, n, 2):
        sleep = arr[i] - arr[i - 1]
        fenwick_tree.update(i, sleep)
    for _ in range(q):
        left, right = map(int, input().split())
        i, j = bisect.bisect_left(arr, left), bisect.bisect_left(arr, right)
        cur = 0
        if i % 2 == 0:
            cur += arr[i] - left
        if j % 2 == 0:
            cur += right - arr[j - 1]
        middle = fenwick_tree.query(j - 1) - fenwick_tree.query(i)
        cur += middle
        print(cur)

if __name__ == '__main__':
    main()
```

## E - Art Gallery on Graph 

### Solution 1:  undirected graph + max heap + visited array based on maximum remaining stamina

```py
import heapq

def main():
    n, m, k = map(int, input().split())
    adj_list = [[] for _ in range(n + 1)]
    for _ in range(m):
        u, v = map(int, input().split())
        adj_list[u].append(v)
        adj_list[v].append(u)
    max_heap = []
    vis = [-1] * (n + 1)
    for _ in range(k):
        p, h = map(int, input().split())
        max_heap.append((-h, -p))
        vis[p] = h
    heapq.heapify(max_heap)
    while max_heap:
        h, p = map(abs, heapq.heappop(max_heap))
        if vis[p] > h: continue
        for nei in adj_list[p]:
            nh = h - 1
            if vis[nei] >= nh: continue
            vis[nei] = nh
            heapq.heappush(max_heap, (-nh, -nei))
    print(sum(1 for v in vis if v >= 0))
    print(' '.join(map(str, [i for i in range(1, n + 1) if vis[i] >= 0])))

if __name__ == '__main__':
    main()
```

## F - Dungeon Explore 

### Solution 1:  dfs on hidden graph + takes at most 2*N visits + visit each node at most 2 times + stack + visited array + O(n^2)

```py
def main():
    n, m = map(int, input().split())
    vis = [0] * (n + 1)
    v = 1
    vis[1] = 1
    stack = [1]
    while True:
        inp = input()
        if inp == 'OK' or inp == '-1': break
        vertices = list(map(int, inp.split()))
        next_ = 0
        for u in reversed(vertices[1:]):
            if vis[u] == 0:
                next_ = u
                break
        if next_:
            v = next_
            stack.append(v)
            print(v, flush = True)
        else:
            stack.pop()
            v = stack[-1]
            print(v, flush = True)
        vis[v] = 1

if __name__ == '__main__':
    main()
```

## G - Banned Substrings 

### Solution 1: 

```py

```