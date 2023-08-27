# Codeforces Round 863 Div 3

## Notes

if the implementation is in python it will have this at the top of the python script for fast IO operations

```py
import os,sys
from io import BytesIO, IOBase
from typing import *
sys.setrecursionlimit(10**6)
# only use pypyjit when needed, it usese more memory, but speeds up recursion in pypy
import pypyjit
pypyjit.set_param('max_unroll_recursion=-1')
 
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

```cpp
#include <bits/stdc++.h>
using namespace std;

inline int read()
{
	int x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}

inline long long readll() {
	long long x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}
```

## A - Insert Digit

### Solution 1:  string

```py
def main():
    n, d = map(int, input().split())
    num = list(map(int, input()))
    for i in range(n):
        if d > num[i]:
            return f"{''.join(map(str, num[:i]))}{d}{''.join(map(str, num[i:]))}"
    return f"{''.join(map(str, num))}{d}"
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        print(main())
```

## B - Conveyor Belts

### Solution 1:  math

```py
def main():
    n, x1, y1, x2, y2 = map(int, input().split())
    adjust = lambda x: n - x + 1 if x > n//2 else x
    x1, y1, x2, y2 = map(adjust, (x1, y1, x2, y2))
    return abs(min(x1, y1) - min(x2, y2))
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        print(main())
```

## C - Restore the Array

### Solution 1:  greedy + push back

```py
def main():
    n = int(input())
    b = list(map(int, input().split()))
    a = b + [b[-1]]
    for i in range(n - 2):
        if max(b[i], b[i + 1]) != b[i]:
            a[i + 1] = b[i]
    print(*a)
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## D - Umka and a Long Flight

### Solution 1:  iterative dp + fibonacci + imbalanced segment search

fibonacci segment search I guess, it kind of reminds me of binary searching but doing it on different dimensions for when index is odd and even until it reaches a fibonacci square of size 1. For odd index it will check which square you take and find if point can be on other side of segment and then narrow the search along that dimension to look in that range now.  So for even index it looks in the column dimension or horizontal dimension and goes to the segment that doesn't contain point, so that updates the column range.  For odd index it looks in the row dimension or vertical dimension and goes to the segment that doesn't contain the poihnt, so updates row range. terminates if it can't find a segment that the point will not be contained within.

```py
def main():
    n, x, y = map(int, input().split())
    fib = [1, 1]
    for i in range(2, n + 2):
        fib.append(fib[-1] + fib[-2])
    ll, ur = (0, 0), (fib[n], fib[n + 1])
    for i, f in enumerate(reversed(fib[:-1])):
        if f == 1: break
        if i%2 == 0:
            left, right = ur[1] - f, ll[1] + f
            if left < y <= right:
                return "No"
            if y <= left:
                ur = (ur[0], left)
            else:
                ll = (ll[0], right)
        else:
            bottom, top = ur[0] - f, ll[0] + f
            if bottom < x <= top:
                return "No"
            if x <= bottom:
                ur = (bottom, ur[1])
            else:
                ll = (top, ll[1])
    return "Yes"
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        print(main())
```

## 

### Solution 1:

```py

```

## F. Is It Flower?

### Solution 1:  graph theory + degree + undirected connected component + undirected graph + bfs

The idea is that for a k flower, you know it will have k^2+k edges and k^2 vertices, so do some math and you can find k by taking num_edges - num_vetices and check that it satisfies the two constraints.  Then you need that the degree of all vertices is 2 or 4, and that all vertices are part of a single connected component.  Then you need to check that there are k components that are size k, where each edge in that connected component is not an edge that connects two vertex with degree 4, You can say those edges are the inner k cycle and we don't need to count that one.  We look at all of the k external cycles and check that they are k-cycles.  Which it is guaranteed if there are k - 1 vertex with degree = 2 and 1 vertex with degree = 4. 

```py
from collections import deque

def main():
    input()
    n, m = map(int, input().split())
    adj_list = [[] for _ in range(n + 1)]
    degrees = [0] * (n + 1)
    for _ in range(m):
        u, v = map(int, input().split())
        adj_list[u].append(v)
        degrees[v] += 1
        adj_list[v].append(u)
        degrees[u] += 1
    k = m - n
    # count check
    if k*k + k != m or k*k != n:
        return "No"
    # degree check
    if any(deg not in (2, 4) for deg in degrees[1:]):
        return "No"
    # check all are part of a single connected component
    def bfs1(node):
        cnt = 0
        queue = deque([node])
        vis = [0]*(n + 1)
        vis[node] = 1
        while queue:
            node = queue.popleft()
            cnt += 1
            for nei in adj_list[node]:
                if vis[nei]: continue
                vis[nei] = 1
                queue.append(nei)
        return cnt == n
    if not bfs1(1):
        return "No"
    # connectivity check
    visited = [0]*(n + 1)
    def bfs2(node):
        size = 0
        queue = deque([node])
        visited[node] = 1
        while queue:
            node = queue.popleft()
            size += 1
            for nei in adj_list[node]:
                if visited[nei] or degrees[node] == degrees[nei] == 4: continue
                visited[nei] = 1
                queue.append(nei)
        return size == k
    num_components = 0
    for i in range(1, n + 1):
        if visited[i]: continue
        num_components += 1
        if not bfs2(i):
            return "No"
    return "Yes" if num_components == k else "No"
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        print(main())
```

## G1. Vlad and the Nice Paths

### Solution 1:

```py

```