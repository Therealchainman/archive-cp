# Codeforces Round 918 Div 4

## A. Odd One Out

### Solution 1:  counter

```py
def main():
    nums = map(int, input().split())
    freq = Counter(nums)
    for k, v in freq.items():
        if v == 1: print(k)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## B. Not Quite Latin Square

### Solution 1:  list of sets

```py
from itertools import product

def main():
    grid = [list(input()) for _ in range(3)]
    rows = [set("ABC") for _ in range(3)]
    cols = [set("ABC") for _ in range(3)]
    for r, c in product(range(3), repeat = 2):
        if grid[r][c] == "?": continue
        rows[c].remove(grid[r][c])
        cols[r].remove(grid[r][c])
    for i in range(3):
        if len(rows[i]) > 0: print(rows[i].pop())

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## C. Can I Square?

### Solution 1:  check if squared

```py
import math

def main():
    n = int(input())
    x = sum(map(int, input().split()))
    if math.sqrt(x) == int(math.sqrt(x)):
        print("YES")
    else:
        print("NO")

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## D. Unnatural Language Processing

### Solution 1:  deque, pattern matching, conditional

```py
from collections import deque

def main():
    n = int(input())
    s = input()
    consonants = "bcd"
    vowels = "ae"
    p = "".join(["C" if ch in consonants else "V" for ch in s])
    dots = deque()
    for i in range(1, n):
        if p[i] == p[i - 1] == "C":
            dots.append(i)
        elif i > 1 and p[i - 1] == "C" and p[i] == p[i - 2] == "V":
            dots.append(i - 1)
    ans = []
    for i in range(n):
        if dots and dots[0] == i:
            ans.append(".")
            dots.popleft()
        ans.append(s[i])
    print("".join(ans))

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## E. Romantic Glasses

### Solution 1:  prefix sum, sort

```py
from itertools import accumulate

def main():
    n = int(input())
    arr = list(map(int, input().split()))
    arr = [arr[i] if i % 2 == 0 else -arr[i] for i in range(n)]
    psum = sorted([0] + list(accumulate(arr)))
    for i in range(1, n + 1):
        if psum[i - 1] == psum[i]: return "YES"
    return "NO"

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        print(main())
```

### Solution 2:  prefix sum, set

This gets TLE because set solution was hacked

```py
from itertools import product

def main():
    n = int(input())
    arr = list(map(int, input().split()))
    epsum = opsum = 0
    seen = set([0])
    for i in range(n):
        if i % 2 == 0: epsum += arr[i]
        else: opsum += arr[i]
        delta = epsum - opsum
        if delta in seen: return "YES"
        seen.add(delta)
    return "NO"

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        print(main())
```

## F. Greetings

### Solution 1:  fenwick tree, sum range query, point update, coordinate compression, sort

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

    def query_range(self, i, j):
        return self.query(j) - self.query(i - 1)

    def __repr__(self):
        return f"array: {self.sums}"

def main():
    n = int(input())
    index_map = {}
    points = []
    endpoints = []
    for _ in range(n):
        a, b = map(int, input().split())
        points.append((a, b))
        endpoints.append(b)
    for b in sorted(endpoints):
        index_map[b] = len(index_map) + 1
    points.sort()
    fenwick = FenwickTree(n)
    ans = 0
    for a, b in points:
        cnt = fenwick.query_range(index_map[b], n)
        ans += cnt
        fenwick.update(index_map[b], 1)
    print(ans)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## G. Bicycles

### Solution 1:  Graph, Dijkstra, Dynamic Programming

```py
from heapq import heappop, heappush

def main():
    N, M = map(int, input().split())
    adj = [[] for _ in range(N)]
    for _ in range(M):
        u, v, w = map(int, input().split())
        u -= 1
        v -= 1
        adj[u].append((v, w))
        adj[v].append((u, w))
    slowness = list(map(int, input().split()))
    vis = set()
    min_heap = [(0, 0, slowness[0])]
    while min_heap:
        cost, u, s = heappop(min_heap)
        s = min(s, slowness[u])
        if (u, s) in vis: continue
        if u == N - 1: return cost
        vis.add((u, s))
        for v, w in adj[u]:
            heappush(min_heap, (cost + s * w, v, s))
    return 0

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        print(main())
```

