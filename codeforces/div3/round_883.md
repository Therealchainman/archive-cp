# Codeforces Round 883 Div 3

## Notes

if the implementation is in python it will have this at the top of the python script for fast IO operations

```py
import os,sys
from io import BytesIO, IOBase
from typing import *
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
#define int long long

inline int read() {
	int x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}
```

## A. Rudolph and Cut the Rope

### Solution 1: 

```py
def main():
    n = int(input())
    res = 0
    for _ in range(n):
        a, b = map(int, input().split())
        if a > b: res += 1
    print(res)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## B. Rudolph and Tic-Tac-Toe

### Solution 1: 

```py
def main():
    n = 3
    grid = [input() for _ in range(n)]
    for r in range(n):
        if all(grid[r][c] == 'X' for c in range(n)):
            return print('X')
        if all(grid[r][c] == 'O' for c in range(n)):
            return print('O')
        if all(grid[r][c] == '+' for c in range(n)):
            return print('+')
    for c in range(n):
        if all(grid[r][c] == 'X' for r in range(n)):
            return print('X')
        if all(grid[r][c] == 'O' for r in range(n)):
            return print('O')
        if all(grid[r][c] == '+' for r in range(n)):
            return print('+')
    if all(grid[r][r] == 'X' for r in range(n)):
        return print('X')
    if all(grid[r][r] == 'O' for r in range(n)):
        return print('O')
    if all(grid[r][r] == '+' for r in range(n)):
        return print('+')
    if all(grid[r][n - r - 1] == 'X' for r in range(n)):
        return print('X')
    if all(grid[r][n - r - 1] == 'O' for r in range(n)):
        return print('O')
    if all(grid[r][n - r - 1] == '+' for r in range(n)):
        return print('+')
    print('DRAW')

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## C. Rudolf and the Another Competition

### Solution 1: 

```py
def main():
    n, h, m = map(int, input().split())
    player = [None] * n
    for i in range(n):
        problems = sorted(list(map(int, input().split())))
        time = penalty = points = 0
        for j in range(m):
            time += problems[j]
            if time > h: break
            penalty += time
            points += 1
        player[i] = (-points, penalty, i + 1)
    player.sort()
    for i, (_, _, idx) in enumerate(player):
        if idx == 1: return print(i)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## D. Rudolph and Christmas Tree

### Solution 1: 

```py
def main():
    n, d, h = map(int, input().split())
    y = list(map(int, input().split()))
    ratio = d / h
    triangle = lambda b, h: b * h / 2
    area = n * triangle(d, h)
    for i in range(1, n):
        delta = y[i] - y[i - 1]
        base = d - delta * ratio
        height = h - delta
        if height > 0:
            area -= triangle(base, height)
    print(area)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## E2. Rudolf and Snowflakes

### Solution 1:  math + binary search

This code shows how it exceeds 10^18 limit on the k = 1,000,000 for 1 + k + k^2 + k^3.
so it only goes up to 1,000,000 and that is all needs to be precomputed

So the all that is not precomputed is all the possible values for 1 + k + k^2, or when p = 2 if you write the snowflake as 1 + k + k^2 + ... + k^p

```py
for k in range(2, 1_000_001):
    v = 1 + k + k**2 + k**3
    if v >= 10**18:
        print(f'{v:,}')
```

This proves that with the smallest k = 2, that you will exceed max once p = 59, so really just need p < 60

```py
v = 0
for i in range(60):
    v += 2**i
    if v > 10**18:
        print(i)
print(f"{v:,}")
```

The only snowflakes not solved by precomputing the first 1 million terms or 1,000,000 terms, is the case of 1 + k + k^2,  Cause in this instance k can be really large, it can go all the way up to 1,000,000,000 as can be proved with this code. 

This is too large to precompute, so the only way is to binary search this specific case. 

```py
for k in range(999_999_980, 1_000_500_000):
    v = 1 + k + k**2
    if v > 10**18:
        print(f"{k:,}", f"{v:,}")
        break
```

```py
vis = set()
INF = int(1e18)

def main():
    n = int(input())
    if n in vis: return print("Yes")
    # solve equation 1 + k + k^2 == n with binary search
    quadratic = lambda k: 1 + k + k**2
    left, right = 2, 1_000_000_000
    while left < right:
        mid = (left + right) >> 1
        if quadratic(mid) == n: return print("Yes")
        if quadratic(mid) <= n: left = mid + 1
        else: right = mid
    print("No")

if __name__ == '__main__':
    for k in range(2, 1_000_001):
        term = k
        num_vertices = 1 + k
        for p in range(59):
            term *= k
            num_vertices += term
            if num_vertices > INF: break
            vis.add(num_vertices)
    T = int(input())
    for _ in range(T):
        main()
```

## F. Rudolph and Mimic

### Solution 1: 

```py
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    # initial counts
    counts = [0] * 10
    for num in arr:
        counts[num] += 1
    print('- 0', flush = True)
    guess = False
    while True:
        arr = list(map(int, input().split()))
        ncounts = [0] * 10
        for num in arr:
            ncounts[num] += 1
        mimic = 0
        for i in range(10):
            if ncounts[i] > counts[i]:
                mimic = i
                break
        if mimic == 0:
            print('- 0', flush = True)
            continue
        if guess:
            print('!', arr.index(mimic) + 1, flush = True)
            break
        guess = True
        index = []
        counts = [0] * 10
        for i in range(len(arr)):
            if arr[i] != mimic:
                index.append(i + 1)
            else:
                counts[arr[i]] += 1
        print(f"- {len(index)} {' '.join(map(str, index))}", flush = True)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## G. Rudolf and CodeVid-23

### Solution 1: 

```py
import math
from heapq import heappop, heappush

def main():
    n, m = map(int, input().split())
    initial = int(input(), 2)
    meds = [None] * m
    for i in range(m):
        d = int(input())
        relief_mask = int(input(), 2)
        sick_mask = int(input(), 2)
        meds[i] = (d, relief_mask, sick_mask)
    min_days = [math.inf] * (1 << n)
    min_days[initial] = 0
    min_heap = [(0, initial)]
    while min_heap:
        days, mask = heappop(min_heap)
        if days > min_days[mask]: continue
        if mask == 0: return print(min_days[mask])
        for d, relief_mask, sick_mask in meds:
            new_mask = mask
            new_mask &= ~relief_mask
            new_mask |= sick_mask
            new_days = days + d
            if new_days < min_days[new_mask]:
                min_days[new_mask] = new_days
                heappush(min_heap, (new_days, new_mask))
    print(-1)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```