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

```py
def main():
    n = int(input())
    N = 10**18
    for d in range(3, 64):
        left, right = 2, int(pow(N, 1 / (d - 1))) + 10
        while left <= right:
            mid = (left + right) >> 1
            value = (mid**d - 1) // (mid - 1)
            if value < n:
                left = mid + 1
            else:
                right = mid - 1
            if (mid**d - 1) % (mid - 1) == 0 and (mid**d - 1) // (mid - 1) == n:
                return print("YES")
    print("NO")

if __name__ == '__main__':
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