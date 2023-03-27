# Codeforces Round 859 Div 4

## Notes

if the implementation is in python it will have this at the top of the python script for fast IO operations

```py
import os,sys
from io import BytesIO, IOBase
from typing import *
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

## A. Plus or Minus

### Solution 1:

```py
def main():
    a, b, c = map(int, input().split())
    if a + b == c:
        print('+')
    else:
        print('-')
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## B. Grab the Candies

### Solution 1:

```py
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    even_sum, odd_sum = sum([x for x in arr if x%2 == 0]), sum([x for x in arr if x%2 == 1])
    if even_sum > odd_sum:
        print("yes")
    else:
        print("no")
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## C. Find and Replace

### Solution 1: set + set intersection + bipartite

```py
def main():
    n = int(input())
    s = input()
    odd, even = set(), set()
    for i, ch in enumerate(s):
        if i % 2 == 0:
            even.add(ch)
        else:
            odd.add(ch)
    ovlap = odd & even
    if ovlap:
        print('no')
    else:
        print('yes')
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## D.  Odd Queries

### Solution 1: prefix sum + query sum

```py
def main():
    n, q = map(int, input().split())
    arr = list(map(int, input().split()))
    psum = [0]*(n + 1)
    for i in range(n):
        psum[i + 1] = psum[i] + arr[i]
    for _ in range(q):
        left, right, k = map(int, input().split())
        len_ = right - left + 1
        left_sum, right_sum = psum[left - 1], psum[-1] - psum[right]
        ext_sum = left_sum + right_sum
        query_sum = k*len_
        sum_ = ext_sum + query_sum
        if sum_&1:
            print("yes")
        else:
            print("no")
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## E. Interview

### Solution 1:  binary search + prefix sum

interactive problem

```py
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    psum = [0]*(n + 1)
    for i in range(n):
        psum[i+1] = psum[i] + arr[i]
    left, right = 0, n - 1
    while left < right:
        mid = (left + right) >> 1
        size = mid - left + 1
        print('?', size, *range(left + 1, mid + 2), flush = True)
        x = int(input())
        if x > psum[mid + 1] - psum[left]:
            right = mid
        else:
            left = mid + 1
    print('!', left + 1, flush = True)
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## F. Bouncy Ball

### Solution 1: simulation + memoization

```py
def main():
    n, m, sx, sy, tx, ty, sd = input().split()
    n, m, sx, sy, tx, ty = map(int, (n, m, sx, sy, tx, ty))
    dirs = {'DL': (1, -1), 'DR': (1, 1), 'UL': (-1, -1), 'UR': (-1, 1)}
    dirs_inv = {v: k for k, v in dirs.items()}
    not_wall = lambda x, y: 1 < x < n and 1 < y < m
    in_bounds = lambda x, y : 0 < x <= n and 0 < y <= m
    def from_flip(x, y, dx, dy):
        if x == 1:
            dx = 1
        elif x == n:
            dx = -1
        if y == 1:
            dy = 1
        elif y == m:
            dy = -1
        return dirs_inv[(dx, dy)]
    # SIMULATE UNTIL HITS THE FIRST WALL
    dx, dy = dirs[sd]
    if (sx, sy) == (tx, ty): 
        print(0)
        return
    while not_wall(sx, sy) or in_bounds(sx + dx, sy + dy):
        sx += dx
        sy += dy
        if (sx, sy) == (tx, ty): 
            print(0)
            return
    bounces = 1
    sd = from_flip(sx, sy, dx, dy)
    hits_target = set()
    for d, (dx, dy) in dirs.items():
        x, y = tx, ty
        dx, dy = -dx, -dy
        while not_wall(x, y) or in_bounds(x + dx, y + dy):
            x += dx
            y += dy
        hits_target.add((x, y, d))
    # SIMULATION OF BALL BOUNCES UNTIL IT HITS TARGET
    seen = set([(sx, sy, sd)])
    while (sx, sy, sd) not in hits_target:
        bounces += 1
        dx, dy = dirs[sd]
        max_x = max_y = 0
        if dx == 1:
            max_x = n - sx
        else:
            max_x = sx - 1
        if dy == 1:
            max_y = m - sy
        else:
            max_y = sy - 1
        min_xy = min(max_x, max_y)
        sx += min_xy * dx
        sy += min_xy * dy
        sd = from_flip(sx, sy, dx, dy)
        if (sx, sy, sd) in seen: 
            bounces = -1
            break
        seen.add((sx, sy, sd))
    print(bounces)
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## G. Subsequence Addition

### Solution 1:  iterative dp

```py
def main():
    n = int(input())
    arr = sorted(list(map(int, input().split())))
    if arr[0] != 1:
        print('no')
        return
    right = 1
    for i in range(1, n):
        if arr[i] <= right:
            right += arr[i]
        else:
            print('no')
            return
    print('yes')
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```