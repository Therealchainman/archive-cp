# Atcoder Beginner Contest 310

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
def main():
    n, p, q = map(int, input().split())
    d = list(map(int, input().split()))
    res = min(p, q + min(d))
    print(res)

if __name__ == '__main__':
    main()
```

## 

### Solution 1: 

```py
def main():
    n, m = map(int, input().split())
    prices = [None] * n
    functions = [set() for _ in range(n)]
    for i in range(n):
        arr = list(map(int, input().split()))
        prices[i] = arr[0]
        for j in range(2, len(arr)):
            functions[i].add(arr[j])
    indices = sorted(range(n), key = lambda i: prices[i])
    for idx in range(n):
        for jdx in range(idx + 1, n):
            i, j = indices[jdx], indices[idx]
            common = functions[i] & functions[j]
            if len(common) < len(functions[i]): continue
            if prices[i] == prices[j] and len(functions[j]) == len(common): continue
            return print('Yes')
    print('No')

if __name__ == '__main__':
    main()
```

## 

### Solution 1: 

```py
def main():
    n = int(input())
    vis = set()
    res = 0
    for _ in range(n):
        s = input()
        res += s not in vis
        vis.add(s)
        vis.add(s[::-1])
    print(res)

if __name__ == '__main__':
    main()
```

## 

### Solution 1: 

```py
from functools import lru_cache

def main():
    N, T, M = map(int, input().split())
    incompatible_masks = set()
    end_mask = (1 << N) - 1
    for _ in range(M):
        a, b = map(int, input().split())
        mask = (1 << a) | (1 << b)
        incompatible_masks.add(mask)    
    @lru_cache(None)
    def dfs(i, mask):
        if i == T: return mask == end_mask
        if mask == end_mask: return 0
        cnt = 0
        for team_mask in range(1, 1 << N):
            if team_mask & mask: continue
            cnt += dfs(i + 1, mask | team_mask)
        if mask == 7:
            print('i', i, 'mask', mask, 'cnt', cnt)
        return cnt
    res = dfs(0, 0)
    print(res)

if __name__ == '__main__':
    main()
```

## 

### Solution 1: 

```py
def main():
    n = int(input())
    arr = list(map(int, list(input())))
    ones = zeros = res = 0
    for num in arr:
        ones, zeros = zeros + (1 if num else ones), ones if num else 1
        res += ones
    return res

if __name__ == '__main__':
    main()
```

## 

### Solution 1: 

```py

```

## 

### Solution 1: 

```py

```