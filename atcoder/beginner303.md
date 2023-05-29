# Atcoder Beginner Contest 303

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

## G - Bags Game 

### Solution 1:  dynammic programming + prefix sum + O(n^3) time

This will time out obviously, need a solution that can solve in O(n^2) 

```py
from itertools import accumulate

def main():
    N, A, B, C, D = map(int, input().split())
    arr = list(map(int, input().split()))
    psum = [0] + list(accumulate(arr))
    def psum_query(left, right):
        return psum[right + 1] - psum[left]
    dp = [[0]*(N+1) for _ in range(N + 1)]
    for i in range(N): # base cases
        dp[i][0] = 0
        dp[i][1] = arr[i]
    for j in range(2, N + 1):
        for i in range(N):
            r = i + j - 1
            if r >= N: break # out of bounds, right boundary is i + j - 1
            # TAKE LEFTMOST ELEMENT
            dp[i][j] = arr[i] - dp[i + 1][j - 1]
            # TAKE RIGHTMOST ELEMENT
            dp[i][j] = max(dp[i][j], arr[r] - dp[i][j - 1])
            # TAKING B ELEMENTS AT COST A
            take = min(B, j)
            # take k elements from left side and take - k elements from right side
            for k in range(take + 1):
                dp[i][j] = max(dp[i][j], psum_query(i, r) - psum_query(i + k, r - (take - k)) - dp[i + k][j - take] - A)
            # TAKING D ELEMENTS AT COST C
            take = min(D, j)
            for k in range(take + 1):
                dp[i][j] = max(dp[i][j], psum_query(i, r) - psum_query(i + k, r - (take - k)) - dp[i + k][j - take] - C)
    return dp[0][N]

if __name__ == '__main__':
    print(main())
```

### Solution 2:  dynamic programming + 

```py

```