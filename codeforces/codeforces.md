# CODEFORCES

## AT THE TOP

```py
import os,sys
from io import BytesIO, IOBase
from typing import *
# sys.setrecursionlimit(1_000_000)

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

def main():
    n, h = map(int, input().split())
    dp = [[0]*(n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[0][i] = 1
    for height in range(1, n + 1):
        for i in range(1, n + 1):
            for j in range(i):
                dp[i][height] += dp[j][height - 1] * dp[i - j - 1][height - 1]
    return dp[n][n] - dp[n][h - 1]
if __name__ == '__main__':
    print(main())
```

## D. How many trees?

### Solution 1:  catalan number's + iterative dp + catalan number's with height of tree

```py
from itertools import product

def main():
    n, h = map(int, input().split())
    dp = [[0]*(n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[0][i] = 1
    for height, i in product(range(1, n + 1), repeat = 2):
        for j in range(i):
            dp[i][height] += dp[j][height - 1] * dp[i - j - 1][height - 1]
    return dp[n][n] - dp[n][h - 1]
    
if __name__ == '__main__':
    print(main())
```