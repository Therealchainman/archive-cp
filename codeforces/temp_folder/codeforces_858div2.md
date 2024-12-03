# Codeforces Round 858 Div 2

## Notes

if the implementation is in python it will have this at the top of the python script for fast IO operations

```py
import os,sys
from io import BytesIO, IOBase
from typing import *
 
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

## A. Walking Master

### Solution 1:

```py
def main():
    a, b, c, d = map(int, input().split())
    delta_y = d - b
    delta_x = a + delta_y - c
    return delta_x + delta_y if delta_x >= 0 and delta_y >= 0 else -1


if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        print(main())
```

## B. Mex Master

### Solution 1:

```py
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    count_zeros = sum([1 for i in arr if i == 0])
    if count_zeros <= (n + 1) // 2:
        return 0
    count_ones = sum([1 for i in arr if i == 1])
    if count_ones + count_zeros == n and count_ones > 0:
        return 2
    return 1

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        print(main())
```

## C. Sequence Master

### Solution 1:

```py
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    if n == 1:
        return sum(num - arr[0] for num in arr)
    min_dist = sum(abs(num) for num in arr)
    if n == 2:
        min_dist = min(min_dist, sum(abs(num - 2) for num in arr))
    if n%2 == 0:
        dist = sum(abs(num + 1) for num in arr)
        for num in arr:
            delta = abs(num - n)
            removal = abs(num + 1)
            min_dist = min(min_dist, dist + delta - removal)
    return min_dist

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        print(main())
```

## D. DSU Master

### Solution 1:

```py

```

## E. Tree Master

### Solution 1:

```py

```