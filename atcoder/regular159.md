# Atcoder Regular Contest 159

## What is used at the top of each submission

```py
import os,sys
from io import BytesIO, IOBase
sys.setrecursionlimit(10**6)
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

##

### Solution 1:

```py
from itertools import product
from collections import deque

def main():
    n, k = map(int, input().split())
    matrix = [list(map(int, input().split())) for _ in range(n)]
    adj_list = [[] for _ in range(n)]
    for i, j in product(range(n), repeat = 2):
        if matrix[i][j] == 1:
            adj_list[i].append(j)
    q = int(input())
    def bfs(src, dst):
        visited = set()
        s_sec = src//n
        d_sec = dst//n
        dst %= n
        src %= n
        visited.add((src, s_sec))
        queue = deque([(src, s_sec, 0)])
        while queue:
            node, sec, dist = queue.popleft()
            if node == dst and d_sec == sec: return dist
            for nei in adj_list[node]:
                if (nei, d_sec) in visited: continue
                visited.add((nei, d_sec))
                queue.append((nei, d_sec, dist + 1))
        return -1
    for _ in range(q):
        u, v = map(int, input().split())
        u -= 1
        v -= 1
        print(bfs(u, v))

if __name__ == '__main__':
    main()
```

##

### Solution 1:

```py
import math

def main():
    a, b = map(int, input().split())
    cnt = 0
    while a > 0 and b > 0:
        g = math.gcd(a, b)
        m = 1
        if cnt > 100_000 and g > 1:
            ma, mb = a//g, b//g
            m = min(ma, mb)
        a -= m*g
        b -= m*g
        cnt += m
    return cnt

if __name__ == '__main__':
    print(main())
```

##

### Solution 1:

```py
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    arr = [[x, i] for i, x in enumerate(arr)]
    mx = 10_000
    results = []
    for i in range(mx):
        arr = sorted(arr, reverse = True)
        if arr[0][0] == arr[-1][0]: break
        res = [0]*n
        for j in range(n):
            arr[j][0] += j + 1
            idx = arr[j][1]
            res[idx] = j + 1
        results.append(res)
    arr = sorted(arr, reverse = True)
    if arr[0][0] != arr[-1][0]: 
        print("No")
        return
    print("Yes")
    print(len(results))
    for res in results:
        print(*res)

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

##

### Solution 1:

```py

```