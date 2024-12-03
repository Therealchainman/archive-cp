# Codeforces Educational Round 146 Div 2

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

## A. Coins

### Solution 1:  math + parity

since 2*x will cover any even values, all that is required is that you can make n - k*y be even and greater than or equal to 0. 

```py
def main():
    n, k = map(int, input().split())
    for y in range(2):
        if (n - k*y)%2 == 0 and (n - k*y) >= 0:
            print("Yes")
            return
    print("No")

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## B. Long Legs

### Solution 1: math

It turns out that you at most will need to iterate to 10^5, even though a and b can be 10^9.  So the fact is that at some point around sqrt(max(a,b)) the function begins to decrease.  So it makes the most sense to only iterate up to that range and find the best k.  For any k if there is any remainder you could have got it from a prior k.  Since you had to increase k times to get to this point. 

```py
import math

def main():
    a, b = map(int, input().split())
    res = math.inf
    for k in range(1, 100_000):
        res = min(res, math.ceil(a / k) + math.ceil(b / k) + k - 1)
    print(res)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## C. Search in Parallel

### Solution 1:  two pointer + sort based on frequency

```py
def main():
    n, s1, s2 = map(int, input().split())
    freq = [0] + list(map(int, input().split()))
    colors = sorted(range(1, n + 1), key = lambda x: -freq[x])
    arrs = [[], []]
    times = [0]*2
    for i in range(n):
        if times[0] + s1 < times[1] + s2:
            arrs[0].append(colors[i])
            times[0] += s1
        else:
            arrs[1].append(colors[i])
            times[1] += s2
    for arr in arrs:
        print(len(arr), *arr)
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## D. 

### Solution 1:

```py

```

## E.

### Solution 1:

```py

```

## F.

### Solution 1:

In computing and graph theory, a dynamic connectivity structure is a data structure that dynamically maintains information about the connected components of a graph.

The set V of vertices of the graph is fixed, but the set E of edges can change. The three cases, in order of difficulty, are:

Edges are only added to the graph (this can be called incremental connectivity);
Edges are only deleted from the graph (this can be called decremental connectivity);
Edges can be either added or deleted (this can be called fully dynamic connectivity).

```py

```