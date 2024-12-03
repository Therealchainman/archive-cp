# Codeforces Educational Round 149 Div 2

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

## A. Grasshopper on a Line

### Solution 1:

```py
def main():
    n, k = map(int, input().split())
    if n % k != 0: 
        print(1)
        print(n)
    else:
        print(2)
        print(1, n - 1)
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## B. Comparison String

### Solution 1:

```py
from itertools import groupby
    
def main():
    n = int(input())
    s = input()
    res = 0
    for _, grp in groupby(s):
        res = max(res, len(list(grp)) + 1)
    print(res)
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## C. Best Binary String

### Solution 1:

```py
def main():
    s = input()
    n = len(s)
    res = [None]*n
    for i in range(n):
        if s[i] == '?':
            res[i] = 0
        else:
            break
    for i in reversed(range(n)):
        if s[i] == '?':
            res[i] = 1
        else:
            break
    left = None
    stack = []
    for i in range(n):
        if s[i] == '?' and left is not None:
            stack.append(i)
        elif s[i] != '?':
            res[i] = int(s[i])
            if s[i] == '1' == left:
                val = 1
            else:
                val = 0
            while stack:
                idx = stack.pop()
                res[idx] = val
            left = s[i]
    print(''.join(map(str, res)))
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## D. Bracket Coloring

### Solution 1:

```py
def main():
    n = int(input())
    s = input()
    if n & 1 or s.count('(') != n // 2:
        print(-1)
        return
    stack, rev_stack = [], []
    grp1, grp2 = [], []
    for i in range(n):
        if s[i] == ')' and stack:
            grp1.extend([i, stack.pop()])
        elif s[i] == ')':
            rev_stack.append(i)
        elif s[i] == '(' and rev_stack:
            grp2.extend([i, rev_stack.pop()])
        elif s[i] == '(':
            stack.append(i)
    colors = [1]*n
    if grp1 and grp2:
        for i in grp2:
            colors[i] = 2
    print(len(set(colors)))
    print(*colors)
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
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