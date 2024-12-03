# Codeforces Educational Round 151 Div 2

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

## 

### Solution 1: 

```py
def main():
    n, k, x = map(int, input().split())
    if x == 1:
        if k == 1 or (k == 2 and n & 1):
            return print('NO')
        res = []
        while n > 0:
            if n == 3:
                res.append(3)
                n -= 3
            if n >= 2:
                res.append(2)
                n -= 2
        print('YES')
        print(len(res))
        print(*res)
    else:
        print('YES')
        print(n)
        print(*[1] * n)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## 

### Solution 1: 

```py
def main():
    xa, ya = map(int, input().split())
    xb, yb = map(int, input().split())
    xc, yc = map(int, input().split())
    res = 1
    delta = lambda x, y: abs(x - y)
    if (xb > xa and xc > xa) or (xb < xa and xc < xa):
        res += min(delta(xa, xb), delta(xa, xc))
    if (yb > ya and yc > ya) or (yb < ya and yc < ya):
        res += min(delta(ya, yb), delta(ya, yc))
    print(res)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## 

### Solution 1: 

```py
from collections import deque

def main():
    s = input()
    m = int(input())
    left, right = list(map(int, input())), list(map(int, input()))
    digits = [deque() for _ in range(10)]
    for i, digit in enumerate(map(int, s)):
        digits[digit].append(i)
    index = -1
    for i in range(m):
        cur_index = index
        for dig in range(left[i], right[i] + 1):
            while digits[dig] and digits[dig][0] <= cur_index:
                digits[dig].popleft()
            if not digits[dig]:
                return print("YES")
            index = max(index, digits[dig][0])
    print("NO")

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## D. Rating System

### Solution 1:  greedy + prefix sum 

greedily calculate the best prefix sum that can be used to attain the best sum in the array

```py
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    best_psum = psum = best_sum = res = 0
    for num in arr:
        psum += num
        best_psum = max(best_psum, psum)
        if best_sum + num < best_psum:
            best_sum = best_psum
            res = best_psum
        else:
            best_sum += num
    print(res)
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## 

### Solution 1: 

```py

```

## F. Swimmers in the Pool

### Solution 1: 

```py

```