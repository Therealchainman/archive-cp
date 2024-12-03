# Codeforces Round 885 Div 2

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

## 

### Solution 1:

```py
def main():
    n, m, k = map(int, input().split())
    x, y = map(int, input().split())
    manhattan_distance = lambda x1, y1, x2, y2: abs(x1 - x2) + abs(y1 - y2)
    res = True
    for _ in range(k):
        x1, y1 = map(int, input().split())
        if manhattan_distance(x, y, x1, y1) % 2 == 0: res = False
    print('YES' if res else 'NO')

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## 

### Solution 1:

```py
def main():
    n, k = map(int, input().split())
    paint = list(map(int, input().split()))
    dp1, dp2 = [0] * (1 + k), [0] * (1 + k)
    stack = [-1] * (k + 1)
    for i in range(n):
        color = paint[i]
        step = i - stack[color] - 1
        stack[color] = i
        if step > dp1[color]:
            dp2[color] = dp1[color]
            dp1[color] = step
        elif step > dp2[color]:
            dp2[color] = step
    for i in range(1, k + 1):
        step = n - stack[i] - 1
        if step > dp1[i]:
            dp2[i] = dp1[i]
            dp1[i] = step
        elif step > dp2[i]:
            dp2[i] = step
    res = min(max(dp1[i] // 2, dp2[i]) for i in range(1, k + 1))
    print(res)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## C. Vika and Price Tags

### Solution 1:  extended euclidean algorithm + math

if a > b
then this pattern happens
a,b,a-b,a-2b,b,a-3b,a-4b,b,...
Can derive formula a = kb + r,
so solve this equation and each time a < b swap the locations
return the number of steps, for if k is odd it will always be k + k // 2 + 1, which can do by swapping (b, r)

else you do (r, b)

```py
def extended_euclid(a, b):
    if a == 0: return 0
    if b == 0: return 1
    if a > b:
        k = a // b
        r = a % b
        return (extended_euclid(b, r) if k & 1 else extended_euclid(r, b)) + k + k // 2
    return 1 + extended_euclid(b, b - a)

def main():
    n = int(input())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    seen = set()
    for a, b in zip(A, B):
        if a == 0 and b == 0: continue
        seen.add(extended_euclid(a, b) % 3)
    if len(seen) <= 1:
        print('Yes')
    else:
        print('No')

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## D. Vika and Bonuses

### Solution 1:

```py

```

## 

### Solution 1:

```py

```