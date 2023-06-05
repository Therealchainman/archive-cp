# Codeforces Round 876 Div 2

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
import math
 
def main():
    n, k = map(int, input().split())
    first, last = [0] * n, [0] * n
    for i in range(0, n, k):
        first[i] = 1
    for i in range(n - 1, -1, -k):
        last[i] = 1
    guess = [0] * n
    for i in range(n // 2 + 1):
        guess[i] = first[i]
    for i in range(n - 1, n // 2, -1):
        guess[i] = last[i]
    cur = 0
    for i in range(n):
        cur += guess[i]
        if cur < math.ceil((i + 1) / k):
            cur += 1
            guess[i] = 1
    cur = 0
    for i in reversed(range(n)):
        cur += guess[i]
        if cur < math.ceil((n - i) / k):
            cur += 1
            guess[i] = 1
    return sum(guess)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        print(main())
```

##

### Solution 1: 

```py
from heapq import heappop, heappush
 
def main():
    n = int(input())
    heap, active_heap = [], []
    for _ in range(n):
        a, b = map(int, input().split())
        heappush(heap, (a, -b))
    res = 0
    while heap:
        a, b = heappop(heap)
        b = -b
        res += b
        heappush(active_heap, a)
        x = len(active_heap)
        while active_heap and active_heap[0] <= x:
            heappop(active_heap)
        while heap and heap[0][0] <= x:
            heappop(heap)
    return res

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        print(main())
```

##

### Solution 1: 

```py
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    p = n - 1
    toggle = 0
    res = [None] * n
    for i in reversed(range(n)):
        while p >= 0 and arr[p] ^ toggle == 0:
            p -= 1
        if p + 1 > i: return print("NO")
        res[i] = p + 1
        toggle ^= 1
    print("YES")
    print(*res)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

##

### Solution 1: 

```py
def main():
    n = int(input())
    colors = list(map(int, input().split()))
    lis = []
    sz = 1
    for i in range(1, n):
        if colors[i] < colors[i - 1]:
            lis.append(sz)
            sz = 1
        else:
            sz += 1
    res = [None] * n
    res[0] = min(n - lis[0], n - lis[-1])
    for k in range(n):


    return ' '.join(map(str, res))
    

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        print(main())
```

##

### Solution 1: 

```py

```