# Codeforces Round 892 Div 2

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

## A. United We Stand

### Solution 1: 

```py
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    mx = max(arr)
    b, c = [num for num in arr if num != mx], [num for num in arr if num == mx]
    if len(b) == 0 or len(c) == 0: return print(-1)
    print(len(b))
    print(len(c))
    print(*b)
    print(*c)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## B. Olya and Game with Arrays

### Solution 1: 

```py
import math

def main():
    n = int(input())
    min_arr = [math.inf] * n
    max_arr = [-math.inf] * n
    for i in range(n):
        m = int(input())
        arr = sorted(map(int, input().split()))
        min_arr[i] = arr[0]
        max_arr[i] = arr[1]
    min_in_max = min(max_arr)
    index = max_arr.index(min_in_max)
    min_arr_vals = min(min_arr)
    max_arr_vals = sum(x for i, x in enumerate(max_arr) if i != index)
    res = min_arr_vals + max_arr_vals
    print(res)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## C. Another Permutation Problem

### Solution 1: 

```py
def main():
    n = int(input())
    arr = list(range(1, n + 1))
    sum_, max_ = sum(p * i for i, p in enumerate(arr, start = 1)), max(p * i for i, p in enumerate(arr, start = 1))
    pmax = [p * i for i, p in enumerate(arr, start = 1)] 
    res = sum_ - max_
    arr.reverse()
    pmax.reverse()
    pmax.append(0)
    for j in range(1, n):
        for i in range(j, n, 2):
            left, right = 0, i
            cur_max = pmax[i + 1]
            cur_sum = sum_
            while left < right:
                left_n, right_n = n - left, n - right
                old_sum = arr[left] * left_n + arr[right] * right_n
                new_sum = arr[left] * right_n + arr[right] * left_n
                cur_sum += new_sum - old_sum
                cur_max = max(cur_max, arr[left] * right_n, arr[right] * left_n)
                left += 1
                right -= 1
            if left == right:
                cur_max = max(cur_max, arr[left] * (n - left))
            res = max(res, cur_sum - cur_max)
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

## 

### Solution 1: 

```py

```

## 

### Solution 1: 

```py

```