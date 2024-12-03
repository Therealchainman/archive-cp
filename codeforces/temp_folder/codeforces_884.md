# Codeforces Round 884

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

## A. Subtraction Game

### Solution 1: math

Basically with a + b, player 1 is either going to pick a or b, and regardless a or b will remain so player 2 picks that and now 0 remains and player 1 cannot make amove and thus loses. 

Also player 2 will always win after making 1 move. 

And player 2 wins in the 2nd round.

```py
def main():
    a, b = map(int, input().split())
    print(a + b)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## B. Permutations & Primes

### Solution 1:  greedy + observation

It's just realizing that the optimal approach is to put the 1 in the middle and the 2 and 3 outside. 

This is because any subarray that contains 1 is going to have mex that is prime, except for the one that includes 1, 2, and 3.  That one will most likely not in many cases.

```py
def main():
    n = int(input())
    if n == 1: return print(1)
    arr = [0] * n
    arr[n // 2 + 1] = 1
    p = 4
    for i in range(n):
        if i == 0: arr[i] = 2
        elif i == n // 2 + 1: arr[i] = 1
        elif i == n: arr[i] = 3
        else: 
            arr[i] = p
            p += 1
    print(*arr)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## C. Particles

### Solution 1:  dynamic programming + even odd index

Basically you can take from even or odd indexes

```py
def main():
    n = int(input())
    charges = list(map(int, input().split()))
    even = odd = 0
    if all(x < 0 for x in charges): return print(max(charges))
    for i in range(n):
        if i & 1:
            odd += max(0, charges[i])
        else:
            even += max(0, charges[i])
    print(max(even, odd))

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

### Solution 1:  union find + 2 colorability

```py
class UnionFind:
    def __init__(self):
        self.size = dict()
        self.parent = dict()

    def exists(self, i):
        return i in self.parent
    
    def find(self, i):
        if not self.exists(i):
            self.size[i] = 1
            self.parent[i] = i
        while i != self.parent[i]:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i, j):
        i, j = self.find(i), self.find(j)
        if i!=j:
            if self.size[i] < self.size[j]:
                i,j=j,i
            self.parent[j] = i
            self.size[i] += self.size[j]
            return True
        return False
    
    @property
    def root_count(self):
        return sum(node == self.find(node) for node in self.parent)

    def __repr__(self) -> str:
        return f'parents: {[(i, parent) for i, parent in enumerate(self.parent)]}, sizes: {self.size}'

def main():
    n, m, k = map(int, input().split())
    dsu = UnionFind()
    for _ in range(k):
        x1, y1, x2, y2 = map(int, input().split())
        

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