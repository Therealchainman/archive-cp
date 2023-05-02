# Atcoder Beginner Contest 300

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

## A - N-choice question

### Solution 1:  loop

```py
def main():
    n, a, b = map(int, input().split())
    c = list(map(int, input().split()))
    for i, v in enumerate(c, start = 1):
        if a + b == v: return i
    return n
 
if __name__ == '__main__':
    print(main())
```

## B - Same Map in the RPG World 

### Solution 1:  modulus + all + cartesian product loop

```py
from itertools import product
 
def main():
    h, w = map(int, input().split())
    arr1 = [list(input()) for _ in range(h)]
    arr2 = [list(input()) for _ in range(h)]
    for r, c in product(range(h), range(w)):
        if all(arr1[(r + i)%h][(c + j)%w] == arr2[i][j] for i, j in product(range(h), range(w))):
            return 'Yes'
    return 'No'
 
if __name__ == '__main__':
    print(main())
```

## C - Cross 

### Solution 1:  dfs 

dfs from each source of cross, and check if the continually larger cross is valid.  anything invalid with a size = 1 is said to be a cross of size 0, which is basically not a cross. 

```py
def main():
    h, w = map(int, input().split())
    grid = [list(input()) for _ in range(h)]
    cross = '#'
    n = min(h, w)
    res = [0]*(n + 1)
    stack = [(r, c, 0) for r, c in product(range(h), range(w)) if grid[r][c] == cross]
    in_bounds = lambda r, c: 0 <= r < h and 0 <= c < w
    while stack:
        r, c, size = stack.pop()
        flag = False
        for nr, nc in ((r + size, c + size), (r - size, c + size), (r + size, c + size), (r - size, c - size)):
            if not in_bounds(nr, nc) or grid[nr][nc] != cross:
                flag = True
                continue
        if flag:
            res[size - 1] += 1
        else:
            stack.append((r, c, size + 1))
    return ' '.join(map(str, res[1:]))
                    
if __name__ == '__main__':
    print(main())
```

## D - AABCC 

### Solution 1:  prime sieve + math + number theory

The integers needed are less than expected because these are two the power, easiest way is two loops through the squared terms because these will terminate quickly.  And move just the middle pointer to the just small enough prime, if not possible then impossible with this c and any larger c.

```py
import math
 
def main():
    n = int(input())
    def prime_sieve(lim):
        sieve,primes = [[] for _ in range(lim)], []
        for integer in range(2,lim):
            if not len(sieve[integer]):
                primes.append(integer)
                for possibly_divisible_integer in range(integer,lim,integer):
                    current_integer = possibly_divisible_integer
                    while not current_integer%integer:
                        sieve[possibly_divisible_integer].append(integer)
                        current_integer //= integer
        return primes
    threshold = math.ceil(math.sqrt(n/12))
    primes = prime_sieve(threshold + 1)
    m = len(primes)
    res = 0
    for a in range(m):
        b = a + 1
        for c in range(a + 2, m):
            constant = primes[a]**2*primes[c]**2
            while constant*primes[b] > n and b > a + 1:
                b -= 1
            if constant*primes[b] > n: break
            while constant*primes[b + 1] <= n and b + 1 < c:
                b += 1
            res += b - a
    return res
 
                    
if __name__ == '__main__':
    print(main())
```

## E - Dice Product 3 

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