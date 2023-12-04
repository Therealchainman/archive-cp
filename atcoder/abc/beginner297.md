# Atcoder Beginner Contest 297

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

## A - Double Click 

### Solution 1:  loop

```py
def main():
    n, d = map(int, input().split())
    arr = list(map(int, input().split()))
    for i in range(1, n):
        if arr[i] - arr[i - 1] <= d: return arr[i]
    return -1
 
if __name__ == '__main__':
    print(main())
```

## B - chess960 

### Solution 1: string

```py
def main():
    s = input()
    b_locations = []
    r_locations = []
    k_loc = None
    for i in range(len(s)):
        if s[i] == "B": b_locations.append(i)
        elif s[i] == "R": r_locations.append(i)
        elif s[i] == 'K': k_loc = i
    return "Yes" if b_locations[0]%2 != b_locations[1]%2 and r_locations[0] < k_loc < r_locations[1] else "No"
 
if __name__ == '__main__':
    print(main())
```

## C - PC on the Table 

### Solution 1:  matrix loop

```py
def main():
    h, w = map(int, input().split())
    mat = [list(input()) for _ in range(h)]
    for i in range(h):
        j = 0
        while j + 1 < w:
            if mat[i][j] == mat[i][j + 1] == 'T':
                mat[i][j] = 'P'
                mat[i][j + 1] = 'C'
                j += 1
            j += 1
    for row in mat:
        print(''.join(row))
 
if __name__ == '__main__':
    main()
```

## D - Count Subtractions 

### Solution 1:  math

Observe that the difference a - b is always equal to the b (smaller element) and will be the same for some multiple times until a becomes equal to or smaller than b. 

```py
def main():
    a, b = map(int, input().split())
    res = 0
    while a != b:
        if a < b:
            a, b = b, a
        m = a // b
        if m > 1: m -= 1
        a -= m*b
        res += m
    print(res)

if __name__ == '__main__':
    main()
```

## E - Kth Takoyaki Set 

### Solution 1:  pointer pointing to smallest value for each coin + add coin to cost that is smallest

```py
import math
    
def main():
    n, k = map(int, input().split())
    coins = list(set(map(int, input().split())))
    pointers = [0]*len(coins)
    costs = [0]*(k + 1)
    for r in range(1, k + 1):
        cost = math.inf
        for i in range(len(coins)):
            j = pointers[i]
            ncost = costs[j] + coins[i]
            if ncost < cost:
                cost = ncost
        costs[r] = cost
        for i in range(len(coins)):
            while costs[pointers[i]] + coins[i] == cost:
                pointers[i] += 1
    print(costs[-1])
 
if __name__ == '__main__':
    main()
```

## F - Minimum Bounding Box 2 

### Solution 1:

```py

```

## G - Constrained Nim 2 

### Solution 1:  sprague grund theorem + nim game + impartial games

Just look at sprague grundy numbers and find a pattern that reduces the required operations/iterations

What you find is that you can represent the nim value for each pile as the p%(L+R)//L and this kind of looks like this for 3, 14
nim values will be 
0: 0,1,2
1: 3,4,5
2: 6,7,8
3: 9,10,11
4: 12,13,14
5: 15, 16

Which kind of makes sense if you think about the 1 nim value, if you have 3, 4, 5 only 1 possible move can happen, you take 3 stones and no more moves are possible after that so in a sense the 3,4,5 represent a single stone in the classical nim game.  And 0 makes sense as well it represents 0 stones, nobody can take stones.
But for 2, why should this represent 2 stones, it's optional you could take all 8 stones but if you take minimum you can take from it at most 2 times.  It is a bit difficult to prove this but if you take an exmaple of piles = [4, 7], so they are 1 and 2, you get 1^2 = 3, and you can check that the first player can win no matter what happens. And you can check with 1^1 or 2^2 that first player can't win. 

```py
import operator
from functools import reduce

def main():
    N, L, R = map(int, input().split())
    piles = list(map(int, input().split()))
    xor_sum = reduce(operator.xor, [p%(L + R)//L for p in piles])
    return "First" if xor_sum > 0 else "Second"

if __name__ == '__main__':
    print(main())
```

This one belows help find the pattern, cause it is finding winning and losing states with brute force algorithm.  So it is deadly slow but that is how you can start finding the solution. 

```py
def main(N, L, R, piles):
    total = sum(piles)
    def grundy(idx, remaining, num_piles):
        # winning state for player
        if num_piles == 1 and L <= remaining <= R: return 1 
        # losing state for player
        if num_piles == 1 and remaining < L: return 0
        grundy_numbers = set()
        for i in range(N):
            for take in range(L, min(R, piles[i]) + 1):
                piles[i] -= take
                new_num_piles = num_piles - (1 if piles[i] < L else 0)
                grundy_numbers.add(grundy(idx + 1, remaining - take, new_num_piles))
                piles[i] += take
        res = next(dropwhile(lambda i: i in grundy_numbers, range(100_000)))
        return res
    num_piles = sum((1 for p in piles if p >= L))
    grundy_number = grundy(0, total, num_piles)
    return "First" if grundy_number > 0 else "Second"
```