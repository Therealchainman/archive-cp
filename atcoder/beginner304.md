# Atcoder Beginner Contest 304

## What is used at the top of each submission

```py
import os,sys
from io import BytesIO, IOBase
sys.setrecursionlimit(10**6)
from typing import *
# only use pypyjit when needed, it usese more memory, but speeds up recursion in pypy
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')
# sys.stdout = open('output.txt', 'w')

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
                    
if __name__ == '__main__':
    print(main())
    # main()
    # sys.stdout.close()
```

## A - First Player 

### Solution 1: 

```py
def main():
    n = int(input())
    names = [None] * n
    ages = [None] * n
    for i in range(n):
        name, age = input().split()
        names[i] = name
        ages[i] = int(age)
    index = min(range(n), key=lambda i: ages[i])
    for i in range(n):
        name = names[(index + i) % n]
        print(name)
 
if __name__ == '__main__':
    main()
```

## B - Subscribers 

### Solution 1: 

```py
def main():
    n = list(input())
    for i in range(3, len(n)):
        n[i] = '0'
    return ''.join(n)
 
if __name__ == '__main__':
    print(main())
```

## C - Virus 

### Solution 1:  directed graph + dfs

```py
def main():
    n, d = map(int, input().split())
    people = [tuple(map(int, input().split())) for _ in range(n)]
    euclidean_dist = lambda p1, p2: (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
    adj_list = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i):
            if euclidean_dist(people[i], people[j]) <= d**2:
                adj_list[i].append(j)
                adj_list[j].append(i) 
    stack = [0]
    vis = [0] * n
    vis[0] = 1
    while stack:
        node = stack.pop()
        for nei in adj_list[node]:
            if vis[nei]: continue
            vis[nei] = 1
            stack.append(nei)
    for i in range(n):
        if vis[i]:
            print("Yes")
        else:
            print("No")
 
if __name__ == '__main__':
    main()
```

## D - A Piece of Cake 

### Solution 1:  binary search + grid + counter

```py
import bisect
from collections import Counter
 
def main():
    R, C = map(int, input().split())
    n = int(input())
    strawberries = [tuple(map(int, input().split())) for _ in range(n)]
    A = int(input())
    arr = list(map(int, input().split()))
    B = int(input())
    brr = list(map(int, input().split()))
    counts = Counter()
    for x, y in strawberries:
        vertical = bisect.bisect_left(arr, x)
        horizontal = bisect.bisect_left(brr, y)
        counts[(vertical, horizontal)] += 1
    m = min(counts.values()) if len(counts) == (A + 1) * (B + 1) else 0
    M = max(counts.values())
    print(m, M)
 
if __name__ == '__main__':
    main()
```

## E - Good Graph 

### Solution 1:  condense graph to connected components + union find

```py
class UnionFind:
    def __init__(self, n: int):
        self.size = [1]*n
        self.parent = list(range(n))
    
    def find(self,i: int) -> int:
        while i != self.parent[i]:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i
 
    """
    returns true if the nodes were not union prior. 
    """
    def union(self,i: int,j: int) -> bool:
        i, j = self.find(i), self.find(j)
        if i!=j:
            if self.size[i] < self.size[j]:
                i,j=j,i
            self.parent[j] = i
            self.size[i] += self.size[j]
            return True
        return False
    
    def __repr__(self) -> str:
        return f'parents: {[(i, parent) for i, parent in enumerate(self.parent)]}, sizes: {self.size}'
 
def main():
    n, m = map(int, input().split())
    dsu = UnionFind(n + 1)
    for _ in range(m):
        u, v = map(int, input().split())
        dsu.union(u, v)
    k = int(input())
    seen = set()
    for _ in range(k):
        u, v = map(lambda x: dsu.find(x), map(int, input().split()))
        seen.add((min(u, v), max(u, v)))
    q = int(input())
    for _ in range(q):
        c1, c2 = map(lambda x: dsu.find(x), map(int, input().split()))
        if (min(c1, c2), max(c1, c2)) in seen:
            print("No")
        else:
            print("Yes")
 
if __name__ == '__main__':
    main()
```

## F - Shift Table 

### Solution 1:  count ways for factors + inclusion exclusion in dynammic programming to remove duplicates

This is a tricky problem for me, I still have trouble understanding how the dp part can remove duplicates.  An example of a duplicates is something like this, suppose you have n = 12
M = 3, and M = 6.  And we know that for 3 that these are the elements that are fixed represented by 1 and the others means that Aoki has option to work or not work on that day.  The fixed ones means Aoki must work on that day else, not all days will be worked by either Takahashi or Aoki. 

So suppose input is 
12
####.####.##
then we know that for M = 3, [1,1,0] that is Aoki must work on first and second day, and then this will be repeated through out.  Now we are going to count the number of possiblities here which there is 2. 

Then for M = 6, [0,0,0,1,1,0] there are two days that Aoki must work and the rest there are two possiblities.  We can see from above that there is a duplicate state we will be counting, which is this one, 
[1,1,0,1,1,0], Thus we need to conclucde that any pattern in M = 3 will be repeated in M = 6, so take the patterns in 3 and subtract them from those in M = 6.  That way those repeated patterns will not count in M = 6 and the remaining ones must be patterns that were not in M = 3. 

Remember M = 6 counted patterns that are same as M = 3 and those that are not, so we want to exclude those repeated and include those that are not.  This is the idea of inclusion exclusion.

```py
def main():
    n = int(input())
    s = input()
    mod = 998244353
    factors = []
    for i in range(1, n):
        # i is factor if n is divisible by i
        if n % i == 0: factors.append(i)
    m = len(factors)
    dp = [[0] * m for _ in range(m)]
    for i in range(m):
        dp[i][i] = 1 # i is divisible by i
        for j in range(i):
            if factors[i] % factors[j] == 0: dp[i][j] = 1 # factor_i is divisible by factor_j
    # count the ways
    counts = [0] * m
    for i in range(m):
        # finds position that must be fixed, that is takahashi doesn't work that day so that '.'
        fixed = [0] * factors[i]
        for j in range(n):
            if s[j] == '.': fixed[j % factors[i]] = 1
        unset = factors[i] - sum(fixed)
        counts[i] = pow(2, unset, mod)
    # dynamic programming to remove the duplicates
    for i in range(m):
        for j in range(i):
            if dp[i][j]:
                counts[i] -= counts[j]
    print(counts)
    return sum(counts) % mod

if __name__ == '__main__':
    print(main())
```

## Ex - Constrained Topological Sort 

### Solution 1: 

```py

```