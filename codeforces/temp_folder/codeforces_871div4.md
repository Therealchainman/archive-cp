# Codeforces Round 871 Div 4

## Notes

if the implementation is in python it will have this at the top of the python script for fast IO operations

```py
import os,sys
from io import BytesIO, IOBase
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

## A. Love Story

### Solution 1:  edit distance or hamming distance

```py
def main():
    s = input()
    t = "codeforces"
    res = 0
    for i in range(len(s)):
        res += (s[i] != t[i])
    return res
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        print(main())
```

## B. Blank Space

### Solution 1:  max size of sliding window

```py
from itertools import groupby
 
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    res = 0
    for key, grp in groupby(arr):
        if key == 0:
            res = max(res, len(list(grp)))
    return res
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        print(main())
```

## C. Mr. Perfectly Fine

### Solution 1: brute force

```py
import math
 
def main():
    n = int(input())
    res = math.inf
    skills = [math.inf]*2
    for _ in range(n):
        cost, skillset = input().split()
        cost = int(cost)
        if skillset == '11':
            res = min(res, cost)
        elif skillset == '10':
            skills[0] = min(skills[0], cost)
        elif skillset == '01':
            skills[1] = min(skills[1], cost)
    res = min(res, sum(skills))
    return res if res < math.inf else -1
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        print(main())
```

## D. Gold Rush

### Solution 1:  visit array + dfs with stack

```py
def main():
    n, m = map(int, input().split())
    vis = set()
    stack = [n]
    while stack:
        x = stack.pop()
        if x == m: return "YES"
        if x%3 == 0:
            x1, x2 = x//3, x - x//3
            if x1 not in vis:
                vis.add(x1)
                stack.append(x1)
            if x2 not in vis:
                vis.add(x2)
                stack.append(x2)
    return "NO"
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        print(main())
```

## E. The Lakes

### Solution 1:  dfs + flood fill

```py
def main():
    n, m = map(int, input().split())
    grid = [list(map(int, input().split())) for _ in range(n)]
    res = 0
    in_bounds = lambda r, c: 0 <= r < n and 0 <= c < m
    def dfs(r, c):
        ans = grid[r][c]
        stack = [(r, c)]
        grid[r][c] = 0
        while stack:
            r, c = stack.pop()
            for nr, nc in [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]:
                if not in_bounds(nr, nc) or grid[nr][nc] == 0: continue
                stack.append((nr, nc))
                ans += grid[nr][nc]
                grid[nr][nc] = 0
        return ans
    for r, c in product(range(n), range(m)):
        if grid[r][c] == 0: continue
        res = max(res, dfs(r, c))
    return res

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        print(main())
```

## F. Forever Winter

### Solution 1: undirected graph + degree count

leaf nodes will have degree count = 1, so we can find the count of leaf nodes. 
The remaining challenge is to find the degree of the intermediate nodes, that are between central and leaf nodes. 
if central and intermediate nodes have same degree, then it will just be the count of degree - 1.
if central and intermediate nodes have different degree, the central nodes degree will appear just once, and the intermediates will appear more than once.  So just get the counts of the intermediate nodes, can figure out from this observation which one it is. 

```py
def main():
    n, m = map(int, input().split())
    adj_list = [[] for _ in range(n + 1)]
    degree = [0]*(n + 1)
    for _ in range(m):
        u, v = map(int, input().split())
        adj_list[u].append(v)
        adj_list[v].append(u)
        degree[u] += 1
        degree[v] += 1
    leaf_count = degree.count(1)
    cnt_unique = len(set(degree)) - 1
    central_count = 0
    if cnt_unique == 2:
        central_count = degree.count(max(degree)) - 1
    else:
        for v in set(degree):
            if v in (0, 1): continue
            central_count = max(central_count, degree.count(v))
    return ' '.join(map(str, [central_count, leaf_count//central_count]))
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        print(main())
```

## G. Hits Different

### Solution 1: math

sum of sequence of squares of natural numbers is equal to n(n + 1)(2n + 1)/6

All you need to track is the lower and upper bound or min and max value in each row.  Iterate backwards in the rows, and note that you can decrement the lower bound by the r - 1, and the upper bound by only r, the reason for that is because of the image description.
Then the formula above can be applied.

![hits different](images/hits_different.png)

```py
def main():
    n = int(input())
    delta = 1
    row = total = i = 0
    rows = [None]*2023
    while total < n:
        cur = [total + 1, total + delta]
        total += delta
        delta += 1
        row += 1
        rows[i] = cur
        i += 1
    res = n*n
    min_val = max_val = n
    sum_squares = lambda x: x*(x + 1)*(2*x + 1)//6
    for r in range(row - 1, 0, -1):
        min_val = max(rows[r - 1][0], min_val - r - 1)
        max_val = min(rows[r - 1][1], max_val - r)
        row_sum = sum_squares(max_val) - sum_squares(min_val - 1)
        res += row_sum
    return res
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        print(main())
```

## H. Don't Blame Me

### Solution 1: iterative dp + bit manipulation

dp array tracks the total number of ways to get to a specific integer, so dp[5] is number of ways can get 5 so far.  This represents the prefix ands in a sense.  That is this 5 could have been formed by any number of ways 5, 5&5, 5&7&7, or whatever. 

for each element in the array, get the count from each prefix, so if there was prefix = 5, which can be formed many ways, just add all the ways to the current integer but take prefix & arr[i], so and the prefix with the current integer.  This will give another prefix and and you know the number of ways you can form that will be the addition of the number of ways to formed that prefix. 

At the end, take the current counts for current element and add it to the dp, to update the total for all prefix. 

basically if the last iteration had 7 ways, you will end up with a sum(counts) = 8, cause those 7 ways will form some new prefixes with new counts that have this integer
so you are considering this pattern
0001.
notice that the last digit is 1, because it needs to be non-empty subsequence, 
so the 0001 represents just arr[i]
but you take all the prefix configurations which will be 2^3, cause 3 0s.  so can be 001, 010, 100,...,111 and add them to current which will consider the count with all those 8 prefixes, then you add them to the total dp, so that means you now are considering 0000, with all configurations minus 1, because can't contain empty set. 

To include empty set just need to initialize dp[0] = 1.

At end all you care about is when the bit_count is equal to k, so sum up those. 

```py
def main():
    n, k = map(int, input().split())
    arr = list(map(int, input().split()))
    dp = [0]*(64)
    mod = int(1e9) + 7
    for i in range(n):
        counts = [0]*64
        counts[arr[i]] = 1
        for j in range(64):
            counts[arr[i] & j] += dp[j]
        for j in range(64):
            dp[j] += counts[j]
            dp[j] %= mod
    return sum(dp[i] for i in range(64) if bin(i)[2:].count('1') == k) % mod

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        print(main())
```

