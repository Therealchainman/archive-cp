# Constructor's Open Cup 2024

## Practice Round

## A. Houses

### Solution 1:  math

```py
def main():
    x1, x2, x = map(int, input().split())
    if x1 > x2: x1, x2 = x2, x1
    print("YES" if x1 <= x <= x2 else "NO")

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## B. Books in Boxes

### Solution 1: 

```py
def check(a, b, c):
    for d in range(-10, 11):
        na = a + d 
        nb = b - d 
        if na >= 0 and nb >= 0 and na % 10 == nb % 10 == c: return True
    return False
 
def main():
    a, b, c = map(int, input().split())
    print("YES" if check(a, b, c) else "NO")
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## C. Decompression

### Solution 1: 

```py
def main():
    s = input()
    ans = []
    for i in range(1, len(s), 2):
        c1, c2 = s[i-1], s[i]
        if c1.isdigit():
            ans.append(c2 * int(c1))
        else:
            ans.append(c1 * int(c2))
    print("".join(ans))
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## D. Letter

### Solution 1: 

```py
def main():
    n, k = map(int, input().split())
    ans = window = 0
    for _ in range(n):
        s = input()
        if window + len(s) > k:
            ans += 1
            window = 0
        window += len(s) + 1
    if window > 0: ans += 1
    print(ans)
 
if __name__ == '__main__':
    T = 1
    for _ in range(T):
        main()
```

## E. Dice Game

### Solution 1: 

```py
def main():
    n, m = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    ans = ""
    if sum(A) > len(B):
        ans += "Y"
    else:
        ans += "N"
    start, end = max(len(A), len(B)), min(sum(A) + 1, sum(B) + 1)
    if end > start:
        ans += "Y"
    else:
        ans += "N"
    if sum(B) > len(A):
        ans += "Y"
    else:
        ans += "N"
    print(ans)
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## F. City Plan

### Solution 1: 

```py
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    queries = sorted([(arr[i], i) for i in range(n)], reverse = True)
    end = queries[0][0]
    ans = [0] * n
    seen = set()
    for d, i in queries:
        if d not in seen:
            ans[i] = d 
        else:
            ans[i] = end - d
        seen.add(d)
    print(*ans)
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## G. Coin Tossing

### Solution 1:  sliding window

```py
def main():
    n, k = map(int, input().split())
    s = input()
    l = wcount = 0
    ans = 1
    for r in range(1, n):
        if s[r] != s[r - 1]: wcount += 1
        while wcount > k:
            if s[l] != s[l + 1]: wcount -= 1
            l += 1
        ans = max(ans, r - l + 1)
    print(ans)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## H. GCD Set

### Solution 1:  primes, math

Using the idea if you multiple all the primes by x, than it is guaranteed that each integer is x * a different prime, and thus the gcd between any pair of integer is going to be x.

```py
PRIMES = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523]

def main():
    n, x = map(int, input().split())
    ans = [0] * n
    for i in range(n):
        ans[i] = x * PRIMES[i]
    print(*ans)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## I. Pairing Numbers

### Solution 1:  sort, greedy

pair up largest negative numbers, and then pair any remaining negatives with 0, or just leave one negative unpaired,  the only other thing than a negative to pair would be a 0 though.  
Than you can pair up the positive integers.  with the largest first, and than once a pair is less than the sum of the small numbers don't pair them okay. 


```py
from collections import deque
 
def main():
    n = int(input())
    dq = deque(sorted(map(int, input().split())))
    ans = 0
    while len(dq) > 1:
        u, v = dq.popleft(), dq.popleft()
        if u > 0 or v > 0: 
            dq.appendleft(v)
            dq.appendleft(u)
            break
        ans += u * v
    while len(dq) > 1:
        u, v = dq.pop(), dq.pop()
        if u * v < u + v: 
            dq.extend([u, v])
            break
        ans += u * v 
    print(ans + sum(dq))
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## J. Pirate Chest

### Solution 1:  ranges, intersection of 1d ranges

```py
def intersection(r1, r2):
    return max(r1[0], r2[0]), min(r1[1], r2[1])
 
def length(s, e):
    return e - s
 
def main():
    n = int(input())
    ranges = [[0, 10**9 + 1]]
    arr = [0] * n
    arr[0] = int(input())
    for i in range(1, n):
        nranges = []
        a, b = map(int, input().split())
        arr[i] = a
        cur, prv = arr[i], arr[i - 1]
        if cur < prv:  # prv < cur
            cur, prv = prv, cur
            b = 1 if b == 2 else 2
        delta = cur - prv
        mid = prv + delta // 2
        if delta % 2 == 0:
            if b == 1: # closer to cur
                cand_range = [mid + 1, 10**9 + 1]
                for r in ranges:
                    s, e = intersection(r, cand_range)
                    if length(s, e) > 0: nranges.append([s, e])
            else: # closer to prv
                cand_range = [0, mid]
                for r in ranges:
                    s, e = intersection(r, cand_range)
                    if length(s, e) > 0: nranges.append([s, e])
            cand_range = [mid, mid + 1]
            for r in ranges:
                s, e = intersection(r, cand_range)
                if length(s, e) > 0: nranges.append([s, e])
        else:
            if b == 1: # closer to cur
                cand_range = [mid + 1, 10**9 + 1]
                for r in ranges:
                    s, e = intersection(r, cand_range)
                    if length(s, e) > 0: nranges.append([s, e])
            else: # closer to prv
                cand_range = [0, mid + 1]
                for r in ranges:
                    s, e = intersection(r, cand_range)
                    if length(s, e) > 0: nranges.append([s, e])
        ranges = nranges
    print(sum(e - s for s, e in ranges))
if __name__ == '__main__':
    T = 1
    for _ in range(T):
        main()
```

## K. Forum

### Solution 1:  sorted list, offline query, process queries backwards

```py
from bisect import bisect_left as lower_bound
from bisect import bisect_right as upper_bound
import math
 
class FenwickTree:
    def __init__(self, x):
        bit = self.bit = list(x)
        size = self.size = len(bit)
        for i in range(size):
            j = i | (i + 1)
            if j < size:
                bit[j] += bit[i]
 
    def update(self, idx, x):
        """updates bit[idx] += x"""
        while idx < self.size:
            self.bit[idx] += x
            idx |= idx + 1
 
    def __call__(self, end):
        """calc sum(bit[:end])"""
        x = 0
        while end:
            x += self.bit[end - 1]
            end &= end - 1
        return x
 
    def find_kth(self, k):
        """Find largest idx such that sum(bit[:idx]) <= k"""
        idx = -1
        for d in reversed(range(self.size.bit_length())):
            right_idx = idx + (1 << d)
            if right_idx < self.size and self.bit[right_idx] <= k:
                idx = right_idx
                k -= self.bit[idx]
        return idx + 1, k
 
class SortedList:
    block_size = 700
 
    def __init__(self, iterable=()):
        self.macro = []
        self.micros = [[]]
        self.micro_size = [0]
        self.fenwick = FenwickTree([0])
        self.size = 0
        for item in iterable:
            self.insert(item)
 
    def insert(self, x):
        i = lower_bound(self.macro, x)
        j = upper_bound(self.micros[i], x)
        self.micros[i].insert(j, x)
        self.size += 1
        self.micro_size[i] += 1
        self.fenwick.update(i, 1)
        if len(self.micros[i]) >= self.block_size:
            self.micros[i:i + 1] = self.micros[i][:self.block_size >> 1], self.micros[i][self.block_size >> 1:]
            self.micro_size[i:i + 1] = self.block_size >> 1, self.block_size >> 1
            self.fenwick = FenwickTree(self.micro_size)
            self.macro.insert(i, self.micros[i + 1][0])
 
    # requires index, so pop(i)
    def pop(self, k=-1):
        i, j = self._find_kth(k)
        self.size -= 1
        self.micro_size[i] -= 1
        self.fenwick.update(i, -1)
        return self.micros[i].pop(j)
 
    def __getitem__(self, k):
        i, j = self._find_kth(k)
        return self.micros[i][j]
 
    def count(self, x):
        return self.upper_bound(x) - self.lower_bound(x)
 
    def __contains__(self, x):
        return self.count(x) > 0
 
    def lower_bound(self, x):
        i = lower_bound(self.macro, x)
        return self.fenwick(i) + lower_bound(self.micros[i], x)
 
    def upper_bound(self, x):
        i = upper_bound(self.macro, x)
        return self.fenwick(i) + upper_bound(self.micros[i], x)
 
    def _find_kth(self, k):
        return self.fenwick.find_kth(k + self.size if k < 0 else k)
 
    def __len__(self):
        return self.size
 
    def __iter__(self):
        return (x for micro in self.micros for x in micro)
 
    def __repr__(self):
        return str(list(self))
 
def main():
    n = int(input())
    sl = SortedList()
    queries = [None] * n
    m = 0
    for i in range(n):
        query = input().split()
        if query[0] == "1":
            queries[i] = (1, i + 1, int(query[1]))
        else:
            queries[i] = (2, int(query[1]), int(query[2]))
            m += 1
    ans = [[] for _ in range(m)]
    remain = Counter()
    for i in reversed(range(n)):
        t, x, v = queries[i]
        if t == 1:
            end = sl.upper_bound((v, math.inf)) - 1
            for j in range(end, -1, -1):
                _, idx = sl[j]
                ans[idx].append(x)
                remain[idx] -= 1
                if remain[idx] == 0:
                    sl.pop(j)    
        else:
            m -= 1
            remain[m] = v
            sl.insert((x, m))
    for resp in ans:
        print(len(resp))
        print(*resp)
 
if __name__ == '__main__':
    T = 1
    for _ in range(T):
        main()
```

## L. Roads

### Solution 1:  tree, combinatorics, counting, dp of bags, O(n^3)

```py
MOD = 998244353
def pair(n):
    return n * (n - 1) // 2
def main():
    n = int(input())
    dp = Counter({(1, 0): 1})
    for _ in range(n - 1):
        ndp = Counter()
        for (len_, plen), cnt in dp.items():
            # TRANSITION 1: ADDING A NEW CITY AT SAME DISTANCE
            if plen > 0:
                ndp[(len_ + 1, plen)] += plen * cnt
                ndp[(len_ + 1, plen)] %= MOD
            # TRANSITION 2: ADDING A NEW CITY AT ONE GREATER DISTANCE
            ndp[(1, len_)] += len_ * cnt * pow(2, pair(len_))
            ndp[(1, len_)] %= MOD
        dp = ndp
    ans = 0
    for (len_, plen), cnt in dp.items():
        ans = (ans + cnt * pow(2, pair(len_), MOD)) % MOD
    print(ans)

if __name__ == '__main__':
    T = 1
    for _ in range(T):
        main()
```

## Main Round

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

##

### Solution 1: 

```py

```