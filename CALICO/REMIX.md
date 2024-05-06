# REMIX

## Problem 1: Ice Cream Bars!

### Solution 1: binary search

Need a better solution to solve when N = 10^10_000

```py
def main():
    N = int(input())
    eval = lambda n: n * (n + 1) / 2
    l, r = 0, 10 ** 15
    while l < r:
        m = (l + r + 1) >> 1
        if eval(m) <= N:
            l = m
        else:
            r = m - 1
    print(l)

if __name__ == "__main__":
    T = int(input())
    for _ in range(T):
        main()
```

## Problem 6: Fractals Against Programmability

### Solution 1:  recursion, grid, split into bounding boxes

```cpp

```

## Problem 8: Bay Area’s Revolutionary Train

### Solution 1:  sortedlist, modular arithmetic, fenwick tree, binary search

```py
from collections import defaultdict

from bisect import bisect_left as lower_bound
from bisect import bisect_right as upper_bound

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
    N, M, K = map(int, input().split())
    sources = list(map(lambda x: int(x) - 1, input().split()))
    targets = list(map(lambda x: int(x) - 1, input().split()))
    station = defaultdict(list)
    for i in reversed(range(N)):
        station[sources[i]].append(i)
    waiting = SortedList(sources)
    dropoff = SortedList() # (distance at which drop off)
    dist = load = 0
    while waiting or dropoff:
        next_station = M + 1
        if load < K and waiting:
            idx = waiting.lower_bound(dist % M)
            if idx == len(waiting): idx = 0
            next_station = waiting[idx]
        if dropoff:
            idx = dropoff.lower_bound(dist % M)
            if idx == len(dropoff): idx = 0
            cand_station = dropoff[idx]
            if next_station < dist % M: 
                if cand_station < dist % M: # both on left side, so take minimum
                    next_station = min(next_station, cand_station)
                else: # next_station on left, cand_station on right
                    next_station = cand_station
            else:
                if cand_station >= dist % M:
                    next_station = min(next_station, cand_station) # both on right side, so take minimum
                if next_station == M + 1: # no one to pick up or drop off
                    next_station = cand_station
        # update dist to get to next_station
        if next_station >= dist % M: 
            delta = next_station - dist % M
        else:
            delta = M - dist % M + next_station
        dist += delta
        idx = dropoff.lower_bound(dist % M)
        if idx < len(dropoff) and dropoff[idx] == dist % M:
            dropoff.pop(idx)
            load -= 1
        else:
            idx = waiting.lower_bound(dist % M)
            load += 1
            waiting.pop(idx)
            dst = targets[station[next_station].pop()]
            dropoff.insert(dst)
    print(dist)

if __name__ == "__main__":
    T = int(input())
    for _ in range(T):
        main()
```

## Problem 3: Bay Area’s Railway Traversal

### Solution 1:  longest difference in modular arithmetic

```py
def main():
    N, M = map(int, input().split())
    sources = list(map(int, input().split()))
    targets = list(map(int, input().split()))
    ans = 0
    for src, dst in zip(sources, targets):
        if dst >= src: ans = max(ans, dst - src)
        else: ans = max(ans, M - src + dst)
    print(ans)

if __name__ == "__main__":
    T = int(input())
    for _ in range(T):
        main()
```

## Problem X: ________ DOLLARS

### Solution 1: 

```py

```

## Problem 11: hollup…Let him cook

### Solution 1: 

```py
def main():
    while True:
        try:
            D = int(input())
            t = input()
            if t == "INCREMENT": print(D + 1)
            elif t == "DECREMENT": print(D - 1)
            else: print(D)
        except:
            break

if __name__ == "__main__":
    main()
```

## Problem 3: @everyone

### Solution 1:  sorting, offline queries, queue, difference array

```py
def main():
    Q, N, M = map(int, input().split())
    pings = [[] for _ in range(N)]
    roles = [[] for _ in range(N)]
    ans = [0] * M
    for i in range(Q):
        action = input().split()
        if action[0] == "A":
            r, u = map(int, action[1:])
            r -= 1; u -= 1
            roles[r].append((i, u))
        elif action[0] == "R":
            r, u = map(int, action[1:])
            r -= 1; u -= 1
            roles[r].append((i, u))
        else:
            r = int(action[1])
            r -= 1
            pings[r].append(i)
    for r in range(N):
        i = 0
        n = len(pings[r])
        marked = set()
        for idx, p in enumerate(pings[r]):
            while i < len(roles[r]) and roles[r][i][0] < p:
                u = roles[r][i][1]
                if u in marked:
                    ans[u] -= n - idx
                    marked.remove(u)
                else:
                    ans[u] += n - idx
                    marked.add(u)
                i += 1
    print(*ans)

if __name__ == "__main__":
    T = int(input())
    for _ in range(T):
        main()
```

## Problem 11: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!0

### Solution 1: 

```py

```

## 

### Solution 1: 

```py

```

