# Leetcode Weekly Contest 364

## 2864. Maximum Odd Binary Number

### Solution 1: greedy + place 1 at least signficant bit position

```py
class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        n = len(s)
        ones = s.count("1") - 1
        res = [0] * n
        res[-1] = 1
        for i in range(ones):
            res[i] = 1
        return "".join(map(str, res))
```

## 2866. Beautiful Towers II

### Solution 1: prefix and suffix sum + max heaps

Basically consider each index in prefix and suffix to be the peak of the mountain, then the suffix and prefix are the largest sum when the peak is at that index.

```py
class Solution:
    def maximumSumOfHeights(self, H: List[int]) -> int:
        n = len(H)
        heap = []
        res = 0
        psum = [0] * (n + 1)
        for i in range(n):
            cur = lost = 0
            while heap and abs(heap[0][0]) > H[i]:
                v, c = heappop(heap)
                v = abs(v)
                delta = v - H[i]
                lost += delta * c
                cur += c
            heappush(heap, (-H[i], cur + 1))
            psum[i + 1] = psum[i] + H[i] - lost
        heap = []
        ssum = [0] * (n + 1)
        for i in reversed(range(n)):
            cur = lost = 0
            while heap and abs(heap[0][0]) > H[i]:
                v, c = heappop(heap)
                v = abs(v)
                delta = v - H[i]
                lost += delta * c
                cur += c
            heappush(heap, (-H[i], cur + 1))
            ssum[i] = ssum[i + 1] + H[i] - lost
        return max(p + s for p, s in zip(psum, ssum))

```

## 2867. Count Valid Paths in a Tree

### Solution 1: union find + prime sieve

Merge each composite connected component with an adjacent prime number, and increase the size of the neighbor nodes for that prime node, because any other adjacent composite numbers can go through it to have paths with everything attached to it. 

```py
def prime_sieve(lim):
    primes = [1] * lim
    primes[0] = primes[1] = 0
    p = 2
    while p * p <= lim:
        if primes[p]:
            for i in range(p * p, lim, p):
                primes[i] = 0
        p += 1
    return primes

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
    def size_(self, i):
        return self.size[self.find(i)]
    def __repr__(self) -> str:
        return f'parents: {[(i, parent) for i, parent in enumerate(self.parent)]}, sizes: {self.size}'

class Solution:
    def countPaths(self, n: int, edges: List[List[int]]) -> int:
        ps = prime_sieve(n + 1)
        dsu = UnionFind(n + 1)
        for u, v in edges:
            if ps[u] == ps[v] == 0: dsu.union(u, v) # union composite nodes
        res = 0
        count = [1] * (n + 1)
        for u, v in edges:
            if ps[u] ^ ps[v]: # prime and composite node
                if not ps[u]: u, v = v, u
                res += count[u] * dsu.size_(v)
                count[u] += dsu.size_(v)
        return res
```

