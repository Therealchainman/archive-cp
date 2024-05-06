# Atcoder Beginner Contest 349

##

### Solution 1: 

```py

```

## F - Subsequence LCM 

### Solution 1:  number theory, prime factorization, dynamic programming, bitmasks, counting

```py
MOD = 998244353

def prime_factors(num):
    counts = Counter()
    for p in range(2, num):
        if p * p > num: break 
        if num % p != 0: continue 
        while num % p == 0: 
            counts[p] += 1
            num //= p
    if num > 1: counts[num] += 1
    return counts

def main():
    N, M = map(int, input().split())
    arr = list(filter(lambda x: M % x == 0, map(int, input().split())))
    pfcount = prime_factors(M)
    bits = list(sorted(pfcount))
    bvals = [pow(bit, pfcount[bit]) for bit in bits]
    k = len(bits)
    mcounts = [0] * (1 << k)
    for num in arr:
        mask = 0
        for i in range(k):
            if num % bvals[i] == 0:
                mask |= 1 << i
        mcounts[mask] += 1
    dp = [[0] * (1 << k) for _ in range(1 << k)] # dp[i][mask] 
    dp[0][0] = 1
    for i in range(1, 1 << k):
        ways = max(0, pow(2, mcounts[i], MOD) - 1) # number of ways to take it
        # 1 way to not take it
        for mask in range(1 << k):
            dp[i][mask] = (dp[i][mask] + dp[i - 1][mask]) % MOD # don't take 
            dp[i][mask | i] = (dp[i][mask | i] + dp[i - 1][mask] * ways) % MOD # take it
    ans = (dp[-1][-1] * pow(2, mcounts[0], MOD)) % MOD
    if M == 1: ans = (ans - 1) % MOD # subtract the empty set
    print(ans)

if __name__ == '__main__':
    main()
```

## G - Palindrome Construction 

### Solution 1:  manacher, greedy, dsu

This isn't completely the answer, but it was good practice to get this solution.  I couldn't figure out how to get the lexicographically smallest output.

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
    def is_same_connected_components(self, i: int, j: int) -> bool:
        return self.find(i) == self.find(j)
    def size_(self, i):
        return self.size[self.find(i)]
    def __repr__(self) -> str:
        return f'parents: {[(i, parent) for i, parent in enumerate(self.parent)]}, sizes: {self.size}'

def main():
    n = int(input())
    arr = list(map(lambda x: int(x) + 1, input().split()))
    i = j = 0
    valid = True
    dsu = UnionFind(n)
    while i < n: 
        while i - j >= 0 and i + j < n and arr[i] - j > 0:
            dsu.union(i - j, i + j)
            j += 1
        k = 1
        while i - k >= 0 and k + arr[i - k] < j and valid:
            if arr[i - k] != arr[i + k]: valid = False 
            k += 1
        i += k
        j -= k
    for i in range(n):
        if i - arr[i] < 0 or i + arr[i]  >= n: continue
        u, v = dsu.find(i - arr[i]), dsu.find(i + arr[i])
        if u == v: 
            valid = False
            break
    if not valid: return print("No")
    ans = [1] * n
    values = {0: 1}
    for i in range(n):
        root = dsu.find(i)
        if root not in values:
            values[root] = len(values) + 1
        ans[i] = values[root]
    print("Yes")
    print(*ans)

if __name__ == "__main__":
    main()
```