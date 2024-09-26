# Leetcode Biweekly Contest 105

## 2706. Buy Two Chocolates

### Solution 1:  max

```py
class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        prices.sort()
        return money - prices[0] - prices[1] if money - prices[0] - prices[1] >= 0 else money
```

## 2707. Extra Characters in a String

### Solution 1:  dynamic programming + O(n^3)

dp[i] = min extra characters for the substring s[0:i] or range [0, i)

```py
class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        dictionary = set(dictionary)
        dp = [0] + [math.inf]*n
        for j in range(n + 1):
            for i in range(j):
                if s[i:j] in dictionary:
                    dp[j] = min(dp[j], dp[i])
                else:
                    dp[j] = min(dp[j], dp[i]+j-i)
        return dp[-1]
```

```cpp
class Solution {
public:
    int minExtraChar(string s, vector<string>& dictionary) {
        unordered_set<string> seen(dictionary.begin(), dictionary.end());
        int N = s.size();
        vector<int> dp(N + 1, 0);
        for (int i = 0; i < N; i++) {
            dp[i + 1] = dp[i] + 1;
            string cur = "";
            for (int j = i; j >= 0; j--) {
                cur = s[j] + cur;
                if (seen.count(cur)) {
                    dp[i + 1] = min(dp[i + 1], dp[i - cur.size() + 1]);
                }
            }
        }
        return dp[N];
    }
};
```

## 2708. Maximum Strength of a Group

### Solution 1:  sort + O(nlogn)

take even number of negative integers because they cancel to positive, take all positive integers.
only reason to take odd number of negative integers is if there is a single element in array and it is negative. 

```py
class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1: return nums[0]
        nums.sort()
        pos_i = bisect.bisect_right(nums, 0)
        neg_i = bisect.bisect_right(nums, -1)
        res = 0
        if neg_i & 1:
            neg_i -= 1
        for i in range(neg_i):
            if res == 0: res = 1
            res *= nums[i]
        for i in range(pos_i, n):
            if res == 0: res = 1
            res *= nums[i]
        return res
```

## 2709. Greatest Common Divisor Traversal

### Solution 1:  union find + prime sieve

If there is a single connected component, than there is always a traversal sequence through the graph to get all pairs be able to reach each other.  All elements that share a prime factor should be grouped into same connected component, because you can also get from one to the other with gcd > 1.

Therefore, simply merge the elements with common prime factors together, and finally check if they can form a whole large set.

can use the first array to track what was the first index of element that had that prime integer, only need to store one representative integer for each prime, so everything gets merged into that representative integer.

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
    
    @property
    def root_count(self):
        return sum(node == self.find(node) for node in range(len(self.parent)))

    def __repr__(self) -> str:
        return f'parents: {[(i, parent) for i, parent in enumerate(self.parent)]}, sizes: {self.size}'
class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        n = len(nums)
        mx = max(nums)
        def prime_sieve(lim):
            sieve,primes = [set() for _ in range(lim)], []
            for integer in range(2,lim):
                if not len(sieve[integer]):
                    primes.append(integer)
                    for possibly_divisible_integer in range(integer,lim,integer):
                        current_integer = possibly_divisible_integer
                        while not current_integer%integer:
                            sieve[possibly_divisible_integer].add(integer)
                            current_integer //= integer
            return sieve
        sieve = prime_sieve(mx + 1)
        dsu = UnionFind(n)
        first = [-1] * (mx + 1)
        for i, num in enumerate(nums):
            for prime in sieve[num]:
                if first[prime] != -1: dsu.union(first[prime], i)
                else: first[prime] = i
        return dsu.root_count == 1
```

### Solution 2:  online prime factorization + union find

This is slower than method above most likely and leetcode confirmed that statement given their test cases.

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
    
    @property
    def root_count(self):
        return sum(node == self.find(node) for node in range(len(self.parent)))

    def __repr__(self) -> str:
        return f'parents: {[(i, parent) for i, parent in enumerate(self.parent)]}, sizes: {self.size}'
class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        n = len(nums)
        last = [-1] * (1 + max(nums))
        dsu = UnionFind(n)
        for i, num in enumerate(nums):
            for prime in range(2, num):
                if prime*prime > num: break
                if num % prime != 0: continue
                if last[prime] != -1: dsu.union(last[prime], i)
                else: last[prime] = i
                while num % prime == 0: num //= prime
            if num > 1:
                if last[num] != -1: dsu.union(last[num], i)
                else: last[num] = i
        return dsu.root_count == 1
```