# Leetcode Weekly Contest 363

## 2859. Sum of Values at Indices With K Set Bits

### Solution 1:  loop + bit manipulation

```py
class Solution:
    def sumIndicesWithKSetBits(self, nums: List[int], k: int) -> int:
        return sum(num for i, num in enumerate(nums) if i.bit_count() == k)
```

## 2860. Happy Students

### Solution 1:  sort + greedy

```py
class Solution:
    def countWays(self, nums: List[int]) -> int:
        n = len(nums)
        nums.extend([-math.inf, math.inf])
        nums.sort()
        return sum(1 for i in range(n + 1) if i < nums[i + 1] and i > nums[i])
```

## 2861. Maximum Number of Alloys

### Solution 1: binary search + greedy

```py
class Solution:
    def maxNumberOfAlloys(self, n: int, k: int, budget: int, composition: List[List[int]], stock: List[int], cost: List[int]) -> int:
        def possible(target):
            min_cost = math.inf
            for i in range(k):
                tcost = 0
                for j in range(n):
                    required = composition[i][j] * target
                    if required > stock[j]:
                        tcost += (required - stock[j]) * cost[j]
                min_cost = min(min_cost, tcost)
            return min_cost <= budget
        left, right = 0, 10**16
        while left < right:
            mid = (left + right + 1) >> 1
            if possible(mid):
                left = mid
            else:
                right = mid - 1
        return left
```

## 2862. Maximum Element-Sum of a Complete Subset of Indices

### Solution 1:  prime sieve + math + number theory + hash + counter

The idea is to store all the states which represnt all the prime factors that have an odd power. Then we can use a counter to store the sum of the elements that have the same state. Because anytime they all have same set of prime factors with odd power they can belong to the same set.  

```py
class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        n = len(nums)
        def prime_sieve(upper_bound):
            prime_factorizations = [[] for _ in range(upper_bound + 1)]
            for i in range(2, upper_bound + 1):
                if len(prime_factorizations[i]) > 0: continue # not a prime
                for j in range(i, upper_bound + 1, i):
                    prime_factorizations[j].append(i)
            return prime_factorizations
        pf = prime_sieve(n + 1)
        def search(factors, val):
            p = [0] * len(factors)
            for i, f in enumerate(factors):
                while val % f == 0:
                    p[i] += 1
                    val //= f
            return p
        sums = Counter()
        for i, num in enumerate(nums, start = 1):
            factors = pf[i]
            pcount = search(factors, i)
            state = tuple((f for f, p in zip(factors, pcount) if p & 1))
            sums[state] += num
        return max(sums.values())
```

