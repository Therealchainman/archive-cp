# Leetcode Weekly Contest 340

## 2614. Prime In Diagonal

### Solution 1:  matrix + prime + math + O(sqrt(n)) primality check

```py
class Solution:
    def diagonalPrime(self, nums: List[List[int]]) -> int:
        n = len(nums)
        memo = {}
        def is_prime(x: int) -> bool:
            if x in memo: return memo[x]
            if x < 2: return False
            for i in range(2, int(math.sqrt(x)) + 1):
                if x % i == 0: return False
            return True
        res = 0
        for i in range(n):
            if is_prime(nums[i][i]):
                res = max(res, nums[i][i])
            if is_prime(nums[i][~i]):
                res = max(res, nums[i][~i])
        return res
```

### Solution 2:  Sieve of Eratosthenes + precompute primality

```py

```

## 2615. Sum of Distances

### Solution 1:  prefix + suffix sums and counts + line sweep + hash table

```py
class Solution:
    def distance(self, nums: List[int]) -> List[int]:
        n = len(nums)
        last_index = Counter()
        suffix = Counter()
        suffix_cnt = Counter()
        prefix, pcnter = Counter(), Counter()
        for i, num in enumerate(nums):
            suffix[num] += i
            suffix_cnt[num] += 1
        ans = [0]*n
        for i, num in enumerate(nums):
            delta = i - last_index[num]
            suffix[num] -= delta*suffix_cnt[num]
            prefix[num] += delta*pcnter[num]
            ans[i] = prefix[num] + suffix[num]
            suffix_cnt[num] -= 1
            pcnter[num] += 1
            last_index[num] = i
        return ans
```

## 2616. Minimize the Maximum Difference of Pairs

### Solution 1:  greedy binary search

count every other greedily to check if it has enough pais where it is less than or equal to target.  But just know you have to move iterator two forward, so there is no overlap

```py
class Solution:
    def minimizeMax(self, nums: List[int], p: int) -> int:
        if p == 0: return 0
        n = len(nums)
        nums.sort()
        def possible(target):
            cnt = 0
            i = 1
            while i < n:
                if nums[i] - nums[i - 1] <= target:
                    cnt += 1
                    i += 1
                i += 1
            return cnt >= p
        left, right = 0, nums[-1] - nums[0]
        while left < right:
            mid = (left + right) >> 1
            if not possible(mid):
                left = mid + 1
            else:
                right = mid
        return left
```

## 2617. Minimum Number of Visited Cells in a Grid

### Solution 1:  bfs with boundary

```py

```

### Solution 2:  sortedlist

```py

```