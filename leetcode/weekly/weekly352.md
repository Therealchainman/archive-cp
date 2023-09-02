# Leetcode Weekly Contest 352

## 2760. Longest Even Odd Subarray With Threshold

### Solution 1:  sliding window

```py
class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        left = res = 0
        while left < n:
            while left < n and nums[left] & 1:
                left += 1
            if left < n and nums[left] > threshold:
                left += 1
                continue
            right = left
            while right + 1 < n and nums[right] % 2 != nums[right + 1] % 2 and nums[right + 1] <= threshold:
                right += 1
            if left < n:
                res = max(res, right - left + 1)
            left = right + 1
        return res
```

## 2761. Prime Pairs With Target Sum

### Solution 1:  prime sieve

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

class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        primes = prime_sieve(n + 1)
        res = []
        for x in range(2, n):
            y = n - x
            if x > y: break
            if primes[x] and primes[y]:
                res.append([x, y])
        return res
```

## 2762. Continuous Subarrays

### Solution 1: sliding window + monotonic deque

```py
class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        res = left = 0
        min_stack, max_stack = deque(), deque()
        for right in range(n):
            while min_stack and nums[right] <= nums[min_stack[-1]]:
                min_stack.pop()
            while max_stack and nums[right] >= nums[max_stack[-1]]:
                max_stack.pop()
            min_stack.append(right)
            max_stack.append(right)
            while abs(nums[max_stack[0]] - nums[min_stack[0]]) > 2:
                left += 1
                while min_stack and min_stack[0] < left:
                    min_stack.popleft()
                while max_stack and max_stack[0] < left:
                    max_stack.popleft()
            delta = right - left + 1
            res += delta
        return res
```

## 2763. Sum of Imbalance Numbers of All Subarrays

### Solution 1:  hash table + track imbalance based on conditions

```py
class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        res = 0
        n = len(nums)
        for i in range(n):
            seen = [0] * (n + 2)
            seen[nums[i]] = 1
            min_val = max_val = nums[i]
            imbalance = 0
            for j in range(i + 1, n):
                prv, val, nxt = nums[j] - 1, nums[j], nums[j] + 1
                if nums[j] > max_val and not seen[prv]: imbalance += 1
                if nums[j] < min_val and not seen[nxt]: imbalance += 1
                if min_val < val < max_val and seen[prv] and seen[nxt] and not seen[val]: imbalance -= 1
                if min_val < val < max_val and not seen[prv] and not seen[nxt] and not seen[val]: imbalance += 1
                min_val = min(min_val, nums[j])
                max_val = max(max_val, nums[j])
                seen[val] = 1
                res += imbalance
        return res
```