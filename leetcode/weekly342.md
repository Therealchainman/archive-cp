# Leetcode Weekly Contest 342

## 2651. Calculate Delayed Arrival Time

### Solution 1:  modulus

```py
class Solution:
    def findDelayedArrivalTime(self, arrivalTime: int, delayedTime: int) -> int:
        return (arrivalTime + delayedTime)%24
```

## 2652. Sum Multiples

### Solution 1:  sum + loop + modulus

```py
class Solution:
    def sumOfMultiples(self, n: int) -> int:
        return sum(i for i in range(1, n + 1) if i%3 == 0 or i%5 == 0 or i%7 == 0)
```

## 2653. Sliding Subarray Beauty

### Solution 1:  constant size sliding window + sortedlist

```py
from sortedcontainers import SortedList
class Solution:
    def getSubarrayBeauty(self, nums: List[int], k: int, x: int) -> List[int]:
        n = len(nums)
        window = SortedList()
        res = []
        for i in range(n):
            window.add(nums[i])
            if len(window) == k:
                res.append(min(0, window[x - 1]))
                window.remove(nums[i - k + 1])
        return res
```

### Solution 2: constant size sliding window + frequency of negative integers + O(50n)

```py
class Solution:
    def getSubarrayBeauty(self, nums: List[int], k: int, x: int) -> List[int]:
        n = len(nums)
        freq, res = [0]*50, []
        for i in range(n):
            if nums[i] < 0: freq[nums[i] + 50] += 1
            if i >= k and nums[i - k] < 0: freq[nums[i - k] + 50] -= 1
            if i >= k - 1:
                cnt = 0
                for j in range(50):
                    cnt += freq[j]
                    if cnt >= x:
                        res.append(j - 50)
                        break
                if cnt < x: res.append(0)
        return res
```

## 2654. Minimum Number of Operations to Make All Array Elements Equal to 1

### Solution 1:  smallest gcd subarray equal to 1 + O(n^2)

```py
class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        ones = nums.count(1)
        if ones: return n - ones
        res = math.inf
        for i in range(n):
            gcd_prefix = nums[i]
            for j in range(i + 1, n):
                gcd_prefix = math.gcd(gcd_prefix, nums[j])
                if gcd_prefix == 1:
                    res = min(res, j - i)
                    break
        return res + n - 1 if res != math.inf else -1
```