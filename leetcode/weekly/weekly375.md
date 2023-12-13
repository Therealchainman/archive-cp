# Leetcode Weekly Contest 375

## Count Tested Devices After Test Operations

### Solution 1:  prefix sum

```py
class Solution:
    def countTestedDevices(self, arr: List[int]) -> int:
        n = len(arr)
        psum = res = 0
        for v in arr:
            if v - psum > 0:
                psum += 1
                res += 1
        return res
```

## Double Modular Exponentiation

### Solution 1:  power function with modulus

```py
class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        n = len(variables)
        ans = []
        for i, (a, b, c, m) in enumerate(variables):
            cur = pow(pow(a, b, 10), c, m)
            if cur == target: ans.append(i)
        return ans
```

## Count Subarrays Where Max Element Appears at Least K Times

### Solution 1:  sliding window

```py
class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        ans = left = count = 0
        target = max(nums)
        for right in range(n):
            if nums[right] == target: count += 1
            while count > k:
                if nums[left] == target: count -= 1
                left += 1
            if count == k:
                while nums[left] != target:
                    left += 1
                ans += left + 1
        return ans
```

## Count the Number of Good Partitions

### Solution 1:  track the last time each element appears, create blocks, and count the number of ways to choose the blocks, 2^(num_blocks - 1)

```py
class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        n = len(nums)
        mod = int(1e9) + 7
        last = {}
        for i, num in enumerate(nums):
            last[num] = i
        right = num_blocks = 0
        for i, num in enumerate(nums):
            right = max(right, last[num])
            if i == right: num_blocks += 1
        return pow(2, num_blocks - 1, mod)        
```

