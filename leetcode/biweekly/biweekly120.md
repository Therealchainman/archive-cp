# Leetcode Weekly Contest 119

## 2971. Find Polygon With the Largest Perimeter

### Solution 1:  sort, reverse iteration, prefix sum

```py
class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        n = len(nums)
        nums.sort()
        psum = sum(nums)
        for i in range(n - 1, 1, -1):
            if psum > 2 * nums[i]: return psum
            psum -= nums[i]
        return -1
```

## 

### Solution 1: 

```py

```

## 

### Solution 1:  

```py

```