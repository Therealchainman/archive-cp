# Rolling Median Deviation

## The algorithm in python

This is the algorithm that uses prefix sums and is O(n) time complexity.  For this specific example it is calculating the smallest median deviation over the array, can also find the maximum if you just swap that out.  Or even find the sum, don't know why you'd want that though.

Note this is for a median when you take subarray of size k

```py
n = len(nums)
nums.sort()
psum = list(accumulate(nums))
def sum_(i, j):
    return psum[j] - (psum[i - 1] if i > 0 else 0)
def deviation(i, j, mid):
    lsum = (mid - i + 1) * nums[mid] - sum_(i, mid)
    rsum = sum_(mid, j) - (j - mid + 1) * nums[mid]
    return lsum + rsum
def RMD(nums, k): # rolling median deviation
    ans = math.inf
    l = 0
    for r in range(k - 1, n):
        mid = (l + r) >> 1
        ans = min(ans, deviation(l, r, mid))
        if k % 2 == 0:
            ans = min(ans, deviation(l, r, mid + 1))
        l += 1
    return ans
```

This is useful, because when you want to calculate the best value to set all elements to in a fixed sized subarray, the median is the best option, can be proved because if you move away from the median, it will increase more elements than those it decreases. 