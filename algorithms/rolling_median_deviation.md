# Rolling Median Deviation

## The algorithm in python

This is the algorithm that uses prefix sums and is O(n) time complexity.  For this specific example it is calculating the smallest median deviation over the array, can also find the maximum if you just swap that out.  Or even find the sum, don't know why you'd want that though. 

```py
def RMD(nums, k): # rolling median deviation
    n = len(nums)
    if k == 0: return 0
    if k > n: return math.inf
    psum = 0
    for i in range(k):
        psum += abs(nums[k // 2] - nums[i])
    ans = psum
    med_i = k // 2
    for i in range(k, n):
        delta = nums[med_i + 1] - nums[med_i]
        psum += delta
        psum += (k // 2) * delta
        v = k // 2 - (1 if k % 2 == 0 else 0)
        psum -= v * delta
        med_i += 1
        psum += nums[i] - nums[med_i] # element added into window
        psum -= nums[med_i] - nums[i - k] # element removed from window
        ans = min(ans, psum) 
    return ans
```