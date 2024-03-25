# Leetcode BiWeekly Contest 122

## 3011. Find if Array Can Be Sorted

### Solution 1:  brute force, sort, bit count

```py
class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        n = len(nums)
        for i in range(1, n):
            for j in range(i, 0, -1):
                if nums[j - 1].bit_count() != nums[j].bit_count(): break
                if nums[j - 1] > nums[j]: nums[j - 1], nums[j] = nums[j], nums[j - 1]
        return nums == sorted(nums)
```

## 3012. Minimize Length of Array Using Operations

### Solution 1:  greedy

```py
class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        def ceil(x, y):
            return (x + y - 1) // y
        freq = Counter(nums)
        min_val = min(nums)
        if any(0 < x % min_val < min_val for x in nums): return 1
        return ceil(freq[min_val], 2)
```

## 3013. Divide an Array Into Subarrays With Minimum Cost II

### Solution 1:  sliding window, max and min heap

```py
class Solution:
    def minimumCost(self, nums: List[int], k: int, dist: int) -> int:
        n = len(nums)
        start = nums[0]
        nums = nums[1:]
        ans = math.inf
        wsum = wcount = 0
        k -= 1
        maxheap, minheap = [], []
        used = [0] * n
        for i in range(n - 1):
            l = i - dist
            if l > 0 and used[l - 1]: # remove outside window
                used[l - 1] = 0
                wsum -= nums[l - 1]
                wcount -= 1
            while wcount < k and minheap: # take from minheap if need more in window
                _, idx = heappop(minheap)
                if idx < l: continue 
                used[idx] = 1
                wsum += nums[idx]
                heappush(maxheap, (-nums[idx], idx))
                wcount += 1
            while maxheap and maxheap[0][1] < l: heappop(maxheap) # remove outsidw window from maxheap
            if wcount < k or nums[i] < -maxheap[0][0]: # add current to maxheap or minheap
                heappush(maxheap, (-nums[i], i))
                wcount += 1
                wsum += nums[i]
                used[i] = 1
            else:
                heappush(minheap, (nums[i], i))
            if wcount > k: # if added to maxheap it may have exceeded k, so need to remove from it
                _, idx = heappop(maxheap)
                if idx < l: continue 
                used[idx] = 0
                wsum -= nums[idx]
                wcount -= 1 
                heappush(minheap, (nums[idx], idx))
            if l >= 0: ans = min(ans, wsum)     
        return ans + start
```