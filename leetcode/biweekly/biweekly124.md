# Leetcode BiWeekly Contest 124

## Apply Operations to Make String Empty

### Solution 1:  counter, reverse

You just need to take the characters from s that achieve the maximum frequency and in the reverse order. 

```py
class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        ans = []
        counts = Counter(s)
        mx = max(counts.values())
        for ch in reversed(s):
            if counts[ch] == mx: 
                counts[ch] = 0
                ans.append(ch)
        return "".join(reversed(ans))
```

## Maximum Number of Operations With the Same Score II

### Solution 1:  dynamic programming

You can use interval dynamic programming to solve this problem.  Want to think about an iterative approach post contest.

```py
class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        n = len(nums)
        @cache
        def dp(i, j, target):
            if j - i + 1 < 2: return 0
            ans = 0
            if i + 1 < n and nums[i] + nums[i + 1] == target:
                ans = max(ans, dp(i + 2, j, target) + 1)
            if j - 1 >= 0 and nums[j - 1] + nums[j] == target:
                ans = max(ans, dp(i, j - 2, target) + 1)
            if nums[i] + nums[j] == target:
                ans = max(ans, dp(i + 1, j - 1, target) + 1)
            return ans
        res = max(dp(2, n - 1, nums[0] + nums[1]), dp(1, n - 2, nums[0] + nums[-1]), dp(0, n - 3, nums[-1] + nums[-2]))
        return res + 1
```

## Maximize Consecutive Elements in an Array After Modification

### Solution 1:  sort, greedy

You are allowed to increment each integer by at most 1. 

Given an array [1,2,3,4,5]  It is easy right to calculate the longest consecutive subarray
What about array [1,2,3,4,7,8] It is obvious that 7 can never connect to 4, because the difference is greater than 2.
what about array [1,2,3,4,6,7] You have a different of 2 between 4 and 6.  And you have created 1,2,3,4 which is length of 4, you can increment them all by 1, then it will match up with 6, 7
What about array [1,1,3,4,6,7] Between the 1 and 3, it realizes it can take 3 right, cause 1,2,3
Then you get 1,2,3,4.  But 2,3,4,5,6 is not possible.  The reason is that you increment the 1 at the beginning. 
What about array [1,1,1,3,4,6,7] This one added in an extra 1 that is going to cause problems. cause it will be smaller than nxt by 2,  so just skip it. 



Observation 1:
Define a contiguous block to be a subarray where the difference between adjacent elements is less than 2.

Observation 2:
You can never connect to adjacent blocks if the difference between last element of left block and first element of right block is greater than 2. 

Observation 3:
You need to track the length of the longest normal consecutive elements, that is without any operation.  Because if you split between two blocks where difference = 2. You can take the longest normal sequence and add that to the start of this contiguous blocks. The reason is you can just increment every single value in the normal sequence to get it to start this new block.  So you can glue together previous to this block.

Observation 4:
[1,1,1] = [1,2]
You can ignore more than 2 of any elements.  It will be useless, it can be built in if you just skip any element that is less than what you are looking for even with an operation so if it is 2 less you can skip.


Observation x:
There are a few edge cases that will be difficult to handle
[1,1,3,4,4,6], but in this example you can see that you can chain together blocks b1, b2, b3.
[1,2,4,5,6,8,9] In this example you can chain together b1, b2 or b2, b3, but not all of them.
Let's change the definition of a contiguous block however.  
Preprocess the array above you'd get [1,2,3,4,5,6] so it is actually all a single block.  I think you should increment elements that are equal to their previous element.  

```py
class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        ans = cur = nxt = norm = 0
        for i in range(n):
            if nums[i] + 1 < nxt: continue
            if nums[i] + 1 == nxt or nums[i] == nxt: #extend
                cur += 1
                nxt += 1
            elif i > 0 and nums[i] - nums[i - 1] == 2:
                cur = norm + 1
                nxt = nums[i] + 1
            else:
                cur = 1
                nxt = nums[i] + 1
            if i > 0 and nums[i] == nums[i - 1] + 1: norm += 1
            else: norm = 1
            ans = max(ans, cur)
        return ans
```

