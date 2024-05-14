# Leetcode Weekly Contest 286

## Summary

I struggled a little too much on Q3,  I just did not know the easy way to 
generate a palindrome for integers in order,  I know you can consider just the
first half of the palindrome, since it is reflected.  But I didn't know that if you
look at 100, 101, 102, 103, 104, 105, 106, ... 190, these are 90 palindromes and when you reflect
you get either two depending if odd or even length. 

My struggle on Q4 was to optimize the dynamic programming solution.  I think my solution is about
O(k^3), but not sure on taht. 

## 2215. Find the Difference of Two Arrays

### Solution: set theory, difference between two sets

```py
class Solution:
    def findDifference(self, nums1: List[int], nums2: List[int]) -> List[List[int]]:
        return [set(nums1)-set(nums2), set(nums2)-set(nums1)]
```


## 2216. Minimum Deletions to Make Array Beautiful

```py
class Solution:
    def minDeletion(self, nums: List[int]) -> int:
        n = len(nums)
        i = cnt = 0
        while i < len(nums)-1:
            if nums[i]==nums[i+1]:
                i += 1
                cnt += 1
            else:
                i += 2
        return cnt + ((n-cnt)%2==1)
```

```py
class Solution:
    def minDeletion(self, nums: List[int]) -> int:
        answer = []
        for num in nums:
            if len(answer)%2==0 or answer[-1]!=num:
                answer.append(num)
        return len(nums)-(len(answer)-len(answer)%2)
```

## 2217. Find Palindrome With Fixed Length

```py
class Solution:
    def kthPalindrome(self, queries: List[int], intLength: int) -> List[int]:
        base = 10**((intLength-1)//2)
        result = [base + q - 1 for q in queries]
        answer = [-1]*len(queries)
        for i, pal in enumerate(result):
            if intLength%2==0:
                spal = str(pal) + str(pal)[::-1]
            else:
                spal = str(pal) + str(pal)[:-1][::-1]
            if len(spal)==intLength:
                answer[i] = int(spal)
        return answer
```


## 2218. Maximum Value of K Coins From Piles

### Solution:  recursive dp that TLE

```py
class Solution:
    def maxValueOfCoins(self, piles: List[List[int]], k: int) -> int:
        num_piles = len(piles)
        # too slow it is O(n*k*k)
        def dfs(current_pile, i, num_coins):
            if num_coins == k or current_pile==num_piles: return 0
            if i==len(piles[current_pile]):
                return dfs(current_pile+1,0,num_coins)
            return max(dfs(current_pile+1,0,num_coins), dfs(current_pile, i+1,num_coins+1) + piles[current_pile][i])
        return dfs(0,0,0)
```