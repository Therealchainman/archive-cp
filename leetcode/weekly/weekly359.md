# Leetcode Weekly Contest 359

## 2828. Check if a String Is an Acronym of Words

### Solution 1:  all + zip_longest + loop

```py
class Solution:
    def isAcronym(self, words: List[str], s: str) -> bool:
        return all(c1 == c2 for c1, c2 in zip_longest(map(lambda x: x[0], words), s, fillvalue = "#"))
```

## 2829. Determine the Minimum Sum of a k-avoiding Array

### Solution 1:  set + greedy

add the smallest numbers first

```py
class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        arr = set()
        i = 1
        while len(arr) != n:
            if k - i not in arr:
                arr.add(i)
            i += 1
        return sum(arr)
```

## 2830. Maximize the Profit as the Salesman

### Solution 1:  dynamic programming + interval + O(n)

given a start, end, gold, 
you already calculated the maximum up to the start house since start is smaller than end
So just take the value from before start, so start-1 and add gold to that

```py
class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        A = defaultdict(list)
        for start, end, gold in offers:
            A[end + 1].append((start + 1, gold))
        dp = [0] * (n + 1)
        for i in range(1, n + 1):
            dp[i] = dp[i - 1]
            for start, gold in A[i]:
                dp[i] = max(dp[i], dp[start - 1] + gold)
        return dp[-1]
            
```

## 2831. Find the Longest Equal Subarray

### Solution 1:  sliding window over each num + sort indices for each num

So this is like a bucketized sliding window algorithm. Each number can be viewed as a bucket and an independent sliding window algorithm runs on each bucket. 
Within each bucket you need to have the indices sorted for where that num is in nums array. This allows you to compute the number that you delete over any interval and also know the size of the num you can get with that number of deletions. 

suppose num = 2
and indices = [1,4,6,9]
so 
1xx1x1xx1, where x can be any other number 
left = 0
and right = 3
These are the pointers for the indices array
that means you are currently considering right - left + 1 = 4 elements of 1, but you need to delete some to get them to be adjacent to each other
you can count the x above which is 5 deletions. 
and you can know it because the length of from the indices[right] - indices[left] + 1 is the size of the entire subarray in consideration.  so it is 9 here, so you just take the 10 and subtract the 4 to get 5.  

So you just need to maximize the right - left + 1, for when it doesn't exceed the k deletions. 

```py
class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        n = len(nums)
        indices = [[] for _ in range(n + 1)]
        for i, num in enumerate(nums):
            indices[num].append(i)
        res = 0
        for index in indices:
            if not index: continue
            left = 0
            for right in range(len(index)):
                while (index[right] - index[left]) - (right - left) > k: left += 1
                res = max(res, right - left + 1)
        return res
```

