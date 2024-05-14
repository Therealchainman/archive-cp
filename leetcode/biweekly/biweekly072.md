# Leetcode Biweekly Contest 72

## 2176. Count Equal and Divisible Pairs in an Array

### Solution: Brute force two for loops

```py
class Solution:
    def countPairs(self, nums: List[int], k: int) -> int:
        cnt = 0
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                cnt += 1 if nums[i]==nums[j] and (i*j)%k==0 else 0
        return cnt
```

## 2177. Find Three Consecutive Integers That Sum to a Given Number

### Solution: math, if divisible by 3 take the surrounding two elements

```py
class Solution:
    def sumOfThree(self, num: int) -> List[int]:
        if num%3!=0: return []
        return [num//3-1, num//3, num//3+1]
```


## 2178. Maximum Split of Positive Even Integers

### Solution: Greedy, find all elements that sum above and remove the single element if possible

```py
class Solution:
    def maximumEvenSplit(self, finalSum: int) -> List[int]:
        """
        if you start from 2, and include all positive unique even integers, at most there could be 100,000.
        This is a good threshold to know cause that means my result can only be that long. 
        """
        ans = []
        sum_ = 0
        for i in range(2, finalSum+1, 2):
            ans.append(i)
            sum_ += i
            if sum_ >= finalSum: break
        if sum_ == finalSum: return ans
        if (sum_ - finalSum)%2==0:
            ans.remove(sum_ - finalSum)
            return ans
        return []
```

## 2179. Count Good Triplets in an Array

### Solution: IDK

```py

```