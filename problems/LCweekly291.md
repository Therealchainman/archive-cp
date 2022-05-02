# Leetcode Weekly Contest 291

## Summary

## 2259. Remove Digit From Number to Maximize Result

### Solution 1: math + greedy, choose the first time you see digit when if you remove it, it will be replaced by larger digit to immediate left of it. 

```py
class Solution:
    def removeDigit(self, number: str, digit: str) -> str:
        idx = 0
        for i, s in enumerate(number):
            if s==digit:
                idx = i
                if i+1 < len(number) and number[i+1] > digit:
                    break
        return number[:idx] + number[idx+1:]
```

## 2260. Minimum Consecutive Cards to Pick Up

### Solution 1: hash table to store last index of card

```py
class Solution:
    def minimumCardPickup(self, cards: List[int]) -> int:
        last_idx = {}
        best = inf
        for i, c in enumerate(cards):
            if c in last_idx:
                best = min(best, i - last_idx[c] + 1)
            last_idx[c] = i
        return best if best < inf else -1
```

## 2261. K Divisible Elements Subarrays

### Solution 1: hash table to store the subarrays

```py
class Solution:
    def countDistinct(self, nums: List[int], k: int, p: int) -> int:
        cntArrays = 0
        n = len(nums)
        seen = set()
        for i in range(n):
            cntDiv = 0
            subarray = []
            for j in range(i,n):
                cntDiv += (nums[j]%p==0)
                if cntDiv > k: break
                subarray.append(nums[j])
                hash_ = tuple(subarray)
                if hash_ in seen: continue
                cntArrays += 1
                seen.add(hash_)
        return cntArrays
```

## 2262. Total Appeal of A String

### Solution 1: Store current delta, and hash table for last index

For each character, you add it to all the previous substrings consider, and so you will add that appeal delta.  but also you want to increase the appeal for all the substrings that will gain an increase of one appeal.  Which will be basically the number of substrings found based on the last index location of this current character.  So you add that to the delta, because it is the new delta for the appeal that we add at each additional character. 

```py
class Solution:
    def appealSum(self, s: str) -> int:
        last_idx = defaultdict(lambda: -1)
        delta = sum_ = 0
        for i, ch in enumerate(s):
            delta += (i-last_idx[ch])
            sum_ += delta
            last_idx[ch] = i
        return sum_
```