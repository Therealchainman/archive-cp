# Leetcode Weekly Contest 355

## 2788. Split Strings by Separator

### Solution 1: 

```py
class Solution:
    def splitWordsBySeparator(self, words: List[str], separator: str) -> List[str]:
        res = []
        for word in words:
            res.extend(filter(None, word.split(separator)))
        return res
```

## 2789. Largest Element in an Array after Merge Operations

### Solution 1: 

```py
class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        res = prv = 0
        for num in reversed(nums):
            if num <= prv:
                prv += num
            else:
                prv = num
            res = max(res, prv)
        return res
```

## 2790. Maximum Number of Groups With Increasing Length

### Solution 1:  sort + greedy

![images](images/number_groups_with_increasing_length.png)

```py
class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        cur = res = 0
        for usage in sorted(usageLimits):
            cur += usage
            if cur > res:
                res += 1
                cur -= res
        return res
```

## 2791. Count Paths That Can Form a Palindrome in a Tree

### Solution 1:  dynamic programming + modulus 2 addition + xor + lowest common ancestor

![image](images/count_number_palindromes_1.png)
![image](images/count_number_palindromes_2.png)
![image](images/count_number_palindromes_3.png)

```py
class Solution:
    def countPalindromePaths(self, parent: List[int], s: str) -> int:
        n = len(parent)
        @cache
        def mask(node):
            i = ord(s[node]) - ord('a')
            return mask(parent[node]) ^ (1 << i) if node else 0
        count = Counter()
        res = 0
        for i in range(n):
            cur_mask = mask(i)
            res += count[cur_mask] + sum(count[cur_mask^ (1 << j)] for j in range(26))
            count[cur_mask] += 1
        return res
```