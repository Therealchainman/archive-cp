# Leetcode Weekly Contest 115

## 2899. Last Visited Integers

### Solution 1:  stack

```py
class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        result, stack = [], []
        k = 0
        for word in words:
            if word == "prev":
                k += 1
                result.append(-1 if k > len(stack) else stack[-k])
            else:
                k = 0
                stack.append(int(word))
        return result
```

## 2901. Longest Unequal Adjacent Groups Subsequence II

### Solution 1:  bfs, backtracking, parent arrays, directed graph, topological ordering, indegrees

```py
class Solution:
    def getWordsInLongestSubsequence(self, n: int, words: List[str], groups: List[int]) -> List[str]:
        adj = [[] for _ in range(n)]
        is_edge = lambda i, j: groups[i] != groups[j] and len(words[i]) == len(words[j]) and sum(1 for x, y in zip(words[i], words[j]) if x != y) == 1
        indegrees = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if is_edge(i, j):
                    adj[i].append(j)
                    indegrees[j] += 1
        queue = deque()
        for i in range(n):
            if indegrees[i] == 0: queue.append(i)
        parents = [None] * n
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                indegrees[v] -= 1
                if indegrees[v] == 0: 
                    parents[v] = u
                    queue.append(v)
        path = []
        while u is not None:
            path.append(words[u])
            u = parents[u]
        return reversed(path)
```

## 2902. Count of Sub-Multisets With Bounded Sum

### Solution 1:  unoptimized dynamic programming solution

Count the number of multisets

knapsack dp problem, can have multiple of same item

you have c of item of size a
update is dp[i] += dp[i - a] + dp[i - a * 2] + ... + dp[i - a * c]

You need to improve this update function though, it mentions the idea of sliding window by keeping the sum of dp[i - a] + ... + dp[i - a * c]

```py
class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        mod = int(1e9) + 7
        n = len(nums)
        dp = [0] * (r + 1)
        freq = Counter(nums)
        dp[0] = freq[0] + 1
        for num in freq:
            if num == 0: continue
            for j in range(r, num - 1, -1):
                for k in range(1, freq[num] + 1):
                    if k * num > j: break
                    dp[j] = (dp[j] + dp[j - k * num]) % mod
        res = 0
        for i in range(l, r + 1):
            res = (res + dp[i]) % mod
        return res
```

```py
from collections import Counter 
class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10 ** 9 + 7 
        counter = Counter(nums)
        dp = [0 for _ in range(r + 1)]
        dp[0] = 1 

        for num, freq in counter.items(): 
            for i in range(r, max(r - num, 0), -1): 
                v = sum(dp[i - num * k] for k in range(freq) if i >= num * k)
                for j in range(i, 0, -num):
                    v -= dp[j] 
                    if j >= num * freq: 
                        v += dp[j - num * freq]
                    dp[j] = (dp[j] + v) % MOD

        return (sum(dp[l:])) * (counter[0] + 1) % MOD
```

