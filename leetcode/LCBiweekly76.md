# Leetcode Biweekly Contest 76

## Summary

## 2239. Find Closest Number to Zero

### Solution 1: 

```py
class Solution:
    def findClosestNumber(self, nums: List[int]) -> int:
        return min(nums, key=lambda x: (abs(x),-x))
```

## 2240. Number of Ways to Buy Pens and Pencils

### Solution 1: number of ways with single loop

```py
class Solution:
    def waysToBuyPensPencils(self, total: int, cost1: int, cost2: int) -> int:
        cnt = 0
        while total >= 0:
            cnt += total//cost2 + 1
            total -= cost1
        return cnt
```

## 2241. Design an ATM Machine

### Solution 1: hash table

```py
class ATM:

    def __init__(self):
        self.cash = [0]*5
        self.values = [20,50,100,200,500]        

    def deposit(self, banknotesCount: List[int]) -> None:
        for i, cnt in enumerate(banknotesCount):
            self.cash[i] += cnt

    def withdraw(self, amount: int) -> List[int]:
        withdrawn = [0]*5
        for i, cash, val in zip(count(4, -1), self.cash[::-1], self.values[::-1]):
            used = min(cash, amount//val)
            amount -= used*val
            withdrawn[i] = used
        if amount == 0:
            self.deposit([-x for x in withdrawn])
            return withdrawn
        return [-1]
```

## 2242. Maximum Score of a Node Sequence

### Solution 1: Undirected graph + hash table

Using the idea of having two fixed nodes and then each fixed node has a single neighbor, want to find the 
maximum combination, to do this need to score the 3 largest valued neighbors for each node

```py
class Solution:
    def maximumScore(self, scores: List[int], edges: List[List[int]]) -> int:
        n = len(scores) # nodes [0,n-1]
        graph_lst = [[] for _ in range(n)]
        for u, v in edges:
            graph_lst[u].append((scores[v], v))
            graph_lst[v].append((scores[u], u))
        for i in range(n):
            graph_lst[i] = nlargest(3, graph_lst[i])
        max_score = -1
        for u, v in edges:
            for nuscore, nu in graph_lst[u]:
                for nvscore, nv in graph_lst[v]:
                    if nu!=nv and nu!=v and nv!=u:
                        max_score = max(max_score, scores[u]+scores[v]+nvscore+nuscore)
        return max_score
```