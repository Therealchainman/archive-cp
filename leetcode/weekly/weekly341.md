# Leetcode Weekly Contest 341

## 2643. Row With Maximum Ones

### Solution 1:  one liner with max and custom comparator

```py
class Solution:
    def rowAndMaximumOnes(self, mat: List[List[int]]) -> List[int]:
        return max([[sum(row), r] for r, row in enumerate(mat)], key = lambda elem: (elem[0], -elem[1]))[::-1]
```

## 2644. Find the Maximum Divisibility Score

### Solution 1:  max + tiebreaker minimize on second element + maximum applied to tuples

```py
class Solution:
    def maxDivScore(self, nums: List[int], divisors: List[int]) -> int:
        result = (-math.inf, -math.inf)
        for div in divisors:
            div_score = sum([1 for num in nums if num%div == 0])
            result = max(result, (div_score, div), key = lambda pair: (pair[0], -pair[1]))
        return result[1]
```

## 2645. Minimum Additions to Make Valid String

### Solution 1:  cycle matching

```py
class Solution:
    def addMinimum(self, word: str) -> int:
        target = "abc"
        res, n, j= 0, len(word), 0
        for i in range(n):
            while word[i] != target[j]:
                res += 1
                j = (j + 1)%len(target)
            j = (j + 1)%len(target)
        return res + (3 - j if j > 0 else 0)
```

### Solution 2:  counting number of "abc" strings

```py
class Solution:
    def addMinimum(self, word: str) -> int:
        cycles = 0
        n = len(word)
        prev = 'z'
        for i in range(n):
            cycles += word[i] <= prev
            prev = word[i]
        return 3*cycles - n
```

## 2646. Minimize the Total Price of the Trips

### Solution 1:  dfs + dynammic programming on tree + path reconstruction

1. build adjacency list
1. construct a frequency array for each shortest path between the node pairs in trips.  can use a iterative dfs algorithm and store the parent nodes along the way to be able to pass back through from end to start node along the shortest path in a tree and compute the frequency of each node.
1. dynammic programming with the states being the current node and if the previous node price was halved or not.  get the minimum of these two options.

```py
class Solution:
    def minimumTotalPrice(self, n: int, edges: List[List[int]], price: List[int], trips: List[List[int]]) -> int:
        # CONSTRUCT ADJACENCY LIST
        adj_list = [[] for _ in range(n)]
        for u, v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
        # BUILD FREQUENCY ARRAY WITH DFS
        freq = [0]*n
        for start, end in trips:
            stack = [(start, -1)]
            parent_arr = [-1]*n
            while stack:
                node, parent = stack.pop()
                parent_arr[node] = parent
                if node == end: break
                for nei in adj_list[node]:
                    if nei == parent: continue
                    stack.append((nei, node))
            # GO THROUGH PATH
            while node != -1:
                freq[node] += 1
                node = parent_arr[node]
        # DYNAMMIC PROGRAMMING ON ARBITRARY ROOT OF TREE
        @cache
        def dp(node, parent, prev_halved):
            halved_sum = math.inf if prev_halved else 0
            full_sum = 0
            for nei in adj_list[node]:
                if nei == parent: continue
                full_sum += dp(nei, node, 0)
                if not prev_halved:
                    halved_sum += dp(nei, node, 1)
            return min(halved_sum + price[node]*freq[node]//2, full_sum + price[node]*freq[node])
        return dp(0, -1, 0)
```