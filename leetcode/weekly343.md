# Leetcode Weekly Contest 343

## 2660. Determine the Winner of a Bowling Game

### Solution 1: loop

```py
class Solution:
    def isWinner(self, player1: List[int], player2: List[int]) -> int:
        def score(player):
            prev = sum_ = 0
            for x in player:
                sum_ += x
                if prev > 0:
                    sum_ += x
                    prev -= 1
                if x == 10:
                    prev = 2
            return sum_
        sum1, sum2 = map(score, (player1, player2))
        if sum1 == sum2: return 0
        return 1 if sum1 > sum2 else 2
```

## 2661. First Completely Painted Row or Column

### Solution 1:  hash table + horizontal and vertical sum

```py
class Solution:
    def firstCompleteIndex(self, arr: List[int], mat: List[List[int]]) -> int:
        R, C = len(mat), len(mat[0])
        n = len(arr)
        horz_sum, vert_sum = [0]*R, [0]*C
        pos = {mat[r][c]: (r, c) for r, c in product(range(R), range(C))}
        for i, val in enumerate(arr):
            r, c = pos[val]
            horz_sum[r] += 1
            vert_sum[c] += 1
            if horz_sum[r] == C or vert_sum[c] == R: return i
        return n
```

## 2662. Minimum Cost of a Path With Special Roads

### Solution 1:  bfs + memoization + shortest path in directed graph

Find the minimum cost to travel to each end node in the specialRoads, so sometimes can use multiple specialRoads to get there.  So using bfs to build up this path.  And relax the cost of each node in the path when you find a better route.

Then treat each end location in the spcialRoads as a start point for getting to the target. 

```py
class Solution:
    def minimumCost(self, start: List[int], target: List[int], specialRoads: List[List[int]]) -> int:
        E = len(specialRoads)
        min_cost = defaultdict(lambda: math.inf)
        queue = deque([tuple(start)])
        min_cost[tuple(start)] = 0
        manhattan_distance = lambda x1, y1, x2, y2: abs(x1 - x2) + abs(y1 - y2)
        while queue:
            x, y = queue.popleft()
            for x1, y1, x2, y2, cost in specialRoads:
                ncost = min_cost[(x, y)] + manhattan_distance(x, y, x1, y1) + cost
                if ncost < min_cost[(x2, y2)]:
                    min_cost[(x2, y2)] = ncost
                    queue.append((x2, y2))
        res = math.inf
        for (x, y), cost in min_cost.items():
            cur_cost = cost + manhattan_distance(*target, x, y) 
            res = min(res, cur_cost)
        return res
```

##

### Solution 1:

```py

```

