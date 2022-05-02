# Leetcode Biweekly Contest 77

## Summary

## 2255. Count Prefixes of a Given String

### Solution 1: check prefixes

```py
class Solution:
    def countPrefixes(self, words: List[str], s: str) -> int:
        return sum(1 for word in words if s[:len(word)] == word)
```

## 2256. Minimum Average Difference

### Solution 1: prefix sum and suffix sum

```py
class Solution:
    def minimumAverageDifference(self, nums: List[int]) -> int:
        psum, ssum = 0, sum(nums)
        n = len(nums)
        bestVal, bestIdx = inf, 0
        for i, num in enumerate(nums):
            psum += num
            ssum -= num
            pavg = psum//(i+1)
            savg = ssum//(n-i-1) if n-i-1 > 0 else 0
            curVal = abs(pavg-savg)
            if curVal < bestVal:
                bestVal = curVal
                bestIdx = i
        return bestIdx
```

## 2257. Count Unguarded Cells in the Grid

### Solution 1: 

```py
class Solution:
    def countUnguarded(self, m: int, n: int, guards: List[List[int]], walls: List[List[int]]) -> int:
        # m is rows, n is cols
        grid = [['U']*n for _ in range(m)] # 0 represents unguarded
        for r, c in walls:
            grid[r][c] = 'w' # 3 represents wall block
        for r, c in guards:
            grid[r][c] = 'G' # 2 represents guard
            for row in range(r+1,m):
                if grid[row][c] in ('E', 'G', 'w'): break
                grid[row][c] = 'E'
            for row in range(r-1,-1,-1):
                if grid[row][c] in ('W', 'G', 'w'): break
                grid[row][c] = 'W'
            for col in range(c+1,n):
                if grid[r][col] in ('S', 'G', 'w'): break
                grid[r][col] = 'S'
            for col in range(c-1,-1,-1):
                if grid[r][col] in ('N', 'G', 'w'): break
                grid[r][col] = 'N'
        cnt = 0
        for r, c in product(range(m), range(n)):
            cnt += (grid[r][c]=='U')
        return cnt
```

## 2258. Escape the Spreading Fire

### Solution 1: multisource bfs for fire + binary search for start time

```py
class Solution:
    def maximumMinutes(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        # can escape at this start time for the person
        def canEscape(start_time):
            queue = deque([(0,0,start_time)])
            visited = [[0]*C for _ in range(R)]
            visited[0][0] = 1
            while queue:
                row, col, time = queue.popleft()
                if row==R-1 and col==C-1: return True
                for nr, nc in [(row+1,col),(row-1,col),(row,col+1),(row,col-1)]:
                    if not in_boundary(nr,nc) or time+1 >= fire_time[R-1][C-1] or visited[nr][nc]: continue
                    queue.append((nr,nc,time+1))
                    visited[nr][nc] = 1
            return False
        # BUILD THE MULTISORCE BFS FOR FIRE TO BUILD THE TIMES
        fire_time = [[-1]*C for _ in range(R)]
        queue = deque()
        in_boundary = lambda r, c: 0<=r<R and 0<=c<C
        for r, c in product(range(R), range(C)):
            if grid[r][c] == 1:
                queue.append((r,c, 0))
                fire_time[r][c] = 0
        while queue:
            row, col, time = queue.popleft()
            for nr, nc in [(row+1,col),(row-1,col),(row,col+1),(row,col-1)]:
                if not in_boundary(nr,nc) or grid[nr][nc] == 2 or fire_time[nr][nc] != -1: continue
                fire_time[nr][nc] = time+1
                queue.append((nr,nc,time+1))
        left, right = 0, 100000
        print(fire_time)
        if fire_time[R-1][C-1] == -1: return 1000000000
        if not canEscape(0): return -1
        while left < right:
            mid = (left+right+1)>>1
            # print(left, mid, right)
            if canEscape(mid):
                left = mid
            else:
                right = mid-1
        
        return left
```

### Solution 2: BFS + compute distance to the safehouse for fire and person + treat edge case when both reach at same time but can reach coming from different direction

```py
class Solution:
    def maximumMinutes(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        in_boundary = lambda r, c: 0<=r<R and 0<=c<C
        def bfs(queue):
            dist = {node: 0 for node in queue}
            while queue:
                r, c = queue.popleft()
                for nr, nc in [(r+1,c),(r-1,c),(r,c+1),(r,c-1)]:
                    if not in_boundary(nr,nc) or grid[nr][nc] == 2 or (nr,nc) in dist: continue
                    dist[(nr,nc)] = dist[(r,c)] + 1
                    queue.append((nr,nc))
            return dist
        queue = deque()
        for r, c in product(range(R), range(C)):
            if grid[r][c] == 1:
                queue.append((r,c))
        dist_fire = bfs(queue)
        dist_person = bfs(deque([(0,0)]))
        
        if (R-1,C-1) not in dist_person: return -1
        if (R-1,C-1) not in dist_fire: return 10**9
        
        def time(r,c):
            return dist_fire[(r,c)] - dist_person[(r,c)]
        t = time(R-1,C-1)
        if grid[-1][-2] !=2 and grid[-2][-1] != 2 and max(time(R-1,C-2), time(R-2,C-1)) > t:
            t+=1
        return max(t - 1,-1)
```