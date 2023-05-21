from typing import *
import math
import heapq

class Solution:
    def shortestBridge(self, grid: List[List[int]]) -> int:
        n = len(grid)
        in_bounds = lambda r, c: 0 <= r < n and 0 <= c < n
        neighborhood = lambda r, c: [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
        def bfs(row, col):
            queue = deque([(row, col)])
            multiqueue = deque([(row, col)])
            grid[row][col] = -1
            while queue:
                r, c = queue.popleft()
                for nr, nc in neighborhood(r, c):
                    if not in_bounds(nr, nc) or grid[nr][nc] != 1: continue
                    grid[nr][nc] = -1
                    queue.append((nr, nc))
                    multiqueue.append((nr, nc))
            steps = 0
            while multiqueue:
                steps += 1
                for _ in range(len(multiqueue)):
                    r, c = multiqueue.popleft()
                    for nr, nc in neighborhood(r, c):
                        if not in_bounds(nr, nc) or grid[nr][nc] == -1: continue
                        if grid[nr][nc] == 1: return steps
                        grid[nr][nc] = -1
                        multiqueue.append((nr, nc))
            return steps
        for r, c in product(range(n), repeat = 2):
            if grid[r][c] == 1:
                return bfs(r, c)
        return -1
        
        