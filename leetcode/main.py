class Solution:
    def latestDayToCross(self, row: int, col: int, cells: List[List[int]]) -> int:
        cells = [(r - 1, c - 1) for r, c in cells]
        neighborhood = lambda r, c: [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        in_bounds = lambda r, c: 0 <= r < row and 0 <= c < col
        land, water = 0, 1
        def bfs(target):
            grid = [[0] * col for _ in range(row)]
            for r, c in map(lambda i: cells[i], range(target)):
                grid[r][c] = water # which ones will be covered with water by target
            queue = deque()
            for c in range(col):
                if grid[0][c] == land:
                    queue.append((0, c))
                    grid[0][c] = water
            while queue:
                r, c = queue.popleft()
                if r == row - 1: return True # reached bottom row
                for nr, nc in neighborhood(r, c):
                    if not in_bounds(nr, nc) or grid[nr][nc] != land: continue
                    grid[nr][nc] = water
                    queue.append((nr, nc))
            return False
        left, right = 0, row * col
        while left < right:
            mid = (left + right + 1) >> 1
            print('mid', mid, bfs(mid))
            if bfs(mid):
                left = mid
            else:
                right = mid - 1
        return left
