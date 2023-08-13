class Solution:
    def minimumSeconds(self, land: List[List[str]]) -> int:
        start, target, empty, stone, flood = 'S', 'D', '.', 'X', '*'
        R, C = len(land), len(land[0])
        frontier, queue = deque(), deque()
        for r, c in product(range(R), range(C)):
            if land[r][c] == start: queue.append((r, c))
            elif land[r][c] == flood: frontier.append((r, c))
        in_bounds = lambda r, c: 0 <= r < R and 0 <= c < C
        neighborhood = lambda r, c: [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        steps = 0
        while queue:
            # update the flooded cells
            for _ in range(len(frontier)):
                r, c = frontier.popleft()
                for nr, nc in neighborhood(r, c):
                    if not in_bounds(nr, nc) or land[nr][nc] not in (empty, start): continue
                    land[nr][nc] = flood
                    frontier.append((nr, nc))
            for _ in range(len(queue)):
                r, c = queue.popleft()
                if (r, c) == (R - 1, C - 1): return steps
                for nr, nc in neighborhood(r, c):
                    if not in_bounds(nr, nc) or land[nr][nc] not in (empty, target): continue
                    land[nr][nc] = start
                    queue.append((nr, nc))
            steps += 1
        return -1