class Solution:
    def knightProbability(self, n: int, k: int, row: int, column: int) -> float:
        board = [[0.0] * n for _ in range(n)]
        board[row][column] = 1.0
        in_bounds = lambda r, c: 0 <= r < n and 0 <= c < n
        manhattan = lambda r, c: abs(r) + abs(c)
        def neighborhood(r, c):
            for dr, dc in repeat(range(1, 3), repeat = 2):
                if manhattan(dr, dc) != 3: continue
                yield r + dr, c + dc
        print(list(neighborhood(0, 0)))
        # for _ in range(k):
        #     nboard = [[0.0] * n for _ in range(n)]
        #     for r, c in product(range(n), repeat = 2):
        #         if board[r][c] == 0.0: continue
        #         for nr, nc in neighborhood(r, c):
        #             if not in_bounds(nr, nc): continue
        #             nboard[nr][nc] += board[r][c] / 8

        #     board = nboard
        return sum(map(sum, board))