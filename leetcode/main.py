class Solution:
    def candyCrush(self, board: List[List[int]]) -> List[List[int]]:
        R, C = len(board), len(board[0])
        def find():
            found = False
            for r, c in product(range(R), range(C)):
                # mark horizontal
                if 0 < c < C - 1 and board[r][c] != 0 and abs(board[r][c - 1]) == abs(board[r][c]) == abs(board[r][c + 1]):
                    for i in range(c - 1, c + 2):
                        board[r][i] = -abs(board[r][i])
                    found = True
                # mark vertical
                if 0 < r < R - 1 and board[r][c] != 0 and abs(board[r - 1][c]) == abs(board[r][c]) == abs(board[r + 1][c]):
                    for i in range(r - 1, r + 2):
                        board[i][c] = -abs(board[i][c])
                    found = True
            for r, c in product(range(R), range(C)):
                if board[r][c] < 0: board[r][c] = 0 # mark as empty
            return found
        def drop():
            for c in range(C):
                lowest_zero = -1
                for r in range(R):
                    if lowest_zero == -1 and board[r][c] == 0:
                        lowest_zero = lowest_zero = r
                    elif lowest_zero >= 0 and board[r][c] > 0:
                        board[lowest_zero][c], board[r][c] = board[r][c], board[lowest_zero][c]
                        lowest_zero += 1
        while not find():
            drop()
        return board