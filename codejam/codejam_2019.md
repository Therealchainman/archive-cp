# Google Code Jam 2019

# Round 1C

## Bacterial Tactics

### Solution 1:  Sprague grundy theorem + nimbers + nim + recursive dp + count distinct winning positions + impartial game

```py
import sys
sys.setrecursionlimit(1_000_000)
from functools import lru_cache
from itertools import dropwhile

def main():
    R, C = map(int, input().split())
    empty, radioactive = '.', '#'
    grid = [list(input()) for _ in range(R)]

    @lru_cache(None)
    def dp(min_row, max_row, min_col, max_col):
        if max_row < min_row or max_col < min_col: return 0
        nimbers = [False]*20
        for row in range(min_row, max_row + 1):
            is_radioactive = False
            for col in range(min_col, max_col + 1):
                if grid[row][col] == radioactive:
                    is_radioactive = True
                    break
            if not is_radioactive:
                nimbers[dp(min_row, row-1, min_col, max_col) ^ dp(row+1, max_row, min_col, max_col)] = 1
        for col in range(min_col, max_col + 1):
            is_radioactive = False
            for row in range(min_row, max_row + 1):
                if grid[row][col] == radioactive:
                    is_radioactive = True
                    break
            if not is_radioactive:
                nimbers[dp(min_row, max_row, min_col, col-1) ^ dp(min_row, max_row, col+1, max_col)] = 1
        return next(dropwhile(lambda i: nimbers[i], range(20)))

    res = 0
    lost = True
    for row in range(R):
        is_radioactive = False
        for col in range(C):
            if grid[row][col] == radioactive:
                is_radioactive = True
                break
        if not is_radioactive:
            nim_sum = dp(0, row-1, 0, C-1) ^ dp(row+1, R-1, 0, C-1)
            if nim_sum == 0: res += C
            lost &= nim_sum > 0
    for col in range(C):
        is_radioactive = False
        for row in range(R):
            if grid[row][col] == radioactive:
                is_radioactive = True
                break
        if not is_radioactive:
            nim_sum = dp(0, R-1, 0, col-1) ^ dp(0, R-1, col+1, C-1)
            if nim_sum == 0: res += R
            lost &= nim_sum > 0
    if lost: return 0
    return res

if __name__ == '__main__':
    T = int(input())
    for t in range(1, T+1):
        print(f'Case #{t}: {main()}')
```