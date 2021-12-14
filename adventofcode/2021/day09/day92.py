# sys.stdout = open('outputs/output.txt', 'w')
from functools import lru_cache
from collections import Counter
with open("inputs/input.txt", "r") as f:
    data = []
    lines = f.read().split()
    for line in lines:
        data.append([int(x) for x in line])
    R, C = len(data), len(data[0])
    @lru_cache(maxsize=None)
    def dfs(x, y):
        for i, j in ((x, y-1), (x, y+1), (x-1, y), (x+1, y)):
            if 0 <= i < R and 0 <= j < C and data[i][j] < data[x][y]:
                return dfs(i,j)
        return (x,y)
    basins = Counter(dfs(i,j) for i in range(R) for j in range(C) if data[i][j] != 9)
    heights = sorted(list(basins.values()))
    print(heights[-1]*heights[-2]*heights[-3])
# sys.stdout.close()