from collections import *
from itertools import *
from heapq import *
import time
def main():
    with open('input.txt', 'r') as f:
        data = f.read().splitlines()
        grid = [list(line) for line in data]
        R, C = len(grid), len(grid[0])
        wall, empty = '#', '.'
        left, right, up, down = (0, -1), (0, 1), (-1, 0), (1, 0)
        directions = {'>': right, '<': left, '^': up, 'v': down}
        target = None
        start = None
        vertical, horizontals = [], []
        for r, c in product(range(R), range(C)):
            if grid[r][c] == empty and r == 0:
                start = (r, c)
            elif grid[r][c] == empty and r == R-1:
                target = (r, c)
            elif grid[r][c] not in (wall, empty):
                if grid[r][c] in '><':
                    horizontals.append((r, c, grid[r][c]))
                elif grid[r][c] in '^v':
                    vertical.append((r, c, grid[r][c]))
        ri, ci = start
        neighborhood = lambda r, c: ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
        in_bounds = lambda r, c: 0 <= r < R and 0 <= c < C
        manhattan = lambda r, c, p: abs(r-target[0]) + abs(c-target[1]) if p != 1 else abs(r-start[0]) + abs(c-start[1])
        minheap = [(0, manhattan(ri, ci, 0), ri, ci, 0)]
        horizontal_blizzard_cache = {}
        vertical_blizzard_cache = {}
        for i in range(C-2):
            horizontal_blizzard_cache[i] = Counter()
            for j, (r, c, d) in enumerate(horizontals):
                horizontal_blizzard_cache[i][(r, c)] += 1
                nr, nc = r + directions[d][0], c + directions[d][1]
                if nc == C-1:
                    nc = 1
                elif nc == 0:
                    nc = C-2
                horizontals[j] = (nr, nc, d)
        for i in range(R-2):
            vertical_blizzard_cache[i] = Counter()
            for j, (r, c, d) in enumerate(vertical):
                vertical_blizzard_cache[i][(r, c)] += 1
                nr, nc = r + directions[d][0], c + directions[d][1]
                if nr == R-1:
                    nr = 1
                elif nr == 0:
                    nr = R-2
                vertical[j] = (nr, nc, d)
        # seen = set()
        vis = set()
        while minheap:
            time, _, row, col, path = heappop(minheap)
            # if time not in seen:
            #     print(f'time: {time}, heap size: {len(minheap)}')
            #     seen.add(time)
            ntime = time + 1
            if (row, col) == target and path == 0:
                nstate = (time, manhattan(row, col, 1), row, col, 1)
                heappush(minheap, nstate)
                continue
            if (row, col) == start and path == 1:
                nstate = (time, manhattan(row, col, 2), row, col, 2)
                heappush(minheap, nstate)
                continue
            if (row, col) == target and path == 2: return time
            for nr, nc in neighborhood(row, col):
                nstate = (ntime, manhattan(nr, nc, path), nr, nc, path)
                if not in_bounds(nr, nc) or grid[nr][nc] == wall or nstate in vis or horizontal_blizzard_cache[ntime%(C-2)][(nr, nc)] > 0 or \
                    vertical_blizzard_cache[ntime%(R-2)][(nr, nc)] > 0: continue
                heappush(minheap, nstate)
                vis.add(nstate)
            # waiting
            if horizontal_blizzard_cache[ntime%(C-2)][(row, col)] == 0 and vertical_blizzard_cache[ntime%(R-2)][(row, col)] == 0:
                nstate = (ntime, manhattan(row, col, path), row, col, path)
                if nstate in vis: continue
                heappush(minheap, nstate)
                vis.add(nstate)

if __name__ == '__main__':
    start_time = time.perf_counter()
    print(main())
    end_time = time.perf_counter()
    print(f'Time Elapsed: {end_time - start_time} seconds')