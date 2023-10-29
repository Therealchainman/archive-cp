import sys

name = "ready_go_part_2_input.txt"

sys.stdout = open(f"outputs/{name}", "w")
sys.stdin = open(f"inputs/{name}", "r")

from collections import deque, Counter
from itertools import product

def main():
    R, C = map(int, input().split())
    grid = [list(input()) for _ in range(R)]
    empty, black, white = ".", "B", "W"
    vis = "#"
    in_bounds = lambda r, c: 0 <= r < R and 0 <= c < C
    neighborhood = lambda r, c: [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
    counts = Counter()
    def bfs(r, c):
        vis2 = set([(r, c)])
        queue = deque([(r, c)])
        sr = sc = sz = cnt = 0
        while queue:
            r, c = queue.popleft()
            sz += 1
            grid[r][c] = vis
            for nr, nc in neighborhood(r, c):
                if not in_bounds(nr, nc): continue
                if (nr, nc) in vis2: continue
                vis2.add((nr, nc))
                if grid[nr][nc] == empty: 
                    sr, sc = nr, nc
                    cnt += 1
                if grid[nr][nc] == white: queue.append((nr, nc))
        return sr, sc, sz, cnt
    for r, c in product(range(R), range(C)):
        if grid[r][c] == vis or grid[r][c] != white: continue
        rr, cc, sz, cnt = bfs(r, c)
        if cnt == 1: counts[(rr, cc)] += sz
    return max(counts.values(), default = 0)

if __name__ == '__main__':
    T = int(input())
    for t in range(1, T + 1):
        print(f"Case #{t}: {main()}")