from collections import *
from itertools import *
import time

def main():
    with open('input.txt', 'r') as f:
        data = f.read().splitlines()
        grid = [list(line) for line in data]
        R, C = len(grid), len(grid[0])
        occupied = set()
        elf = '#'
        for r, c in product(range(R), range(C)):
            if grid[r][c] == elf:
                occupied.add((r, c))
        north, south, west, east = (-1, 0), (1, 0), (0, -1), (0, 1)
        northwest, southwest, northeast, southeast = (-1, -1), (1, -1), (-1, 1), (1, 1)
        proposals = [north, south, west, east]
        rounds = 10
        for round in range(1, rounds+1):
            next_move = {}
            counter = Counter()
            for r, c in occupied:
                for proposal in proposals:
                    taken = False
                    proposed = None
                    if proposal == north:
                        for dr, dc in [north, northwest, northeast]:
                            nr, nc = r + dr, c + dc
                            if (nr, nc) in occupied:
                                taken = True
                                break
                    elif proposal == south:
                        for dr, dc in [south, southwest, southeast]:
                            nr, nc = r + dr, c + dc
                            if (nr, nc) in occupied:
                                taken = True
                                break
                    elif proposal == west:
                        for dr, dc in [west, northwest, southwest]:
                            nr, nc = r + dr, c + dc
                            if (nr, nc) in occupied:
                                taken = True
                                break
                    elif proposal == east:
                        for dr, dc in [east, northeast, southeast]:
                            nr, nc = r + dr, c + dc
                            if (nr, nc) in occupied:
                                taken = True
                                break
                    if not taken: 
                        proposed = proposal
                        nr, nc = r + proposed[0], c + proposed[1]
                        next_move[(r, c)] = (nr, nc)
                        counter[(nr, nc)] += 1
                        break

            for (r, c), (nr, nc) in next_move.items():
                if counter[(nr, nc)] == 1:
                    occupied.discard((r, c))
                    occupied.add((nr, nc))
            proposals = proposals[1:] + proposals[:1]
        minRow, maxRow, minCol, maxCol = min(r for r, c in occupied), max(r for r, c in occupied), min(c for r, c in occupied), max(c for r, c in occupied)
        return sum(1 for r, c in product(range(minRow, maxRow+1), range(minCol, maxCol+1)) if (r, c) not in occupied)

if __name__ == '__main__':
    start_time = time.perf_counter()
    print(main())
    end_time = time.perf_counter()
    print(f'Time Elapsed: {end_time - start_time} seconds')