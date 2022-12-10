from collections import *
from itertools import *

"""
if sprit_pos = 4
sprite position: ...xxx....
sprite size as 3
"""

def main():
    with open('input.txt', 'r') as f:
        data = f.read().splitlines()
        screen_len = 40
        grid = [['.' for _ in range(screen_len)] for y in range(6)]
        cycle, sprite_pos = 0, 1
        neighborhood = lambda i: (i-1, i, i+1)
        row = lambda i: i//screen_len
        col = lambda i: i%screen_len
        def update_grid():
            c = col(cycle)
            if c in neighborhood(sprite_pos):
                grid[row(cycle)][c] = '#'
        for ins in data:
            if ins == 'noop':
                update_grid()
                cycle += 1
            else:
                _, delta = ins.split()
                delta = int(delta)
                update_grid()
                cycle += 1
                update_grid()
                cycle += 1
                sprite_pos += delta
        return "\n".join(["".join(row) for row in grid])
if __name__ == "__main__":
    print(main())