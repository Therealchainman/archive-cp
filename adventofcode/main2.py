from collections import *
from itertools import *
import math

class Cube:
    def __init__(self, rows, cols, grid):
        right, down, left, up = 0, 1, 2, 3
        self.right, self.down, self.left, self.up = right, down, left, up
        self.wrap = {1: {right: (2, right), down: (3, down), left: (4, right), up: (6, right)}, 2: {right: (5, left), down: (3, left), left: (1, left), up: (6, up)}, \
            3: {right: (2, up), down: (5, down), left: (4, down), up: (1, up)}, 4: {right: (5, right), down: (6, down), left: (1, right), up: (3, right)}, \
            5: {right: (2, left), down: (6, left), left: (4, left), up: (3, up)}, 6: {right: (5, up), down: (2, down), left: (1, down), up: (4, up)}}
        self.headings = {right: (0, 1), down: (1, 0), left: (0, -1), up: (-1, 0)}
        self.heading_map = ['right', 'down', 'left', 'up']
        self.horizontal = (self.right, left)
        self.vertical = (up, down)
        self.empty, self.wall, self.void = '.', '#', ' '
        self.R, self.C = rows, cols
        self.grid = grid
        self.build()

    def build(self):
        self.face = {}
        cnt = 1
        for r in range(len(self.grid)):
            for c in range(len(self.grid[r])):
                if self.grid[r][c] == self.void: continue
                sr, sc = r, c
                cu = self.construct(r, c)
                self.face[cnt] = cu
                cnt += 1
    
    def construct(self, starting_row, starting_col):
        cube_face = [[' ']*self.C for _ in range(self.R)]
        for r, c in product(range(starting_row, starting_row + self.R), range(starting_col, starting_col + self.C)):
            cube_face[r-starting_row][c-starting_col] = self.grid[r][c]
            self.grid[r][c] = self.void
        return cube_face

    def get_row_col(self, prev_heading, new_heading, row, col):
        if new_heading in self.horizontal:
            ncol = 0 if new_heading == self.right else self.C-1
            nrow = row if prev_heading in self.horizontal else col
        else:
            nrow = 0 if new_heading == self.down else self.R-1
            ncol = col if prev_heading in self.vertical else row
        return nrow, ncol

    def move(self, steps, heading, cur, r, c, cube):
        for i in range(steps):
            nr, nc = r + self.headings[heading][0], c + self.headings[heading][1]
            ncur, nheading = cur, heading
            if nr in (-1, self.R) or nc in (-1, self.C):
                print('========Needs to transition to another face of the cube========')
                print('cube, heading, row, col', cur, self.heading_map[heading], nr, nc)
                ncur, nheading = self.wrap[cur][heading]
                nr, nc = self.get_row_col(heading, nheading, r, c)
                print('new cube, heading, row, col', ncur, self.heading_map[nheading], nr, nc)
            if cube.face[ncur][nr][nc] == self.wall: break
            r, c, cur, heading = nr, nc, ncur, nheading
        print(f'number of movements: {i}')
        return r, c, cur, heading

            
def display(grid):
    print("\n".join(["".join(row) for row in grid]))

def main():
    with open('input.txt', 'r') as f:
        data = f.read().splitlines()
        grid = [list(line) for line in data[:-2]]
        cube = Cube(50, 50, grid)
        instructions = data[-1]
        right, down, left, up = 0, 1, 2, 3
        heading_map = ['right', 'down', 'left', 'up']
        r = c = num = 0
        heading = right
        cur_cube = 1
        cnt = 0
        for x in instructions:
            if x.isdigit():
                num = num * 10 + int(x)
            else:
                print(f'move count: {cnt}')
                print('starting position and facing direction and moves and cube', r, c, heading_map[heading], num, cur_cube)
                r, c, cur_cube, heading = cube.move(num, heading, cur_cube, r, c, cube)
                print('ending position and facing direction and cube', r, c, heading_map[heading], cur_cube)
                print('--------------------------------------------------------------------------------------------------------------------------')
                cnt += 1
                # if cnt == 5: break
                num = 0
                if x == 'R':
                    heading = (heading + 1) % 4
                elif x == 'L':
                    heading = (heading - 1) % 4
        print('starting position and facing direction and moves and cube', r, c, heading_map[heading], num, cur_cube)
        r, c, cur_cube, heading = cube.move(num, heading, cur_cube, r, c, cube)
        print('ending position and facing direction and cube', r, c, heading_map[heading], cur_cube)
        print('--------------------------------------------------------------------------------------------------------------------------')
        return r, c, cur_cube, heading
        
if __name__ == '__main__':
    start_time = time.perf_counter()
    print(main())
    end_time = time.perf_counter()
    print(f'Time Elapsed: {end_time - start_time} seconds')