import time

def display(grid):
    print("\n".join(["".join(row) for row in grid]))

def main():
    with open('input.txt', 'r') as f:
        data = f.read().splitlines()
        grid = [list(line) for line in data[:-2]]
        R = len(grid)
        empty, wall, void = '.', '#', ' '
        instructions = data[-1]
        right, down, left, up = 0, 1, 2, 3
        headings = {right: (0, 1), down: (1, 0), left: (0, -1), up: (-1, 0)}
        def move(steps, heading, r, c):
            for _ in range(steps):
                nr, nc = r + headings[heading][0], c + headings[heading][1]
                nr %= R
                nc %= C
                if grid[nr][nc] == wall: break
                while grid[nr][nc] == void:
                    nr += headings[heading][0]
                    nc += headings[heading][1]
                    nr %= R
                    nc %= C
                if grid[nr][nc] == wall: break
                r = nr
                c = nc
            return r, c
        C = 0
        for row in grid:
            C = max(C, len(row))
        for i in range(R):
            n = len(grid[i])
            grid[i].extend([void]*(C-n))
        r, c = 0, grid[0].index(empty)
        num = 0
        heading = right
        for x in instructions:
            if x.isdigit():
                num = num * 10 + int(x)
            else:
                r, c = move(num, heading, r, c)
                num = 0
                if x == 'R':
                    heading = (heading + 1) % 4
                elif x == 'L':
                    heading = (heading - 1) % 4
        r, c = move(num, heading, r, c)
        return 1000*(r+1) + 4*(c+1) + heading


if __name__ == '__main__':
    start_time = time.perf_counter()
    print(main())
    end_time = time.perf_counter()
    print(f'Time Elapsed: {end_time - start_time} seconds')