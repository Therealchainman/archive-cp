import math
import itertools
def power(x: int, y: int, serial: int) -> int:
    rack_id = x + 10
    power_level = (rack_id*y+serial)*rack_id
    power_level = (power_level // 100) % 10
    power_level -= 5
    return power_level
def main():
    with open('input.txt', 'r') as f:
        serial = 4172
        N = 300
        cells = [[0]*N for _ in range(N)]
        for y, x in itertools.product(range(N), repeat = 2):
            cells[y][x] = power(x+1, y+1, serial)
        ps = [[0]*(N+1) for _ in range(N+1)]
        # BUILD 2D PREFIX SUM
        for r, c in itertools.product(range(1,N+1), repeat = 2):
            ps[r][c] = ps[r-1][c] + ps[r][c-1] + cells[r-1][c-1] - ps[r-1][c-1]
        # FIND MAX
        maxPower = -math.inf
        X, Y, S = 0, 0, 0
        for y, x in itertools.product(range(N), repeat = 2):
            for s in range(1, min(N-x, N-y)+1):
                power_level = ps[y+s][x+s] - ps[y][x+s] - ps[y+s][x] + ps[y][x]
                if power_level > maxPower:
                    maxPower = power_level
                    X, Y, S = x+1, y+1, s
        return ','.join(map(str, (X, Y, S)))

if __name__ == "__main__":
    print(main())