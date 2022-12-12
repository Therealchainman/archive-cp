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
        X = Y = 300
        cells = [[0]*(X+1) for _ in range(Y+1)]
        for y, x in itertools.product(range(1, Y+1), range(1, X+1)):
            cells[y][x] = power(x, y, serial)
        res = max([(x, y) for y, x in itertools.product(range(1, Y-1), range(1, X-1))], key = lambda coord: sum((cells[coord[1]+i][coord[0]+j] for i, j in itertools.product(range(3), range(3)))))
        return ','.join(map(str, res))

if __name__ == "__main__":
    print(main())