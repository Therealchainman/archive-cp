from collections import namedtuple
with open("inputs/input.txt", "r") as f:
    Point = namedtuple("Point", ["x", "y"])
    data = f.read().splitlines()
    data = [x.split(" -> ") for x in data]
    data = [(Point(int(x[0].split(",")[0]), int(x[0].split(",")[1])), Point(int(x[1].split(",")[0]), int(x[1].split(",")[1]))) for x in data]
    matrix = [[0 for _ in range(1000)] for _ in range(1000)]
    for start, end in data:
        deltaX = 1 if end.x>start.x else -1 if end.x<start.x else 0
        deltaY = 1 if end.y>start.y else -1 if end.y<start.y else 0
        ix, iy = start.x, start.y
        while ix != end.x or iy != end.y:
            matrix[ix][iy] += 1
            ix += deltaX
            iy += deltaY
        matrix[end.x][end.y] += 1
    cnt = sum(1 for x in range(len(matrix)) for y in range(len(matrix[0])) if matrix[x][y]>=2)
    print(cnt)