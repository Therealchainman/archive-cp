from collections import deque
# sys.stdout = open('outputs/output.txt', 'w')
with open("inputs/input.txt", "r") as f:
    data = []
    lines = f.read().splitlines()
    for line in lines:
        data.append([int(x) for x in line])
    R, C = len(data), len(data[0])
    step = 0
    while True:
        num = 0
        step += 1
        dq = deque()
        for i in range(R):
            for j in range(C):
                data[i][j] += 1
                if data[i][j] > 9:
                    dq.append((i, j))
                    data[i][j] = 0
        while dq:
            i, j = dq.popleft()
            num += 1
            for dr in range(-1,2):
                for dc in range(-1,2):
                    if dr==0 and dc==0:
                        continue
                    nr = i+dr
                    nc = j+dc
                    if (0 <= nr < R) and (0 <= nc < C) and data[nr][nc] > 0:
                        data[nr][nc] += 1
                        if data[nr][nc] > 9:
                            dq.append((nr, nc))
                            data[nr][nc] = 0
        if num == R*C:
            break
    print(step)