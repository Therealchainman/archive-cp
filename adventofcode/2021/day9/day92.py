# sys.stdout = open('outputs/output.txt', 'w')
from collections import deque
def bfs(input, i, j):
    dq = deque([(i, j)])
    vis = set()
    vis.add((i, j))
    sz = 0
    # print("call bfs")
    while len(dq) > 0:
        i, j = dq.popleft()
        sz += 1
        a = ord(input[i][j]) - ord('0')
        for dr in range(-1,2):
            for dc in range(-1,2):
                if abs(dc-dr) != 1: continue
                nr = i + dr
                nc = j + dc
                if nr >= 0 and nr < len(input) and nc >= 0 and nc < len(input[nr]) and (nr,nc) not in vis:
                    b = ord(input[nr][nc]) - ord('0')
                    # print(f"a: {a}, b: {b}")
                    if b == 9: continue
                    if b > a:
                        vis.add((nr,nc))
                        dq.append((nr,nc))
    return sz
with open("inputs/input.txt", "r") as f:
    input = []
    while True:
        line = f.readline().replace('\n', '')
        if not line:
            break
        input.append(line)
    basins = []
    for i in range(len(input)):
        for j in range(len(input[i])):
            works = True
            for dr in range(-1,2):
                for dc in range(-1,2):
                    if abs(dc-dr) != 1: continue
                    nr = i + dr
                    nc = j + dc
                    if nr >= 0 and nr < len(input) and nc >= 0 and nc < len(input[nr]):
                        if input[nr][nc] <= input[i][j]:
                            works = False
                            break
            if works:
                basins.append(bfs(input, i, j))

    basins.sort()
    print(basins)
    print(basins[-1]*basins[-2]*basins[-3])
# sys.stdout.close()