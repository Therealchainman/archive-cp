# sys.stdout = open('outputs/output.txt', 'w')
with open("inputs/input.txt", "r") as f:
    input = []
    while True:
        line = f.readline().replace('\n', '')
        if not line:
            break
        input.append(line)
    res = 0
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
                res += ord(input[i][j]) - ord('0') + 1
    print(res)
# sys.stdout.close()