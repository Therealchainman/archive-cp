# sys.stdout = open('outputs/output.txt', 'w')
with open("inputs/input.txt", "r") as f:
    data = []
    lines = f.read().splitlines()
    for line in lines:
        data.append([int(x) for x in line])
    R, C = len(data), len(data[0])
    sumRisk = sum(data[i][j] + 1 for i in range(R) for j in range(C) if all(data[i][j] < data[i+dr][j+dc] for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)) if 0 <= i+dr < R and 0 <= j+dc < C))
    print(sumRisk)
# sys.stdout.close()