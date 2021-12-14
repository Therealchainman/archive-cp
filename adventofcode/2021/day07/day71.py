import numpy as np
# sys.stdout = open('outputs/output.txt', 'w')
with open("inputs/input.txt", "r") as f:
    positions = list(map(int, f.read().split(',')))
    minFuel = int(abs(positions - np.median(positions)).sum())
    print(minFuel)
# sys.stdout.close()