import numpy as np
with open("inputs/input.txt", "r") as f:
    positions = np.array(list(map(int, f.read().split(','))))
    minFuel = int(sum(n*(n+1)/2 for n in abs(positions - int(np.mean(positions)))))
    print(minFuel)
