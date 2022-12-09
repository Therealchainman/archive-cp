from collections import *
import math
from itertools import *
def main():
    with open('input.txt', 'r') as f:
        data = f.read().splitlines()
        seen = set([(0, 0)])
        knots = [[0]*2 for _ in range(10)]
        neighborhood = lambda x, y: [(x-1, y-1), (x-1, y), (x-1, y+1), (x, y-1), (x, y+1), (x+1, y-1), (x+1, y), (x+1, y+1), (x, y)]
        for line in data:
            dir_, x = line.split()
            x = int(x)
            if dir_ == 'R':
                for _ in range(x):
                    knots[0][0] += 1
                    for i in range(1, 10):
                        if tuple(knots[i]) in neighborhood(*knots[i-1]): continue
                        if knots[i][0] > knots[i-1][0]:
                            knots[i][0] -= 1
                        if knots[i][0] < knots[i-1][0]:
                            knots[i][0] += 1
                        if knots[i][1] > knots[i-1][1]:
                            knots[i][1] -= 1
                        if knots[i][1] < knots[i-1][1]:
                            knots[i][1] += 1
                    seen.add(tuple(knots[-1]))
            elif dir_ == 'L':
                for j in range(x):
                    knots[0][0] -= 1
                    for i in range(1, 10):
                        if tuple(knots[i]) in neighborhood(*knots[i-1]): continue
                        if knots[i][0] > knots[i-1][0]:
                            knots[i][0] -= 1
                        if knots[i][0] < knots[i-1][0]:
                            knots[i][0] += 1
                        if knots[i][1] > knots[i-1][1]:
                            knots[i][1] -= 1
                        if knots[i][1] < knots[i-1][1]:
                            knots[i][1] += 1
                    seen.add(tuple(knots[-1]))
            elif dir_ == 'U':
                for _ in range(x):
                    knots[0][1] -= 1
                    for i in range(1, 10):
                        if tuple(knots[i]) in neighborhood(*knots[i-1]): continue
                        if knots[i][0] > knots[i-1][0]:
                            knots[i][0] -= 1
                        if knots[i][0] < knots[i-1][0]:
                            knots[i][0] += 1
                        if knots[i][1] > knots[i-1][1]:
                            knots[i][1] -= 1
                        if knots[i][1] < knots[i-1][1]:
                            knots[i][1] += 1
                    seen.add(tuple(knots[-1]))
            else:
                for _ in range(x):
                    knots[0][1] += 1
                    for i in range(1, 10):
                        if tuple(knots[i]) in neighborhood(*knots[i-1]): continue
                        if knots[i][0] > knots[i-1][0]:
                            knots[i][0] -= 1
                        if knots[i][0] < knots[i-1][0]:
                            knots[i][0] += 1
                        if knots[i][1] > knots[i-1][1]:
                            knots[i][1] -= 1
                        if knots[i][1] < knots[i-1][1]:
                            knots[i][1] += 1
                    seen.add(tuple(knots[-1]))
        return len(seen)
if __name__ == "__main__":
    print(main())