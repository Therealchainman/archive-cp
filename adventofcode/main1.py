from collections import *
import math
from itertools import *
def main():
    with open('input.txt', 'r') as f:
        data = f.read().splitlines()
        seen = set()
        head = [0,0]
        tail = [0,0]
        seen.add(tuple(tail))
        neighborhood = lambda x, y: [(x-1, y-1), (x-1, y), (x-1, y+1), (x, y-1), (x, y+1), (x+1, y-1), (x+1, y), (x+1, y+1), (x, y)]
        for line in data:
            dir_, x = line.split()
            x = int(x)
            if dir_ == 'R':
                for i in range(x):
                    head[0] += 1
                    if tuple(tail) in neighborhood(*head): continue
                    tail[0] += 1
                    if tail[1] > head[1]:
                        tail[1] -= 1
                    elif tail[1] < head[1]:
                        tail[1] += 1
                    seen.add(tuple(tail))
            elif dir_ == 'L':
                for i in range(x):
                    head[0] -= 1
                    if tuple(tail) in neighborhood(*head): continue
                    tail[0] -= 1
                    if tail[1] > head[1]:
                        tail[1] -= 1
                    elif tail[1] < head[1]:
                        tail[1] += 1
                    seen.add(tuple(tail))
            elif dir_ == 'U':
                for i in range(x):
                    head[1] += 1
                    if tuple(tail) in neighborhood(*head): continue
                    tail[1] += 1
                    if tail[0] > head[0]:
                        tail[0] -= 1
                    elif tail[0] < head[0]:
                        tail[0] += 1
                    seen.add(tuple(tail))
            else:
                for i in range(x):
                    head[1] -= 1
                    if tuple(tail) in neighborhood(*head): continue
                    tail[1] -= 1
                    if tail[0] > head[0]:
                        tail[0] -= 1
                    elif tail[0] < head[0]:
                        tail[0] += 1
                    seen.add(tuple(tail))
        return len(seen)
if __name__ == "__main__":
    print(main())