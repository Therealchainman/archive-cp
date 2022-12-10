from collections import *
from itertools import *
def main():
    with open('input.txt', 'r') as f:
        data = f.read().splitlines()
        arr = (20, 60, 100, 140, 180, 220)
        cycle = val = 1
        res = 0
        for ins in data:
            if ins == 'noop':
                cycle += 1
                if cycle in arr:
                    res += cycle*val
            else:
                _, delta = ins.split()
                delta = int(delta)
                cycle += 1
                if cycle in arr:
                    res += cycle*val
                cycle += 1
                val += delta
                if cycle in arr:
                    res += cycle*val
        return res
if __name__ == "__main__":
    print(main())