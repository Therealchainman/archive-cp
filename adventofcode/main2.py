from collections import defaultdict, deque, Counter
from math import inf
import re
def main():
    with open('input.txt', 'r') as f:
        data = list(map(lambda coords: re.findall(r"\d+", coords), f.read().splitlines()))
        res = 0
        threshold = 10000
        for i in range(0, 400):
            for j in range(0, 400):
                dist = 0
                for pair in data:
                    x, y = map(int, pair)
                    dist += abs(x - i) + abs(y - j)
                res += (dist < threshold)
        return res
if __name__ == "__main__":
    print(main())