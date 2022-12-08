from collections import Counter
from itertools import product
def main():
    with open('input.txt', 'r') as f:
        data = []
        lines = f.read().splitlines()
        for line in lines:
            data.append([int(x) for x in line])
        n = len(data)
        leftVis, rightVis, aboveVis, belowVis = Counter(), Counter(), Counter(), Counter()
        for r in range(n):
            stack = []
            for c in range(n):
                while stack and data[r][c] >= data[r][stack[-1]]:
                    prev = stack.pop()
                    rightVis[(r, prev)] = c - prev
                stack.append(c)
            while stack:
                prev = stack.pop()
                rightVis[(r, prev)] = n - prev - 1
            stack = []
            for c in reversed(range(n)):
                while stack and data[r][c] >= data[r][stack[-1]]:
                    prev = stack.pop()
                    leftVis[(r, prev)] = prev - c
                stack.append(c)
            while stack:
                prev = stack.pop()
                leftVis[(r, prev)] = prev 
        for c in range(n):
            stack = []
            for r in range(n):
                while stack and data[r][c] >= data[stack[-1]][c]:
                    prev = stack.pop()
                    belowVis[(prev, c)] = r - prev
                stack.append(r)
            while stack:
                prev = stack.pop()
                belowVis[(prev, c)] = n - prev - 1
            stack = []
            for r in reversed(range(n)):
                while stack and data[r][c] >= data[stack[-1]][c]:
                    prev = stack.pop()
                    aboveVis[(prev, c)] = prev - r
                stack.append(r)
            while stack:
                prev = stack.pop()
                aboveVis[(prev, c)] = prev
        return max(leftVis[(r, c)] * rightVis[(r, c)] * belowVis[(r, c)] * aboveVis[(r, c)] for r, c in product(range(n), repeat = 2))
if __name__ == "__main__":
    print(main())