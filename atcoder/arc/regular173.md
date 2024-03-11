# Atcoder Regular Contest 173

## A - Neq Number 

### Solution 1:  binary search, digit dp

```py
M = 10 ** 18
def calc(x):
    dp = Counter({(-1, True, True): 1}) # (last digit, is_tight, is_zero) -> count
    for d in map(int, str(x)):
        ndp = Counter()
        for (last, tight, zero), cnt in dp.items():
            for i in range(10 if not tight else d + 1):
                if not zero and i == last: continue
                ndp[(i, tight and i == d, zero and i == 0)] += cnt
        dp = ndp
    ans = sum(cnt for (_, _, zero), cnt in dp.items() if not zero)
    return ans
def main():
    K = int(input())
    left, right = 0, M
    while left < right:
        mid = (left + right) >> 1
        if calc(mid) < K: left = mid + 1
        else: right = mid
    print(left)       

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## B - Make Many Triangles 

### Solution 1:  max number of points on a line, lines, geometry

```py
from itertools import product
def floor(x, y):
    return x // y
def main():
    N = int(input())
    points = [None] * N
    for i in range(N):
        x, y = map(int, input().split())
        points[i] = (x, y)
    line = 0
    for (x1, y1), (x2, y2) in product(points, repeat = 2):
        if (x1, y1) == (x2, y2): continue
        cnt = 2
        for x3, y3 in points:
            if (x3, y3) == (x1, y1) or (x3, y3) == (x2, y2): continue
            dx13, dy13, dx12, dy12 = x3 - x1, y3 - y1, x2 - x1, y2 - y1
            if dx13 * dy12 == dx12 * dy13: cnt += 1
        line = max(line, cnt)
    ans = min(floor(N, 3), N - line)
    print(ans)

if __name__ == '__main__':
    main()
```

## C - Not Median 

### Solution 1: 

```py

```

## D - Bracket Walk 

### Solution 1:  directed graph, strongly connected components, detect negative cycle, bellman ford algorithm

```py

```