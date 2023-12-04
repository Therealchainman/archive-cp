# Atcoder Beginner Contest 320

## A - Leyland Number

### Solution 1:  math

```py
def main():
    a, b = map(int, input().split())
    print(a**b + b**a)

if __name__ == '__main__':
    main()
```

## B - Longest Palindrome

### Solution 1:  is palindrome + two pointers

```py
def main():
    s = input()
    n = len(s)
    res = 1
    def is_palindrome(part):
        left, right = 0, len(part) - 1
        while left < right and part[left] == part[right]:
            left += 1
            right -= 1
        return left >= right
    for i in range(n):
        for j in range(i + 1, n + 1):
            cur = s[i : j]
            if is_palindrome(cur):
                res = max(res, len(cur))
    print(res)

if __name__ == '__main__':
    main()
```

## C - Slot Strategy 2 (Easy)

### Solution 1:  brute force

```py
import math
from itertools import product

def main():
    m = int(input())
    s1, s2, s3 = list(map(int, input())), list(map(int, input())), list(map(int, input()))
    res = math.inf
    for dig in range(10):
        vals = [[] for _ in range(3)]
        for i in range(3 * m):
            if s1[i % m] == dig and len(vals[0]) < 3:
                vals[0].append(i)
            if s2[i % m] == dig and len(vals[1]) < 3:
                vals[1].append(i)
            if s3[i % m] == dig and len(vals[2]) < 3:
                vals[2].append(i)
        if not len(vals[0]) == len(vals[1]) == len(vals[2]) == 3: continue
        for i, j, k in product(range(3), repeat = 3):
            nums = set([vals[0][i], vals[1][j], vals[2][k]])
            if len(nums) < 3: continue
            res = min(res, max(nums))
    print(res if res != math.inf else -1)

if __name__ == '__main__':
    main()
```

## D - Relative Position

### Solution 1:  weighted undirected graph + dfs

```py
def main():
    n, m = map(int, input().split())
    pos = [None] * n
    pos[0] = (0, 0)
    adj_list = [[] for _ in range(n)]
    for _ in range(m):
        u, v, x, y = map(int, input().split())
        u -= 1
        v -= 1
        adj_list[u].append((v, x, y))
        adj_list[v].append((u, -x, -y))
    stack = [0]
    while stack:
        u = stack.pop()
        x, y = pos[u]
        for v, dx, dy in adj_list[u]:
            if pos[v] is not None: continue
            nx, ny = x + dx, y + dy
            pos[v] = (nx, ny)
            stack.append(v)
    for i in range(n):
        if pos[i] is None:
            print("undecidable")
        else:
            print(*pos[i])
        
if __name__ == '__main__':
    main()
```

## E - Somen Nagashi

### Solution 1:  heap + greedy

```py
from heapq import heappush, heappop, heapify

def main():
    n, m = map(int, input().split())
    people = list(range(n))
    heapify(people)
    res = [0] * n
    free = []
    for _ in range(m):
        t, w, s = map(int, input().split())
        while free and free[0][0] <= t:
            _, u = heappop(free)
            heappush(people, u)
        if not people: continue
        u = heappop(people)
        res[u] += w
        heappush(free, (t + s, u))
    for i in range(n):
        print(res[i])

if __name__ == '__main__':
    main()
```

## F - Fuel Round Trip

### Solution 1:  dynamic programming

```py
import math

def main():
    N, H = map(int, input().split())
    pos = list(map(int, input().split()))
    stations = [None] * (N - 1)
    for i in range(N - 1):
        p, f = map(int, input().split())
        stations[i] = (p, f)
    dp = [[math.inf] * (H + 1) for _ in range(H + 1)] # j, k
    if pos[0] > H: return print(-1)
    for k in range(pos[0], H + 1):
        dp[H - pos[0]][k] = 0
    for i in range(N - 1):
        ndp = [[math.inf] * (H + 1) for _ in range(H + 1)]
        d = pos[i + 1] - pos[i]
        for j in range(H + 1):
            for k in range(H + 1):
                if dp[j][k] == math.inf: continue
                p, f = stations[i]
                # refuel none
                if k + d <= H and j - d >= 0:
                    nj, nk = j - d, k + d
                    ndp[nj][nk] = min(ndp[nj][nk], dp[j][k])
                # refuel j
                if k + d <= H:
                    nj, nk = min(j + f, H) - d, k + d
                    if nj >= 0:
                        ndp[nj][nk] = min(ndp[nj][nk], dp[j][k] + p)
                # refuel k
                if j - d >= 0 and k >= f and k - f + d <= H:
                    nj, nk = j - d, k - f + d
                    ndp[nj][nk] = min(ndp[nj][nk], dp[j][k] + p)
                    if k == H:
                        for w in range(nk + 1, H + 1):
                            ndp[nj][w] = min(ndp[nj][w], dp[j][k] + p)
        dp = ndp
    res = min(dp[i][i] for i in range(H + 1))
    print(res if res != math.inf else -1)

if __name__ == '__main__':
    main()
```

## G - Slot Strategy 2 (Hard)

### Solution 1:  graph matching + bipartite graph matching + graph theory

```py

```

