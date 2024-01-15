# Atcoder Beginner Contest 335

## C - Loong Tracking

### Solution 1:  stack

```py
dirs = {
    "U": (0, 1),
    "D": (0, -1),
    "R": (1, 0),
    "L": (-1, 0)
}

def main():
    N, Q = map(int, input().split())
    body = [(i, 0) for i in reversed(range(1, N + 1))]
    for _ in range(Q):
        t, v = input().split()
        if t == "1":
            dr, dc = dirs[v]
            r, c = body[-1]
            body.append((r + dr, c + dc))
        else:
            print(*body[-int(v)])

if __name__ == '__main__':
    main()
```

## D - Loong and Takahashi

### Solution 1:  spiral, grid

```py
def main():
    N = int(input())
    grid = [[0] * N for _ in range(N)]
    grid[N // 2][N // 2] = "T"
    v = 1
    for i in range(N // 2):
        # top row 
        for c in range(N):
            if grid[i][c] != 0: continue
            grid[i][c] = v
            v += 1
        # right column
        for r in range(N):
            if grid[r][N - i - 1] != 0: continue
            grid[r][N - i - 1] = v
            v += 1
        # bottom row
        for c in reversed(range(N)):
            if grid[N - i - 1][c] != 0: continue 
            grid[N - i - 1][c] = v 
            v += 1
        # left column
        for r in reversed(range(N)):
            if grid[r][i] != 0: continue
            grid[r][i] = v 
            v += 1
    for row in grid:
        print(*row)

if __name__ == '__main__':
    main()
```

## E - Non-Decreasing Colorful Path

### Solution 1:   priority queue, undirected graph

```py
from heapq import heappop, heappush

def main():
    N, M = map(int, input().split())
    arr = list(map(int, input().split()))
    adj = [[] for _ in range(N)]
    for _ in range(M):
        u, v = map(int, input().split())
        u -= 1; v -= 1
        adj[u].append(v)
        adj[v].append(u)
    vis = [0] * N
    min_heap = [(arr[0], -1, 0)]
    while min_heap:
        _, score, u = heappop(min_heap)
        score = -score
        if u == N - 1: return print(score)
        if vis[u]: continue
        vis[u] = 1
        for v in adj[u]:
            if arr[v] < arr[u]: continue
            nscore = score + (1 if arr[v] > arr[u] else 0)
            heappush(min_heap, (arr[v], -nscore, v))
    print(0)

if __name__ == '__main__':
    main()
```

## F - Hop Sugoroku

### Solution 1:  DP, square root 

```py
mod = 998244353
def main():
    N = int(input())
    arr = list(map(int, input().split()))
    dp = [0] * N
    deltas = [0] * N
    dp[0] = 1
    for i in range(N):
        x = (dp[i] + deltas[i]) % mod
        for j in range(i + arr[i], N, arr[i]):
            dp[j] = (dp[j] + x) % mod
            if arr[i] == arr[j]:
                deltas[j] = (deltas[j] + x) % mod
                break
    print(sum(dp) % mod)

if __name__ == '__main__':
    main()
```

## G - Discrete Logarithm Problems 

### Solution 1:  Number theory, modular arithmetic, fermat's little theorem, gcd, primitive roots, multiplicative order and additive order

```py
import math

# collect all the prime factors
def prime_factors(num: int) -> List[int]:
    factors = []
    while num % 2 == 0:
        if not factors or (factors and factors[-1] != 2):
            factors.append(2)
        num //= 2
    for i in range(3, math.isqrt(num) + 1, 2):
        while num % i == 0:
            if not factors or (factors and factors[-1] != i):
                factors.append(i)
            num //= i
    if num > 2:
        factors.append(num)
    return factors

def main():
    N, P = map(int, input().split())
    pf = prime_factors(P - 1)
    arr = list(map(int, input().split()))
    mult_orders = [P - 1] * N
    for i in range(N):
        while mult_orders[i] > 1:
            found = False
            for pfactor in pf:
                if pfactor > mult_orders[i] or mult_orders[i] % pfactor: continue
                x = mult_orders[i] // pfactor
                if pow(arr[i], x, P) == 1:
                    mult_orders[i] = x
                    found = True
                    break
            if not found: break
    mult_pairs = Counter()
    for i in range(N):
        mult_pairs[mult_orders[i]] += 1
    mult_pairs = list(mult_pairs.items())
    ans = 0
    for i in range(len(mult_pairs)):
        k1, v1 = mult_pairs[i]
        for j in range(i, len(mult_pairs)):
            k2, v2 = mult_pairs[j]
            if k1 % k2 == 0 or k2 % k1 == 0:
                ans += v1 * v2
    print(ans)

if __name__ == '__main__':
    main()
```
