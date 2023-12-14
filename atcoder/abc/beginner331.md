# Atcoder Beginner Contest 331

## D. Tile Pattern

### Solution 1:  2D prefix sum + periodicity

```py
from itertools import product

def main():
    N, Q = map(int, input().split())
    grid = [list(input()) for _ in range(N)]
    psum = [[0] * (N + 1) for _ in range(N + 1)]
    for r, c in product(range(1, N + 1), repeat = 2):
        psum[r][c] = psum[r - 1][c] + psum[r][c - 1] - psum[r - 1][c - 1] + (grid[r - 1][c - 1] == "B")
    def g(r, c):
        r_span, c_span = r // N, c // N
        return (
            psum[N][N] * r_span * c_span
            + psum[N][c % N] * r_span
            + psum[r % N][N] * c_span
            + psum[r % N][c % N]
        )
    def f(r1, c1, r2, c2):
        return g(r2, c2) - g(r1, c2) - g(r2, c1) + g(r1, c1)
    for _ in range(Q):
        r1, c1, r2, c2 = map(int, input().split())
        print(f(r1, c1, r2 + 1, c2 + 1))

if __name__ == '__main__':
    main()
```

## E - Set Meal 

### Solution 1:  hash map, sort, offline query

```py
def main():
    N, M, L = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    sides = sorted(range(M), key = lambda i: B[i], reverse = True)
    bad_combos = [set() for _ in range(N)]
    for _ in range(L):
        c, d = map(int, input().split())
        c -= 1
        d -= 1
        bad_combos[c].add(d)
    ans = 0
    for i in range(N):
        for j in sides:
            if j not in bad_combos[i]:
                ans = max(ans, A[i] + B[j])
                break
    print(ans)
    
if __name__ == '__main__':
    main()
```

## 

### Solution 1: 

```py

```

## 

### Solution 1: 

```py

```