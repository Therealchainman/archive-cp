# Atcoder Beginner Contest 338

## C - Leftover Recipes 

### Solution 1:  math, division

```py
import math

def main():
    N = int(input())
    Q = list(map(int, input().split()))
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    cnt_a = ans = 0
    while True:
        cnt_b = math.inf
        for i in range(N):
            if B[i] == 0: continue
            cnt_b = min(cnt_b, (Q[i] // B[i]))
        ans = max(ans, cnt_a + cnt_b)
        cnt_a += 1
        flag = True
        for i in range(N):
            Q[i] -= A[i]
            if Q[i] < 0: 
                flag = False
        if not flag: break
    print(ans)

if __name__ == '__main__':
    main()
```

## D - Island Tour 

### Solution 1:  modified prefix sums, observe that only change when cut separates two sections

```py
def main():
    N, M = map(int, input().split())
    arr = list(map(lambda x: int(x) - 1, input().split()))
    adj = [Counter() for _ in range(N)]
    cur = 0
    for i in range(M - 1):
        u, v = arr[i], arr[i + 1]
        adj[u][v] += 1
        adj[v][u] += 1
        cur += abs(u - v)
    ans = cur
    for r in range(N - 2):
        for u in adj[r]:
            if u > r:
                old = u - r
                new = N - u + r
                cur += (new - old) * adj[r][u]
            elif u < r:
                old = N - r + u
                new = r - u
                cur += (new - old) * adj[r][u]
        ans = min(ans, cur)
    print(ans)

if __name__ == '__main__':
    main()
```

## E - Chords 

### Solution 1:  stack, chords on circle

```py
UNVISITED = -1
def main():
    N = int(input())
    A = [UNVISITED] * (2 * N + 1)
    B = [UNVISITED] * (2 * N + 1)
    for i in range(N):
        u, v = map(int, input().split())
        if u > v: u, v = v, u
        A[u] = i
        B[v] = i
    stk = []
    for i in range(1, 2 * N + 1):
        if A[i] != UNVISITED: stk.append(A[i])
        else:
            if not stk: return print("Yes")
            v = stk.pop()
            if B[i] != v: return print("Yes")
    print("No")

if __name__ == '__main__':
    main()
```

## F - Negative Traveling Salesman

### Solution 1:  traveling salesman problem, dynamic programming, floyd warshall, all pairs shortest path

```py
import math
def main():
    N, M = map(int, input().split())
    dist = [[math.inf] * N for _ in range(N)]
    for i in range(N):
        dist[i][i] = 0
    for _ in range(M):
        u, v, w = map(int, input().split())
        u -= 1; v -= 1
        dist[u][v] = w
    end_mask = (1 << N) - 1
    # floyd warshall, all pairs shortest path
    for i in range(N):
        for j in range(N):
            for k in range(N):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    # dp for tsp
    dp = [[math.inf] * N for _ in range(1 << N)]
    for start in range(N):
        dp[1 << start][start] = 0
    for mask in range(1 << N):
        for u in range(N):
            if dp[mask][u] == math.inf: continue
            for v in range(N):
                if (mask >> v) & 1: continue
                nmask = mask | (1 << v)
                dp[nmask][v] = min(dp[nmask][v], dp[mask][u] + dist[u][v])
    ans = min(dp[end_mask])
    print(ans if ans < math.inf else "No")

if __name__ == '__main__':
    main()
```

## 

### Solution 1: 

```py

```

