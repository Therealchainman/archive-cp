# Codeforces Round 933 Div 3

## B. Rudolf and 121

### Solution 1:  greedy, reconstruct array

```py
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    ans = [0] * n
    for i in range(n - 2):
        delta = max(0, arr[i] - ans[i])
        ans[i] += delta
        ans[i + 1] += 2 * delta
        ans[i + 2] += delta
    print("YES" if arr == ans else "NO")

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## C. Rudolf and the Ugly String

### Solution 1:  inclusion exclusion principle, count map and pie and remove double counts

```py
def main():
    n = int(input())
    s = input()
    ans = s.count("map") + s.count("pie") - s.count("mapie")
    print(ans)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()

```

## D. Rudolf and the Ball Game

### Solution 1:  simulation, vis array

```py
def main():
    n, m, x = map(int, input().split())
    players = [0] * n
    players[x - 1] = 1
    for _ in range(m):
        r, c = input().split()
        r = int(r)
        nplayers = [0] * n
        for i in range(n):
            if not players[i]: continue 
            if c == "0":  nplayers[(i + r) % n] = 1
            elif c == "1": nplayers[(i - r) % n] = 1
            else: 
                nplayers[(i - r) % n] = 1
                nplayers[(i + r) % n] = 1
        players = nplayers
    print(sum(players))
    ans = [i + 1 for i in range(n) if players[i]]
    print(*ans)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## E. Rudolf and k Bridges

### Solution 1:  prefix sum, monotonic queue dp, sliding window dp

```py
from collections import deque
import math
def main():
    n, m, k, d = map(int, input().split())
    grid = [list(map(int, input().split())) for _ in range(n)]
    dp = [[0] * m for _ in range(n)]
    psum = [0] * n
    for i in range(n):
        dp[i][0] = 1
        monoq = deque([0])
        for j in range(1, m):
            while monoq and j - monoq[0] - 1 > d: monoq.popleft()
            dp[i][j] = dp[i][monoq[0]] + grid[i][j] + 1
            while monoq and dp[i][j] <= dp[i][monoq[-1]]: monoq.pop()
            monoq.append(j)
        psum[i] = dp[i][-1]
        if i > 0: psum[i] += psum[i - 1]
    ans = math.inf
    for i in range(k - 1, n):
        cur = psum[i]
        if i - k >= 0: cur -= psum[i - k]
        ans= min(ans, cur)
    print(ans)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## F. Rudolf and Imbalance

### Solution 1:  binary search, greedy, find max difference

```py
import bisect
import math

def main():
    n, m, k = map(int, input().split())
    A = list(map(int, input().split()))
    D = list(map(int, input().split()))
    F = sorted(map(int, input().split()))
    p = max(range(1, n), key = lambda i: A[i] - A[i - 1]) if n > 1 else -1
    val = (A[p] + A[p - 1]) // 2 if n > 1 else A[0]
    def update(rating):
        nonlocal ans
        if n == 1:
            ans = min(ans, abs(rating - A[0]))
            return
        ans = min(ans, max(abs(rating - A[p]), abs(rating - A[p - 1])))
    ans = A[p] - A[p - 1] if n > 1 else math.inf
    for d in D:
        i = bisect.bisect_right(F, val - d)
        rating = d + F[i] if i < k else math.inf
        update(rating)
        if i > 0: 
            i -= 1
            update(d + F[i])
    for i in range(1, n):
        if i == p: continue
        ans = max(ans, A[i] - A[i - 1])
    print(ans)


if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## G. Rudolf and Subway

### Solution 1:  bfs, create new unweighted undirected graph, find shortest path from source to destination 

```py
from collections import deque
def main():
    n, m = map(int, input().split())
    edges = [None] * m
    compress_idx = {}
    for i in range(m):
        u, v, c = map(int, input().split())
        u -= 1; v -= 1
        edges[i] = (u, v, c)
        if c not in compress_idx: compress_idx[c] = len(compress_idx)
    src, dst = map(int, input().split())
    src -= 1; dst -= 1
    N = n + len(compress_idx)
    adj = [[] for _ in range(N)]
    for u, v, col in edges:
        w = compress_idx[col] + n
        adj[u].append(w)
        adj[w].append(u)
        adj[v].append(w)
        adj[w].append(v)
    q = deque([src])
    vis = [0] * N
    dist = 0
    while q:
        for _ in range(len(q)):
            u = q.popleft()
            if u == dst: return print(dist // 2)
            if vis[u]: continue
            vis[u] = 1
            for v in adj[u]:
                if not vis[v]: q.append(v)
        dist += 1
    
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

