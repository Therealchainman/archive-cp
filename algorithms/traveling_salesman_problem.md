# Traveling Salesman Problem

This problem is too famous to not include special algorithms for it.  The statement is to find the minimum edge weight to travel to every vertex at least once. 

## Directed Graphs

## Negative weighted edges

If there exists negative weighted cycle, there isn't really a solution, since you can achieve negative infinity as minimal cost.
You can use Floyd Warshal algorithm to precompute the shortest path for all pairs of nodes. Then you and implement the standard dp solution for traveling salesman problem.

O(n^3 + n^2*2^n)

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
    # dp for tsp O(n^2*2^n)
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
```

## dp bitmask to find if path exists ending at node u with set of nodes seen

```py
dp = [[False] * N for _ in range(1 << N)]
for i in range(N):
    dp[1 << i][i] = True # start from each node
for mask in range(1 << N):
    # travel from u to v
    for u in range(N):
        if not dp[mask][u]: continue 
        for v in adj[u]:
            if (mask >> v) & 1: continue
            dp[mask | (1 << v)][v] = dp[mask][u]
```