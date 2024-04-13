# Olympiads School CCO Training Contest 1

## Permutation Sorting

### Solution 1:  

```py
def main():
    n, q = map(int, input().split())
    arr = list(map(int, input().split()))
    for _ in range(q):
        t, l, r = map(int, input().split())
        l -= 1
        arr[l : r] = sorted(arr[l : r], reverse = t)
    p = int(input())
    p -= 1
    print(arr[p])
if __name__ == "__main__":
    main()
```

## Tree Journey

### Solution 1:  tree, bfs, max height, math pattern for tree

```py
from collections import deque
def main():
    N, K = map(int, input().split())
    adj = [[] for _ in range(N)]
    for _ in range(N - 1):
        u, v = map(int, input().split())
        u -= 1; v -= 1
        adj[u].append(v)
        adj[v].append(u)
    H = -1
    q = deque([(0, -1)])
    while q:
        for _ in range(len(q)):
            u, p = q.popleft()
            for v in adj[u]:
                if v == p: continue
                q.append((v, u))
        H += 1
    ans = min(H, K) + 1
    K = max(0, K - H)
    ans = min(N, ans + K // 2)
    print(ans)
if __name__ == "__main__":
    main()
```

## Palindrome Path

### Solution 1:  

```py
from itertools import product
def main():
    N, M, Q = map(int, input().split())
    S = list(map(int, input()))
    adj = [[[] for _ in range(N)] for _ in range(2)]
    for _ in range(M):
        u, v = map(int, input().split())
        u -= 1; v -= 1
        adj[S[v]][u].append(v)
        adj[S[u]][v].append(u)
    # precompute answer for every pair of nodes (u, v)
    vis = [[False] * N for _ in range(N)]
    dp = [[False] * N for _ in range(N)]
    for i in range(N):
        dp[i][i] = True # base case
        vis[i][i] = True
    for i in range(N):
        for j in adj[S[i]][i]:
            dp[i][j] = True
            vis[i][j] = True
    stk = []
    for i, j in product(range(N), repeat = 2):
        if dp[i][j]: stk.append((i, j))
    """
    O(N^2M)
    """
    while stk: # O(N^2 + M) how?
        u, v = stk.pop()
        for b in range(2):
            for x in adj[b][u]:
                # can I remove this loop?
                for y in adj[b][v]:
                    if vis[x][y]: continue
                    dp[x][y] = True
                    stk.append((x, y))
                    vis[x][y] = True
    for _ in range(Q):
        u, v = map(int, input().split())
        u -= 1; v -= 1
        if dp[u][v]:
            print("YES")
        else:
            print("NO")
if __name__ == "__main__":
    main()
```