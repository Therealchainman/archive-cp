# Atcoder Beginner Contest 344

## C - A+B+C

### Solution 1:  hash table, set

```py
from itertools import product
def main():
    N = int(input())
    A = list(map(int, input().split()))
    M = int(input())
    B = list(map(int, input().split()))
    L = int(input())
    C = list(map(int, input().split()))
    Q = int(input())
    queries = list(map(int, input().split()))
    sums = set()
    for a, b, c in product(A, B, C):
        sums.add(a + b + c)
    for i in range(Q):
        print("Yes" if queries[i] in sums else "No")
```

## D - String Bags

### Solution 1:  kmp, dp

```py
import math
def kmp(s):
    n = len(s)
    pi = [0] * n
    for i in range(1, n):
        j = pi[i - 1]
        while j > 0 and s[i] != s[j]: 
            j = pi[j - 1]
        if s[j] == s[i]: j += 1
        pi[i] = j
    return pi

def main():
    T = input()
    n = len(T)
    N = int(input())
    bags = [None] * N
    for i in range(N):
        bags[i] = list(map(str, input().split()[1:]))
    dp = [math.inf] * (n + 1)
    dp[0] = 0 # empty string
    for bag in bags: # 100
        ndp = dp.copy()
        for s in bag: # 10
            ns = len(s)
            parr = kmp(s + "#" + T)[ns:] # 110
            for i in range(n + 1): # 100
                if parr[i] == ns:
                    ndp[i] = min(ndp[i], dp[i - ns] + 1)
        dp = ndp
    print(dp[-1] if dp[-1] < math.inf else -1)
    
if __name__ == '__main__':
    main()
```

## E - Insert or Erase

### Solution 1:  doubly linked list, insert, erase, nxt and prv pointers

```py
from collections import defaultdict
def main():
    N = int(input())
    arr = list(map(int, input().split()))
    prv = defaultdict(lambda: None)
    nxt = defaultdict(lambda: None)
    Q = int(input())
    for i in range(N):
        if i > 0:
            prv[arr[i]] = arr[i - 1]
        if i + 1 < N:
            nxt[arr[i]] = arr[i + 1]
    start = arr[0]
    def erase(x):
        prv[nxt[x]] = prv[x]
        nxt[prv[x]] = nxt[x]
    def insert(x, y): # insert y after x
        nxt[y] = nxt[x]
        prv[y] = x
        prv[nxt[x]] = y
        nxt[x] = y
    for _ in range(Q):
        query = list(map(int, input().split()))
        if query[0] == 1:
            x, y = query[1:]
            insert(x, y) # insert y after x
        else:
            x = query[1]
            if start == x:
                start = nxt[x]
            erase(x)
    ans = []
    while start is not None:
        ans.append(start)
        start = nxt[start]
    print(*ans)

if __name__ == '__main__':
    main()
```

## F - Earn to Advance 

### Solution 1:  dp, min actions, dp on grid

```py
from collections import defaultdict
from itertools import product
import math

def ceil(x, y):
    return (x + y - 1) // y 

def main():
    N = int(input())
    P = [list(map(int, input().split())) for _ in range(N)]
    R = [list(map(int, input().split())) for _ in range(N)]
    D = [list(map(int, input().split())) for _ in range(N)]
    dp = [[defaultdict(lambda: (math.inf, 0)) for _ in range(N)] for _ in range(N)]
    dp[0][0][P[0][0]] = (0, 0) # (action, money)
    for r, c in product(range(N), repeat = 2):
        if r > 0: # move down
            for payer, (actions, money) in dp[r - 1][c].items():
                need = max(0, D[r - 1][c] - money)
                take = ceil(need, payer)
                npayer = max(payer, P[r][c])
                nmoney = money - D[r - 1][c] + take * payer
                naction = actions + take + 1
                if naction < dp[r][c][npayer][0]: dp[r][c][npayer] = (naction, nmoney)
                elif naction == dp[r][c][npayer][0] and nmoney > dp[r][c][npayer][1]: dp[r][c][npayer] = (naction, nmoney)
        if c > 0: # move right
            for payer, (actions, money) in dp[r][c - 1].items():
                need = max(0, R[r][c - 1] - money)
                take = ceil(need, payer)
                npayer = max(payer, P[r][c])
                nmoney = money - R[r][c - 1] + take * payer
                naction = actions + take + 1
                if naction < dp[r][c][npayer][0]: dp[r][c][npayer] = (naction, nmoney)
                elif naction == dp[r][c][npayer][0] and nmoney > dp[r][c][npayer][1]: dp[r][c][npayer] = (naction, nmoney)
    print(min([x for x, _ in dp[-1][-1].values()]))

if __name__ == '__main__':
    main()
```

## G - Points and Comparison 

### Solution 1:  lines

```py

```
