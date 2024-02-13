# Yet Another Contest 8

## Permutation Sorting

### Solution 1:  prefix max, greedy

minimize range of sorting, the only elements that don't need to be sorted are those at their corresponding index, and no element prior belongs after it.

```py
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    pmax = cnt = 0
    for i, num in enumerate(arr, start = 1):
        if num == i and pmax < num: cnt += 1
        pmax = max(pmax, num)
    ans = n - cnt
    print(ans)
if __name__ == '__main__':
    main()
```

## No More Modern Art

### Solution 1:  counter, xor

If you do the math the xors cancel and you are really just trying to find if num ^ x exists in the array.  If it does then you can xor num with that element to get x. 

recall a ^ b = x => a ^ x = b, so we are just finding if ths b exists in the array.

```py
from collections import Counter
def main():
    n, x = map(int, input().split())
    arr = list(map(int, input().split()))
    counts = Counter(arr)
    for num in arr:
        counts[num] -= 1
        target = num ^ x
        if counts[target] > 0: return print("YES")
        counts[num] += 1
    print("NO")
if __name__ == '__main__':
    main()
```

## Herobrine

### Solution 1: 

I'm very curious about the approach to this problem!

```py
from collections import Counter
def main():
    n = int(input())
    parent = list(map(int, input().split()))
    adj = [[] for _ in range(n + 1)]
    for i in range(n):
        adj[parent[i]].append(i + 1)
        adj[i + 1].append(parent[i])
    counts = [Counter() for _ in range(n + 1)]
    for i in range(1, n + 1):
        lst = list(map(int, input().split()))
        for ore in lst[1:]: counts[i][ore] += 1
    ans = [0] * (n + 1)    
    def dfs(u, p):
        for v in adj[u]:
            if v == p: continue
            dfs(v, u)
            for k, cnt in counts[v].items():
                counts[u][k] += cnt
        scount = counts[u].most_common()
        while scount:
            ans[u] = max(ans[u], len(scount) * scount[-1][1])
            scount.pop()
    dfs(0, -1)
    for x in ans[1:]: print(x)
    
if __name__ == '__main__':
    main()
```

## Fluke 2

### Solution 1: 

I think I'm close to solution that get's lot of credit meh. 

```py
import sys
from itertools import product
def play1(N, M):
    print(1, flush = True) # player 1
    grid = [[0] * (M + 1) for _ in range(N + 1)]
    print(1, 1, flush = True)
    grid[1][1] ^= 1
    progress = input()
    if progress != "C": return progress
    r, c = map(int, input().split())
    grid[r][c] ^= 1
    progress = input()
    sys.stdout.flush()
    if progress != "C": return progress
    while True:
        if r > 1 and r < N and grid[r - 1][c]:
            r += 1
        elif c > 1 and c < M and grid[r][c - 1]:
            c += 1
        elif r < N and r > 1 and grid[r + 1][c]:
            r -= 1
        elif c < M and c > 1 and grid[r][c + 1]:
            c -= 1
        elif r > 1:
            r -= 1
        else: c -= 1
        print(r, c, flush = True)
        grid[r][c] ^= 1
        progress = input()
        sys.stdout.flush()
        if progress != "C": return progress
        r1, c1 = map(int, input().split())
        grid[r1][c1] ^= 1
        progress = input()
        sys.stdout.flush()
        if progress != "C": return progress
def play2(N, M):
    print(2, flush = True) # player 2
    grid = [[0] * (M + 1) for _ in range(N + 1)]
    r, c = map(int, input().split())
    grid[r][c] ^= 1
    progress = input()
    sys.stdout.flush()
    if progress != "C": return progress
    for r1, c1 in product(range(1, N + 1), range(1, M + 1)):
        if (r1, c1) == (r, c): continue
        print(r1, c1, flush = True)
        grid[r1][c1] ^= 1
        break
    pr, pc = r, c
    progress = input()
    sys.stdout.flush()
    if progress != "C": return progress
    while True:
        r, c = map(int, input().split())
        grid[r][c] ^= 1
        progress = input()
        sys.stdout.flush()
        if progress != "C": return progress
        if r > 1 and r < N and grid[r - 1][c]:
            print(r - 1, c, flush = True)
        elif c > 1 and c < M and grid[r][c - 1]:
            print(r, c - 1, flush = True)
        elif r < N and r > 1 and grid[r + 1][c]:
            print(r + 1, c, flush = True)
        elif c < M and c > 1 and grid[r][c + 1]:
            print(r, c + 1, flush = True)
        else:
            print(pr, pc, flush = True)
            progress = input()
            if progress != "C": return progress
            pr, pc = r, c
def main():
    N, M, T = map(int, input().split())
    for _ in range(T):
        if N <= 2 and M == 2: 
            if play2(N, M) != "W": break
        elif N <= 2:
            if play3(N, M) != "W": break
        else: 
            if play1(N, M) != "W": break
    
if __name__ == '__main__':
    main()
```

## Hidden Tree

### Solution 1:

try this

```py

```

## Into the Woods

### Solution 1: 

try this

```py

```