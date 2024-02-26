# Atcoder Beginner Contest 342

## C - Many Replacement 

### Solution 1:  hash map

```py
import string
def main():
    N = int(input())
    S = list(input())
    Q = int(input())
    char_map = {ch: ch for ch in string.ascii_lowercase}
    for _ in range(Q):
        c, d = input().split()
        for ch in char_map:
            if char_map[ch] == c: char_map[ch] = d
    ans = "".join([char_map.get(ch) for ch in S])
    print(ans)
if __name__ == '__main__':
    main()
```

## D - Square Pair 

### Solution 1:  math, square, multiplicity of prime, counter

```py
def main():
    N = int(input())
    arr = list(map(int, input().split()))
    ans = 0
    counts = [0] * 200_000
    for num in arr:
        x, i = num, 2
        while i * i <= x:
            while x % (i * i) == 0:
                x //= i * i
            i += 1
        ans += counts[x]
        counts[x] += 1
    print(ans + counts[0] * (N - counts[0]))

if __name__ == '__main__':
    main()
```

## E - Last Train 

### Solution 1:  max heap, dp, math, backwards, from last station to station i

```py
import math
from heapq import heappush, heappop
def main():
    N, M = map(int, input().split())
    adj = [[] for _ in range(N)]
    ans = [-math.inf] * N
    for _ in range(M):
        l, d, k, c, u, v = map(int, input().split())
        u -= 1; v -= 1
        adj[v].append((u, l, d, k, c))
    maxheap = [(-math.inf, N - 1)]
    while maxheap:
        t, u = heappop(maxheap)
        t *= -1
        if t < ans[u]: continue
        for v, l, d, k, c in adj[u]:
            rem = t - l - c 
            nt = l + min(k - 1, rem // d) * d 
            if nt < 0: continue
            if nt > ans[v]:
                ans[v] = nt 
                heappush(maxheap, (-nt, v))
    for t in ans[:-1]:
        print(t if t > -math.inf else "Unreachable")
if __name__ == '__main__':
    main()
```

## F - Black Jack 

### Solution 1: 

```py

```

## G - Retroactive Range Chmax 

### Solution 1: 

```py

```

