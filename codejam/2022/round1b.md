# Google Codejam Round 1B

## Pancake Deque

### Solution 1: double ended queue + greedily choose the smallest pancake

```py
from collections import deque
def main():
    N = int(input())
    D = deque(map(int,input().split()))
    prev = cnt = 0
    while D:
        if D[0] < D[-1]:
            cost = D.popleft()
        else:
            cost = D.pop()
        if cost >= prev:
            cnt += 1
        prev = max(prev, cost)
    return cnt
    
if __name__ == '__main__':
    T = int(input())
    for t in range(1,T+1):
        print(f"Case #{t}: {main()}")
```

## Controlled Inflation

### Solution 1: dynamic programming with the max and mins + greedy

This assumes that the best way is always to travel from either min or max of previous array
and then travel from there across the current array and end at min or max

```py
from functools import lru_cache
import sys
sys.setrecursionlimit(1000000)
def main():
    N, P = map(int,input().split())
    products = [list(map(int,input().split())) for _ in range(N)]
    mins, maxs = [min(products[i]) for i in range(N)], [max(products[i]) for i in range(N)]
    
    @lru_cache(None)
    def dfs(i, last_min):
        if i==N: return 0
        dist = maxs[i] - mins[i]
        prev = 0 if i==0 else mins[i-1] if last_min else maxs[i-1]
        return dist + min(
            abs(prev-mins[i]) + dfs(i+1, False),
            abs(prev-maxs[i]) + dfs(i+1, True)
        )
    return dfs(0, True)
if __name__ == '__main__':
    T = int(input())
    for t in range(1,T+1):
        print(f"Case #{t}: {main()}")
```

## ASeDatAb

### Solution 1: Simply pick so that eventually it will become 11111111

Only passes test set 1

```py
from random import shuffle
import sys
def main():
    BINARY = '10000000'
    while True:
        print(BINARY, flush=True)
        N = int(input())
        if N == 0:
            return True
        elif N == -1:
            return False
        arr = []
        for _ in range(N):
            arr.append('1')
        for _ in range(8-N):
            arr.append('0')
        shuffle(arr)
        BINARY = "".join(arr)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```