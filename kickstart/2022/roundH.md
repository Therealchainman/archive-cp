# Google Kickstart 2022 Round H

## 

### Solution 1: 

```py
def main():
    L, N = map(int, input().split())
    laps = pos = 0
    start = None
    for _ in range(N):
        dist, dir = input().split()
        dist = int(dist)
        if start is None:
            start = dir
        if dir == 'C':
            pos += dist
            cur = max(0, pos // L - (dir != start))
        else:
            pos -= dist
            cur = max(0, -pos // L - (dir != start) + 1)
        # print('dist, dir, pos, cur, start', dist, dir, pos, cur, start)
        if pos >= L or pos <= 0:
            start = dir
        pos %= L
        laps += cur
        # print('laps', laps)
    return laps
if __name__ == '__main__':
    T = int(input())
    for t in range(1, T+1):
        print(f'Case #{t}: {main()}')
```

## 

### Solution 1: 

```py
from math import *
def main():
    L = int(input())
    res = L
    for i in range(3, L+1):
        cand = 2*(L-i)//i + (L-i)%i + 4 + i
        res = min(res, cand)
    return res
if __name__ == '__main__':
    T = int(input())
    for t in range(1, T+1):
        print(f'Case #{t}: {main()}')
```

## 

### Solution 1: 

```py
from sys import *
setrecursionlimit(int(1e6))
def main():
    N = int(input())
    arr = list(map(int, input().split()))
    adj_list = [[] for _ in range(N)]
    for _ in range(N-1):
        u, v = map(int, input().split())
        adj_list[u-1].append(v-1)
        adj_list[v-1].append(u-1)
    size = [1]*N
    def smaller(node, parent):
        for child in adj_list[node]:
            if child == parent: continue
            child_small_segment_size = smaller(child, node)
            if arr[child] < arr[node]:
                size[node] += child_small_segment_size
        return size[node]
    smaller(0, -1)
    def larger(node, parent):
        for child in adj_list[node]:
            if child == parent: continue
            if arr[child] > arr[node]:
                size[child] += size[node]
            larger(child, node)
    larger(0, -1)
    return max(size)
if __name__ == '__main__':
    T = int(input())
    for t in range(1, T+1):
        print(f'Case #{t}: {main()}')
```

## Level Design

### Solution 1: 

```py

```