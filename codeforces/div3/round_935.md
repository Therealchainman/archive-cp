# Codeforces Round 935 Div 3

## 

### Solution 1: 

```py

```

## 

### Solution 1: 

```py

```

## 

### Solution 1: 

```py

```

## 

### Solution 1: 

```py

```

## G. Cook and Porridge

### Solution 1:  max heap, pointer, suffix max

```py
from heapq import heappop, heappush
def main():
    n, D = map(int, input().split())
    students = [[] for _ in range(D + 1)]
    data = [None] * n
    for i in range(n):
        k, s = map(int, input().split())
        data[i] = (k, s)
    smax = [0] * n
    for i in reversed(range(n)):
        smax[i] = max(data[i][0], (smax[i + 1] if i + 1 < n else 0))
    p = 0
    maxheap = []
    for i in range(1, D + 1):
        if maxheap and smax[p] < abs(maxheap[0][0]):
            k, _, s, j = heappop(maxheap)
            if i + s <= D: students[i + s].append(j)
        else:
            k, s = data[p]
            if i + s <= D: students[i + s].append(p)
            p += 1
        if p == n: return print(i)
        for j in students[i]:
            k, s = data[j]
            heappush(maxheap, (-k, i, s, j))
    print(-1)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## H. GCD is Greater

### Solution 1: 

```py

```

