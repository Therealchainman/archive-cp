# Google Kickstart 2022 Round F

## 

### Solution 1: 

```py
def main():
    m, n, p = map(int, input().split())
    arr = [0]*n
    for i in range(1, m+1):
        walk = list(map(int, input().split()))
        if i == p:
            him = walk
        else:
            for j in range(n):
                arr[j] = max(arr[j], walk[j])
    return sum(max(0, x - y) for x, y in zip(arr, him))
    
if __name__ == '__main__':
    T = int(input())
    for t in range(1, T+1):
        print(f'Case #{t}: {main()}')
```

## 

### Solution 1: 

```py
from math import hypot
def main():
    Rs, Rh = map(int, input().split())
    N = int(input())
    disks = []
    squared_dist = lambda x, y: x*x + y*y
    intersects = lambda d: d <= (Rh+Rs)*(Rh+Rs)
    for _ in range(N):
        x, y = map(int, input().split())
        disks.append((x, y, 0))
    M = int(input())
    for _ in range(M):
        x, y = map(int, input().split())
        disks.append((x, y, 1))
    x, y = 2, 3
    disks = list(filter(lambda x: intersects(squared_dist(x[0], x[1])), disks))
    disks.sort(key = lambda x: squared_dist(x[0], x[1]))
    red = yellow = 0
    prev_team = -1
    for _, _, team in disks:
        if prev_team != -1 and prev_team != team: break
        red += (team==0)
        yellow += team
        prev_team = team
    return f'{red} {yellow}'
    
if __name__ == '__main__':
    T = int(input())
    for t in range(1, T+1):
        print(f'Case #{t}: {main()}')
```

## 

### Solution 1: 

```py
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    total = 0
    for i in range(n):
        prefix = 0
        for j in range(i, n):
            prefix += arr[j]
            if prefix < 0: break
            total += prefix
    return total
    
if __name__ == '__main__':
    T = int(input())
    for t in range(1, T+1):
        print(f'Case #{t}: {main()}')
```

## 

### Solution 1: 

```py

```