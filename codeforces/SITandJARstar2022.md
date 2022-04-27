# SIT & JUB STAR Contest 2022

## Templates

### Solution 1: greedy

```py
from itertools import product
n = int(input())
arr = [input() for _ in range(n)]
template_arr = list(arr[0])
for i, j in product(range(1,n), range(len(template_arr))):
    if template_arr[j] != arr[i][j]:
        template_arr[j] = '?'
print("".join(template_arr))
```

## Odd Division

### Solution 1: math

```py
s = list(map(int,list(input())))
cnt = 0
for i in range(len(s)):
    if s[i]%2==1 and (i+1==len(s) or s[i+1]!=0):
        cnt += 1
if s[-1]%2==0 or s[0]==0:
    print(-1)
else:
    print(cnt)
```

## Market

### Solution 1: binary search algorithm

```py
from bisect import bisect_right, bisect_left
n = int(input())
sellers_arr = sorted(list(map(int,input().split())))
m = int(input())
buyers_arr = sorted(list(map(int,input().split())))
mostCost = 0
for x in range(sellers_arr[0], max(sellers_arr[-1], buyers_arr[-1])+1):
    num_sellers = bisect_right(sellers_arr, x) 
    num_buyers = m - bisect_left(buyers_arr, x)
    num_sales = min(num_sellers, num_buyers)
    mostCost = max(mostCost, num_sales*x)
print(mostCost)
```

## Yet another Card Game

### Solution 1: simulate + queue

```py
from collections import deque
def main():
    n = int(input())
    finn = deque(map(int,input().split()))
    levi = deque(list(map(int,input().split())))
    max_turns = int(1e6)
    for turn in range(1, max_turns+1):
        a, b = finn.popleft(), levi.popleft()
        if a > b:
            finn.append(a)
            finn.append(b)
        else:
            levi.append(b)
            levi.append(a)
        if not finn or not levi:
            return turn
    return -1
 
 
if __name__ == '__main__':
    print(main())
```

## Best Sandwich Recipe

### Solution 1: binary search 

```py
from bisect import bisect_right, bisect_left
def main():
    input()
    B = sorted(list(map(int,input().split())))
    C = sorted(list(map(int,input().split())))
    S = sorted(list(map(int,input().split())))
    cnt = lo_c = lo_s = 0
    for b in B:
        c = bisect_right(C, b, lo=lo_c)
        if c==len(C): break
        s = bisect_right(S, C[c], lo=lo_s)
        if s == len(S): break
        cnt += 1
        lo_c, lo_s = c+1, s+1
    return cnt
 
if __name__ == '__main__':
    print(main())
```

## ZigZags

### Solution 1:  dynamic programming

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

##

### Solution 1: 

```py

```