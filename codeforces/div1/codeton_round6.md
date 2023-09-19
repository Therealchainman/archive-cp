# Codeforces CodeTon Round 6

## A. MEXanized Array

### Solution 1: 

```py
def main():
    n, k, x = map(int, input().split())
    mex = res = 0
    for _ in range(n):
        if mex < k and mex <= x:
            res += mex
            mex += 1
        else:
            res += x - (1 if mex == x else 0)
    if mex < k: return print(-1)
    print(res)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## B. Friendly Arrays

### Solution 1: 

```py
def main():
    n, m = map(int, input().split())
    A = list(map(int, input().split()))
    B = map(int, input().split())
    B_or = 0
    for b in B:
        B_or |= b
    if n & 1:
        mn = mx = 0
        for a in A:
            mn ^= a
            mx ^= a | B_or
    else:
        mn = mx = 0
        for a in A:
            mn ^= a | B_or
            mx ^= a
    print(mn, mx)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## C. Colorful Table

### Solution 1: 

```py
def main():
    n, k = map(int, input().split())
    arr = list(map(int, input().split()))
    result = [0] * k
    first, last = [n - 1] * k, [0] * k
    for i in range(n):
        num = arr[i] - 1
        first[num] = min(first[num], i)
        last[num] = max(last[num], i)
    left, right = 0, n - 1
    largest = max(arr)
    for i in range(n):
        if arr[i] == largest:
            left = i
            break
    for i in reversed(range(n)):
        if arr[i] == largest:
            right = i
            break
    for num in sorted(set(arr), reverse = True):
        num -= 1
        # move left
        left = min(left, first[num])
        # move right
        right = max(right, last[num])
        # update result
        result[num] = 2 * (right - left + 1)
    print(*result)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## D. Prefix Purchase

### Solution 1:  suffix array + math

Think step functions and overlap

```py
import math

def main():
    n = int(input())
    cost = list(map(int, input().split()))
    k = int(input())
    pmin = [math.inf] * (n + 1)
    for i in reversed(range(n)):
        pmin[i] = min(pmin[i + 1], cost[i])
    diff = [0] * n
    for i in range(n):
        if k < pmin[i]: break
        if pmin[i + 1] == math.inf:
            x = k // pmin[i]
            diff[i] = x
            k -= x * pmin[i]
        elif pmin[i] < pmin[i + 1]:
            delta = pmin[i + 1] - pmin[i]
            remainder = k % pmin[i]
            step = remainder // delta
            x = max(0, k // pmin[i] - step)
            diff[i] = x
            k -= x * pmin[i]
    ans = [0] * (n + 1)
    for i in reversed(range(n)):
        ans[i] = diff[i] + ans[i + 1]
    print(*ans[:-1])

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## E. Another MEX Problem

### Solution 1:  dynamic programming + bit manipulation

```py

```

## F. Lazy Numbers

### Solution 1: 

```py

```

## G. MEXanization

### Solution 1: 

```py

```

## H. Standard Graph Problem

### Solution 1: 

```py

```

