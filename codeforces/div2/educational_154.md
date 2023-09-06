# Codeforces Educational 154

## A. Prime Deletion

### Solution 1:  bitmask + brute force + prime check

```py
import math

def is_prime(num):
    for i in range(2, int(math.sqrt(num)) + 1):
        if num % i == 0: return False
    return True

def main():
    s = input()
    n = len(s)
    for mask in range(1 << n):
        num = len_ = 0
        for i in range(n):
            if (mask >> i) & 1:
                num = num * 10 + int(s[i])
                len_ += 1
        if is_prime(num) and len_ > 1: return print(num)
    print(-1)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## B. Two Binary Strings

### Solution 1:  dynamic programming

```py
def main():
    s = list(map(int, input()))
    t = list(map(int, input()))
    n = len(s)
    arr = [s[i] if s[i] == t[i] else 2 for i in range(n)]
    dp = [False] * n
    dp[0] = True
    for i in range(1, n):
        if arr[i] == 2: continue
        dp[i] = dp[i - 1]
        if dp[i]: continue
        for j in range(i - 1, -1, -1):
            if arr[i] == arr[j] and dp[j]:
                dp[i] = True
                break
    res = "YES" if dp[-1] else "NO"
    print(res)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## C. Queries for the Array

### Solution 1:  two pointers

```py
def main():
    s = input()
    n = len(s)
    t = 1
    m = n
    sz = 0
    for ch in s:
        if ch == "+":
            sz += 1
        elif ch == "-":
            sz -= 1
            if sz < m: m = n # not unsorted
            t = max(1, min(t, sz))
        elif ch == "0":
            if t >= sz: return print("NO")
            m = min(m, sz) # unsorted from m size
        else:
            if m <= sz: return print("NO")
            t = max(1, sz) # sorted for less than or equal to t size
            m = n # reset m to max, there is no unsort on array currently
    print("YES")

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## D. Sorting By Multiplication

### Solution 1:  dynamic programming + greedy

```py

```

## E. Non-Intersecting Subpermutations

### Solution 1: 

```py

```

