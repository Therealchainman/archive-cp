# Codeforces Round 913 Div 3

## A. Rook

### Solution 1: 

```py
def main():
    cols = "abcdefgh"
    rows = "12345678"
    pos = input()
    c, r = pos[0], pos[1]
    ans = []
    for col in cols:
        ans.append(col + r)
    for row in rows:
        ans.append(c + row)
    ans = set(ans)
    ans.remove(pos)
    for p in ans: print(p)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## B. YetnotherrokenKeoard

### Solution 1: 

```py
from collections import deque

def main():
    keys = input()
    lstack, ustack = deque(), deque()
    for i, ch in enumerate(keys):
        if ch == "b":
            if lstack: lstack.pop()
        elif ch == "B":
            if ustack: ustack.pop()
        elif ch.islower():
            lstack.append(i)
        else:
            ustack.append(i)
    ans = []
    while lstack or ustack:
        if not lstack:
            ans.append(ustack.popleft())
        elif not ustack:
            ans.append(lstack.popleft())
        elif lstack[0] < ustack[0]:
            ans.append(lstack.popleft())
        else:
            ans.append(ustack.popleft())
    res = "".join(map(lambda x: keys[x], ans))
    print(res)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## C. Removal of Unattractive Pairs

### Solution 1: 

```py
def main():
    n = int(input())
    s = input()
    ans = 1 if n & 1 else 0
    freq = Counter(s)
    cnt = freq.most_common()[0][1]
    if cnt > n // 2:
        ans = max(ans, 2 * cnt - n)
    print(ans)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## D. Jumping Through Segments

### Solution 1: 

```py
# FFFTTTT, return first T

def main():
    n = int(input())
    levels = [None] * n
    for i in range(n):
        L, R = map(int, input().split())
        levels[i] = (L, R + 1) # [L, R) ranges
    def intersection(a, b):
        return min(a[1], b[1]) - max(a[0], b[0]) > 0
    def possible(k):
        range = [0, 1]
        for L, R in levels:
            range[0] -= k
            range[1] += k
            if not intersection(range, [L, R]): return False
            range[0] = max(range[0], L)
            range[1] = min(range[1], R)
        return True
    lo, hi = 0, int(1e9) + 1
    while lo < hi:
        mid = (lo + hi) >> 1
        if possible(mid):
            hi = mid
        else:
            lo = mid + 1
    print(lo)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
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

