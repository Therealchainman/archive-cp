# Codeforces Round 927 Div 3

## 

### Solution 1: 

```py
from collections import deque
def main():
    n, m = map(int, input().split())
    q = deque(map(int, input().split()))
    arr = []
    s = input()
    for i in range(n):
        if s[i] == "L": arr.append(q.popleft())
        else: arr.append(q.pop())
    cur = 1
    ans = [0] * n
    for i in reversed(range(n)):
        cur = (cur * arr[i]) % m
        ans[i] = cur
        if cur == 0: break
    print(*ans)
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

## 

### Solution 1: 

```py

```

## 

### Solution 1: 

```py

```

