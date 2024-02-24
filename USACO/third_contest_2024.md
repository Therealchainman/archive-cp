# Third Contest 2024

## 

### Solution 1: 

```py
def main():
    S = input()
    print("E" if S[-1] == "0" else "B")

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## 

### Solution 1: 

```py
def main():
    N, M = map(int, input().split())
    S = input()
    cap = list(map(int, input().split()))
    adj = [0] * N
    indegrees = [0] * N
    for u in range(N):
        v = (u + 1) % N if S[u] == "R" else (u - 1) % N
        adj[u] = v
        indegrees[v] += 1
    def top(u, rem):
        while rem > 0:
            take = min(rem, cap[u])
            rem -= take
            cap[u] -= take
            u = adj[u]
            if indegrees[u] != 1: break
    for i in range(N):
        if indegrees[i] > 0: continue
        top(i, M)
    print(sum(cap))

if __name__ == '__main__':
    main()
```

## 

### Solution 1: 

```py
from bisect import bisect_right
def main():
    N, Q = map(int, input().split())
    close = list(map(int, input().split()))
    time = list(map(int, input().split()))
    diff = sorted([c - t for c, t in zip(close, time)])
    for _ in range(Q):
        V, S = map(int, input().split())
        i = bisect_right(diff, S)
        farms = N - i
        if farms >= V: print("YES")
        else: print("NO")

if __name__ == '__main__':
    main()
```

## Target Practice II

### Solution 1: 

```py

```

## Test Tubes

### Solution 1: 

8
1 3
1
2
2 3
12
22
2 3
22
12
2 3
11
21
4 3
1111
2121
4 3
1212
2121
4 3
2121
1212
5 3
12121
21211

4
3 3
121
121
4 3
1212
1212
4 3
1211
1212
4 3
1212
1211

```py
def solve(p1, p2, P, swapped):
    t1, t2 = 1, 2
    if swapped: t1, t2 = t2, t1
    # bottom element in p1 is 1, and bottom element in p2 is 2
    ans = []
    if len(p1) == 1:
        while len(p2) > 1:
            val = p2.pop()
            if val == 1: ans.append((t2, t1))
            else: ans.append((t2, 3))
        if len(ans) > 1: ans.append((3, t2))
    elif len(p2) == 1:
        while len(p1) > 1:
            val = p1.pop()
            if val == 2: ans.append((t1, t2))
            else: ans.append((t1, 3))
        if len(ans) > 1: ans.append((3, t1))
    else:
        while len(p2) > 1:
            val = p2.pop()
            if val == 1: ans.append((t2, 3))
            else: 
                if p1[-1] != 2: p1.append(2)
                ans.append((t2, t1))
        while len(p1) > 1:
            val = p1.pop()
            if val == 1: ans.append((t1, 3))
            else: ans.append((t1, t2))
        ans.append((3, t1))
    print(len(ans))
    if P != 1:
        for x, y in ans:
            print(x, y)
def solve2(p1, p2, P):
    ans = []
    c1, c2 = 1, 2
    if p1[0] == 2: c1, c2 = c2, c1
    t1, t2 = 1, 2
    if p1[-1] != c1 and p2[-1] == c1:
        t1, t2 = t2, t1
        p1, p2 = p2, p1
    t3 = False
    while len(p2) > 1:
        val = p2.pop()
        if val == c2: 
            ans.append((t2, 3))
            t3 = True
        else: 
            if p1[-1] != c1: p1.append(c1)
            ans.append((t2, t1))
    while len(p1) > 0:
        val = p1.pop()
        if val == c2: 
            ans.append((t1, 3))
            t3 = True
        else: ans.append((t1, t2))
    if t3: ans.append((3, t1))
    print(len(ans))
    if P != 1:
        for x, y in ans:
            print(x, y)

def main():
    N, P = map(int, input().split())
    f = list(map(int, input()))
    s = list(map(int, input()))
    p1, p2 = [f[0]], [s[0]]
    for v in f:
        if p1[-1] != v: p1.append(v)
    for v in s:
        if p2[-1] != v: p2.append(v)
    # print("p", p1, p2)
    if p1[0] == p2[0]: solve2(p1, p2, P)
    elif p1[0] == 1: solve(p1, p2, P, False)
    else: solve(p2, p1, P, True)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## Moorbles

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