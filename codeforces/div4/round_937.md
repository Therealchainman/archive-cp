# Codeforces Round 937 Div 4

## D. Product of Binary Decimals

### Solution 1:  precompute, dfs, factorization

```py
pre = [10, 11, 100, 101, 110, 111, 1000, 1001, 1010, 1011, 1100, 1101, 1110, 1111, 10000, 10001, 10010, 10011, 10100, 10101, 10110, 10111, 11000, 11001, 11010, 11011, 11100, 11101, 11110, 11111, 100000]
 
def check(n):
    stk = [n]
    vis = set([n])
    while stk:
        x = stk.pop()
        if x == 1: return True
        for v in pre:
            if x % v == 0:
                cur = x // v
                if cur in vis: continue
                stk.append(cur)
                vis.add(cur)
    return False
def main():
    n = int(input())
    ans = "YES" if check(n) else "NO"
    print(ans)
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## E. Nearly Shortest Repeating Substring

### Solution 1:  number theory, factorization, greedy

```py
 
import math
def factors(n):
    f = []
    for i in range(1, int(math.sqrt(n))+1):
        if n % i == 0:
            f.append(i)
            if i != n//i:
                f.append(n//i)
    return sorted(f) 
 
def check(k, s):
    idx = None
    chars = set()
    cnt = 0
    for i in range(k, len(s)):
        if s[i] != s[i % k]:
            cur = i % k
            if idx is None or idx == cur:
                idx = cur
                chars.add(s[i])
                cnt += 1
            else: return False
    if len(chars) == 1 and cnt + 1 == len(s) // k: return True
    return cnt <= 1
 
def main():
    n = int(input())
    s = input()
    facts = factors(n)
    for f in facts:
        if check(f, s): return print(f)
    print(n)
    
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## F. 0, 1, 2, Tree!

### Solution 1:  math, complete binary trees, ceil division

```py
import math
def ceil(x, y):
    return (x + y - 1) // y
def main():
    a, b, c = map(int, input().split())
    if c != a + 1: return print(-1)
    p = int(math.log2(a)) if a > 0 else -1
    ans = p + 1
    if a > 0:
        over = a - pow(2, p) + 1
        rem = pow(2, p) - over
        b = max(0, b - rem)
        p += 1
    ans += ceil(b, c)
    print(ans)
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## G. Shuffling Songs

### Solution 1:  dp bitmask, longest path in undirected graph, hamiltonian path existence of all subgraphs

```py
def main():
    N = int(input())
    adj = [[] for _ in range(N)]
    node = [None] * N
    for i in range(N):
        g, w = input().split()
        node[i] = (g, w)
    for i in range(N):
        for j in range(i):
            if node[i][0] == node[j][0] or node[i][1] == node[j][1]:
                adj[i].append(j)
                adj[j].append(i)
    dp = [[False] * N for _ in range(1 << N)]
    for i in range(N):
        dp[1 << i][i] = True # start from each node
    for mask in range(1 << N):
        # travel from u to v
        for u in range(N):
            if not dp[mask][u]: continue 
            for v in adj[u]:
                if (mask >> v) & 1: continue
                dp[mask | (1 << v)][v] = dp[mask][u]
    ans = N
    for mask in range(1 << N):
        cnt = bin(mask).count("1")
        if N - cnt >= ans: continue
        for u in range(N):
            if dp[mask][u]:
                ans = N - cnt
                break
    print(ans)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

