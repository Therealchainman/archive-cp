# Codeforces Round 932 Div 2

## 

### Solution 1: 

```py

```

## B. Informatics in MAC

### Solution 1: 

```py

```

## C. Messenger in MAC

### Solution 1: 

```py

```

## D. Exam in MAC

### Solution 1:  inclusion exclusion, combinatorics, count pairs, parity, math

```py
def main():
    n, c = map(int, input().split())
    s = list(map(int, input().split()))
    # diff => y - x = s[i]
    # add => x + y = s[i]
    diff = add = even = 0
    for i in range(n):
        diff += c - s[i] + 1
        add += s[i] // 2 + 1
        if s[i] % 2 == 0: even += 1
    odd = n - even
    intersect_count = even * (even + 1) // 2 + odd * (odd + 1) // 2
    ans = (c + 1) * (c + 2) // 2 - add - diff + intersect_count
    print(ans)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## E. Distance Learning Courses in MAC

### Solution 1:  bit manipulation, sparse table for bitwise or range queries, prefix sum for count of each bit, bitwise largest common prefix

```py
BITS = 30
LOG = 14
def lcp(x, y):
    w = 0
    for i in reversed(range(BITS)):
        vx, vy = (x >> i) & 1, (y >> i) & 1
        if vx != vy: break 
        w |= (vx << i)
    return w
def main():
    n = int(input())
    pcount = [[0] * BITS for _ in range(n)]
    st = [[0] * n for _ in range(LOG)]
    for i in range(n):
        x, y = map(int, input().split())
        w = lcp(x, y)
        st[0][i] = w
        y -= w
        for j in range(BITS):
            pcount[i][j] = 1 if (y >> j) & 1 else 0
            if i > 0: pcount[i][j] += pcount[i - 1][j]
    # CONSTRUCT SPARSE TABLE
    for i in range(1, LOG):
        j = 0
        while (j + (1 << (i - 1))) < n:
            st[i][j] = st[i - 1][j] | st[i - 1][j + (1 << (i - 1))]
            j += 1
    # QUERY SPARSE TABLE
    def query(l, r):
        res = 0
        for i in reversed(range(LOG)):
            if (1 << i) <= r - l + 1:
                res |= st[i][l] 
                l += 1 << i
        return res
    q = int(input())
    ans = [0] * q
    for i in range(q):
        l, r = map(int, input().split())
        l -= 1; r -= 1
        ans[i] = query(l, r)
        for j in reversed(range(BITS)):
            s = (ans[i] >> j) & 1
            cnt = pcount[r][j] + s
            if l > 0: cnt -= pcount[l - 1][j]
            if cnt >= 1: ans[i] |= (1 << j)
            if cnt > 1: ans[i] |= ((1 << j) - 1); break
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

