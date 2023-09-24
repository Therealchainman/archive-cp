# Atcoder Beginner Contest 321

## A - 321-like Checker

### Solution 1: 

```py
def main():
    n = list(map(int, input()))
    res = "Yes" if all(n[i] < n[i - 1] for i in range(1, len(n))) else 'No'
    print(res)

if __name__ == '__main__':
    main()
```

## B - Cutoff

### Solution 1: 

```py
def main():
    n, k = map(int, input().split())
    arr = sorted(map(int, input().split()))
    sum_ = sum(arr[1:-1])
    needed = k - sum_
    if needed <= arr[0]: return print(0)
    if needed > arr[-1]: return print(-1)
    print(needed)

if __name__ == '__main__':
    main()
```

## C - 321-like Searcher

### Solution 1: 

```py
def main():
    k = int(input())
    arr = []
    for mask in range(1, 1 << 10):
        val = 0
        for i in reversed(range(10)):
            if (mask >> i) & 1:
                val = val * 10 + i
        arr.append(val)
    arr.sort()
    print(arr[k])

if __name__ == '__main__':
    main()
```

## D - Set Menu

### Solution 1: 

```py
import bisect

def main():
    n, m, p = map(int, input().split())
    A = list(map(int, input().split()))
    B = sorted(map(int, input().split()))
    psum = [0] * (m + 1)
    for i in range(m):
        psum[i + 1] = psum[i] + B[i]
    res = 0
    for a in A:
        i = bisect.bisect_right(B, p - a)
        res += i * a + psum[i] + (m - i) * p
    print(res)

if __name__ == '__main__':
    main()
```

## E - Complete Binary Tree

### Solution 1: 

```py
def main():
    t = int(input())
    for _ in range(t):
        n, x, k = map(int, input().split())
        print("n, x, k", n, x, k)
        res = 0
        # first tree
        u = x
        rem = k
        while (u << 1) <= n and rem > 0:
            u <<= 1
            rem -= 1
        if rem == 0:
            res += min(n, u + pow(2, k) - 1) - u + 1
        prev_even = x % 2 == 0
        k -= 1
        if k == 0: 
            res += 1
        k -= 1
        x >>= 1
        while x > 0 and k >= 0:
            u = 2 * x + prev_even
            print("start u", u)
            rem = k
            while (u << 1) <= n and rem > 0:
                u <<= 1
                rem -= 1
            print("u", u, "rem", rem)
            if rem == 0:
                res += min(n, u + pow(2, k) - 1) - u + 1
            prev_even = x % 2 == 0
            print("res", res, "k", k, "x", x)
            x >>= 1
            k -= 1

        print(res)
        

if __name__ == '__main__':
    main()
```

## F - #(subset sum = K) with Add and Erase

### Solution 1: 

```py

```

## G - Electric Circuit

### Solution 1: 

```py

```

