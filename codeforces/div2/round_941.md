# Codeforces Round 941 Div 2

## D. Missing Subsequence Sum

### Solution 1:  binary digits, powers of 2, fill in gaps by analysis

```py
def main():
    n, k = map(int, input().split())
    i = 0
    while 2 ** (i + 1) <= k:
        i += 1
    ans = []
    for j in range(23):
        if j == i: continue
        ans.append(2 ** j)
    ans.append(k + 1)
    ans.append(k + 1 + 2 ** i)
    ans.append(k - 2 ** i)
    print(len(ans))
    print(*ans)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## E. Folding Strip

### Solution 1: 

```py

```

## F. Missing Subarray Sum

### Solution 1: 

```py

```

