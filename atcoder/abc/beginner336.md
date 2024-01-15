# Atcoder Beginner Contest 336

## C - Even Digits 

### Solution 1:  Convert to base 5

```py
def main():
    N = int(input()) - 1
    if N == 0: return print(N % 5)
    values = [0, 2, 4, 6, 8]
    base_five = []
    while N > 0:
        base_five.append(N % 5)
        N //= 5
    ans = []
    for v in reversed(base_five):
        ans.append(values[v])
    print("".join(map(str, ans)))

if __name__ == '__main__':
    main()
```

## D - Pyramid 

### Solution 1:  dynamic programming 

```py
def main():
    N = int(input())
    arr = list(map(int, input().split()))
    dp = [0] * (N + 1)
    for i in reversed(range(N)):
        dp[i] = min(arr[i], dp[i + 1] + 1)
    plen = ans = 0
    for i in range(N):
        plen = min(arr[i], plen + 1)
        ans = max(ans, min(plen, dp[i]))
    print(ans)
if __name__ == '__main__':
    main()
```

## E - Digit Sum Divisible 

### Solution 1:  digit dp, digit sums

```py
# for each fixed digit sum
# (index, digit sum, remainder modulos fixed digit sum, tight)

def main():
    N = input()
    ans = 0
    for ds in range(1, 9 * 14 + 1):
        dp = Counter({(0, 0, 1): 1})
        for d in map(int, N):
            ndp = Counter()
            for (dig_sum, rem, tight), cnt in dp.items():
                for dig in range(10 if not tight else d + 1):
                    ndig_sum, nrem, ntight = dig_sum + dig, (rem * 10 + dig) % ds, tight and dig == d
                    if ndig_sum > ds: break
                    ndp[(ndig_sum, nrem, ntight)] += cnt
            dp = ndp
        ans += dp[(ds, 0, 0)] + dp[(ds, 0, 1)]
    print(ans)

if __name__ == '__main__':
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

