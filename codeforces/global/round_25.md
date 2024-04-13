# Codeforces Global Round 25

## C. Ticket Hoarding

### Solution 1: greedy, sorting

```py
def main():
    n, m, k = map(int, input().split())
    arr = list(map(int, input().split()))
    ans = 0
    queries = sorted([(arr[i], i) for i in range(n)])
    tarr = []
    for cost, i in queries:
        take = min(k, m)
        if take == 0: break
        ans += cost * take
        k -= take
        tarr.append((i, take))
    tarr.sort()
    pen = 0
    for _, take in tarr:
        ans += take * pen
        pen += take
    print(ans)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## D. Buying Jewels

### Solution 1: 

```py

```

## E. No Palindromes

### Solution 1:  palindromes, splitting into non-palindromes

possibly could try every partition with string hashing? 

```py
def checker(s, p):
    return not check(s[:p]) and not check(s[p:])

def check(s):
    return s == s[::-1]

def main():
    s = input()
    n = len(s)
    if all(ch == s[0] for ch in s): return print("NO")
    if not check(s):
        print("YES")
        print(1)
        print(s)
        return 
    for i in range(n):
        if s[i] != s[0]: break
    if checker(s, i + 1):
        print("YES")
        print(2)
        print(s[:i + 1], s[i + 1:])
        return
    i += 1
    if checker(s, i + 1):
        print("YES")
        print(2)
        print(s[:i + 1], s[i + 1:])
    else:
        print("NO")

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## G. Clacking Balls

### Solution 1: 

```py

```

## I. Growing Trees

### Solution 1: 

```py

```