# Meta Hacker Cup 2017

# Round 2

## 

```py
def main():
    N, M, K = map(int, input().split())
    if 2 * K + 3 > max(N, M): return -1
    if K == 1 and N >= 5 and M >= 5: return 5
    if K != 1 and N >= 3 * K + 1 and M >= 3 * K + 1: return 4
    n = (min(N, M) + K - 1) // K # ceil division
    if n < 2: return -1
    return n

if __name__ == '__main__':
    T = int(input())
    for t in range(1, T + 1):
        print(f"Case #{t}: {main()}")
```

## 

```py

```

## 

```py

```

##

```py

```

##

```py

```

##

```py

```

##

```py

```