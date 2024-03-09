# Codeforces Round 930 Div 2

## A. Shuffle Party

### Solution 1:  math

```py
def main():
    n = int(input())
    ans = 1
    while ans * 2 <= n:
        ans *= 2 
    print(ans)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## B. Binary Path

### Solution 1:  greedy

```py
def main():
    n = int(input())
    grid = [list(map(int, input())) for _ in range(2)]
    start = end = None
    for i in range(1, n):
        top, bot = grid[0][i], grid[1][i - 1]
        if start is None and top == bot:
            start = i
        elif bot < top:
            end = i + 1
            break
        elif start is not None and top < bot:
            start = None
    if end is None: end = n + 1
    if start is None: start = end - 1
    arr = []
    for i in range(n):
        if i < start - 1:
            arr.append(grid[0][i])
        elif i == start - 1:
            arr.append(grid[0][i])
            arr.append(grid[1][i])
        else:
            arr.append(grid[1][i])
    print("".join(map(str, arr)))
    print(end - start)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## C. Bitwise Operation Wizard

### Solution 1:  bit manipulation, interactive

```py
def main():
    n = int(input())
    # FIND MAX_I
    max_i = 0
    for i in range(1, n):
        print(f"? {i} {i} {max_i} {max_i}")
        resp = input()
        if resp == ">": max_i = i
    # FIND CANDIDATES
    cands = [0]
    best_i = 0
    for i in range(1, n):
        print(f"? {i} {max_i} {max_i} {best_i}")
        resp = input()
        if resp == "=": cands.append(i)
        elif resp == ">": best_i = i; cands = [i]
    # FIND MIN IN CANDIDATES
    min_i = cands[0]
    for i in cands[1:]:
        print(f"? {i} {i} {min_i} {min_i}")
        resp = input()
        if resp == "<": min_i = i
    # PRINT ANSWER
    print(f"! {min_i} {max_i}")

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## D. Pinball

### Solution 1:  prefix sum, suffix, binary search, math

```py

```

## E. Pokémon Arena

### Solution 1: 

```py

```

## 

### Solution 1: 

```py

```

