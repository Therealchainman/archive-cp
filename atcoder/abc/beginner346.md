# Atcoder Beginner Contest 346

## E - Paint 

### Solution 1: offline query, sort, reverse, track count of blocked rows and cols

```py
MAXN = 200_000 + 5

def main():
    R, C, M = map(int, input().split())
    queries = [None] * M
    for i in range(M):
        t, a, x = map(int, input().split())
        queries[i] = (t, a, x)
    colors = [0] * MAXN
    row = [0] * R
    col = [0] * C
    num_rows = num_cols = 0
    for t, a, x in reversed(queries):
        a -= 1
        if t == 1: # repaint row
            if not row[a]:
                colors[x] += C - num_cols
                row[a] = 1
                num_rows += 1
        else: # repaint col
            if not col[a]:
                colors[x] += R - num_rows
                col[a] = 1
                num_cols += 1
    total = sum(colors)
    colors[0] += R * C - total
    ans = []
    for c, cnt in enumerate(colors):
        if cnt > 0: ans.append((c, cnt))
    print(len(ans))
    for color, cnt in ans:
        print(color, cnt)
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