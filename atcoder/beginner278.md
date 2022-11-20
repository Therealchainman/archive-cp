# Atcoder Beginner Contest 278

## Shift

### Solution 1:  deque + simulation

```py
from collections import deque
def main():
    n, k = map(int, input().split())
    queue = deque(map(int, input().split()))
    for _ in range(k):
        queue.popleft()
        queue.append(0)
    return ' '.join(map(str, queue))

if __name__ == '__main__':
    print(main())
```

## Misjudge the Time

### Solution 1:  math + string + modular arithmetic

```py
def is_valid(h, m):
    return h >= 0 and h <= 23 and m >= 0 and m <= 59

def is_confusing(minutes):
    h = minutes // 60
    m = minutes % 60
    last_digit = str(h%10)
    h_str = str(h).zfill(2)
    m_str = str(m).zfill(2)
    first_digit = m_str[0]
    h_swap = int(h_str[0] + first_digit)
    m_swap = int(last_digit + m_str[1])
    return is_valid(h, m) and is_valid(h_swap, m_swap)

def main():
    h, m = map(int, input().split())
    minutes = h * 60 + m
    while not is_confusing(minutes):
        minutes = (minutes + 1)%1440
    h = minutes // 60
    m = minutes % 60
    return f'{h} {m}'

if __name__ == '__main__':
    print(main())
```

## FF

### Solution 1:  set

```py
def main():
    n, q = map(int, input().split())
    following = set()
    result = []
    for _ in range(q):
        t, a, b = map(int, input().split())
        if t == 1:
            following.add((a, b))
        elif t == 2:
            following.discard((a, b))
        else:
            if (a, b) in following and (b, a) in following:
                result.append('Yes')
            else:
                result.append('No')
    return '\n'.join(result)

if __name__ == '__main__':
    print(main())
```

## All Assign Point Add

### Solution 1: dictionary + greedy

```py
from math import inf
def main():
    n = int(input())
    arr = map(int, input().split())
    q = int(input())
    result = []
    updated = {i: x for i, x in enumerate(arr)}
    assigned = 0
    for _ in range(q):
        query = list(map(int, input().split()))
        if query[0] == 1:
            assigned = query[1]
            updated.clear()
        elif query[0] == 2:
            i, x = query[1:]
            i -= 1
            if i in updated:
                updated[i] += x
            else:
                updated[i] = assigned + x
        elif query[0] == 3:
            i = query[1] - 1
            if i in updated:
                result.append(updated[i])
            else:
                result.append(assigned)
    return '\n'.join(map(str, result))

if __name__ == '__main__':
    print(main())
```

## Grid Filling

### Solution 1:  dictionary + find overlap of integers

```py
from itertools import product
def main():
    H, W, N, h, w = map(int, input().split())
    grid = [[0] * W for _ in range(H)]
    for i in range(H):
        row = map(int, input().split())
        for j, x in enumerate(row):
            grid[i][j] = x
    bounding_boxes = {} # (min(i), min(j), max(i), max(j))
    for i, j in product(range(H), range(W)):
        n = grid[i][j]
        if n in bounding_boxes:
            min_i, min_j, max_i, max_j = bounding_boxes[n]
            bounding_boxes[n] = (min(min_i, i), min(min_j, j), max(max_i, i), max(max_j, j))
        else:
            bounding_boxes[n] = (i, j, i, j)
    total_distinct = len(bounding_boxes)
    ans = [[total_distinct]*(W-w+1) for _ in range(H-h+1)]
    for i, j in product(range(H-h+1), range(W-w+1)):
        for n, (min_i, min_j, max_i, max_j) in bounding_boxes.items():
            if min_i >= i and min_j >= j and max_i < i+h and max_j < j+w:
                ans[i][j] -= 1
    for row in ans:
        print(' '.join(map(str, row)))

if __name__ == '__main__':
    main()
```