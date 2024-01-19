# Codeforces Educational 161 Div 2

## B. Forming Triangles

### Solution 1:  triangle inequality, math, combinatorics

```py
def main():
    n = int(input())
    cnt = [0] * (n + 1)
    for num in map(int, input().split()):
        cnt[num] += 1
    ans = psum = 0
    for c in map(lambda x: cnt[x], range(n + 1)):
        ans += c * (c - 1) * (c - 2) // 6 # pick 3 from count of items
        ans += c * (c - 1) // 2 * psum  # pick 2 from count of items and 1 from others
        psum += c
    print(ans)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## C. Closest Cities

### Solution 1:  prefix sum

```py
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    m = int(input())
    psum = [0] * n
    ssum = [0] * n
    psum[1] = ssum[-2] = 1
    for i in range(1, n - 1):
        psum[i + 1] = psum[i] + (1 if abs(arr[i + 1] - arr[i]) < abs(arr[i] - arr[i - 1]) else abs(arr[i] - arr[i + 1]))
    for i in range(n - 2, 0, -1):
        ssum[i - 1] = ssum[i] + (1 if abs(arr[i - 1] - arr[i]) < abs(arr[i] - arr[i + 1]) else abs(arr[i] - arr[i - 1]))
    for _ in range(m):
        x, y = map(int, input().split())
        x -= 1
        y -= 1
        ans = psum[y] - psum[x] if y > x else ssum[y] - ssum[x]
        print(ans)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## D. Berserk Monsters

### Solution 1:  doubly linked list, set, prv, nxt arrays

```py
def main():
    n = int(input())
    attack = [0] + list(map(int, input().split()))
    defense = [0] + list(map(int, input().split()))
    prv, nxt = [0] * (n + 2), [0] * (n + 2)
    for i in range(1, n + 1):
        prv[i] = i - 1 if i > 0 else 0
        nxt[i] = i + 1 if i < n else n + 1
    ans = [0] * n
    in_bounds = lambda idx: 1 <= idx <= n
    def kill(idx):
        dmg = 0
        if in_bounds(prv[idx]): dmg += attack[prv[idx]]
        if in_bounds(nxt[idx]): dmg += attack[nxt[idx]]
        return dmg > defense[idx]
    alive = [1] * (n + 1)
    marked = set(range(1, n + 1))
    def populate():
        dead = []
        for i in marked:
            if kill(i):
                dead.append(i)
                alive[i] = 0
        return dead
    dead = populate()
    for r in range(n):
        marked.clear()
        for i in dead:
            prv[nxt[i]] = prv[i]
            nxt[prv[i]] = nxt[i]
            if in_bounds(prv[i]) and alive[prv[i]]:
                marked.add(prv[i])
            if in_bounds(nxt[i]) and alive[nxt[i]]:
                marked.add(nxt[i])
            ans[r] += 1
        dead = populate()
    print(*ans)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## E. Increasing Subsequences

### Solution 1:  bitmasks, bit manipulation

```py

```

## F. Replace on Segment

### Solution 1: 

```py

```