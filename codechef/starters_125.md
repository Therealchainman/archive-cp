# Starters 125


## Binary Minimal

### Solution 1:  greedy

```py
def main():
    N, K = map(int, input().split())
    S = list(map(int, input()))
    if S.count(1) > K:
        for i in range(N):
            if K == 0: break
            if S[i] == 1:
                S[i] = 0
                K -= 1
        print("".join(map(str, S)))
    else:
        print("0" * (N - K))

if __name__ == "__main__":
    T = int(input())
    for _ in range(T):
        main()
```

## Bucket Game

### Solution 1:  greedy

```py
def main():
    N = int(input())
    arr = list(map(int, input().split()))
    A = B = cnt = s = 0
    for num in arr:
        if num == 1:
            if A > B: B += 1
            else: A += 1
        else:
            s += num
            cnt += 1
    if A > B:
        if s & 1:
            B += cnt
        else:
            A += cnt
    else:
        if s & 1:
            A += cnt
        else:
            B += cnt
    if A == B: print("Draw")
    elif A > B: print("Alice")
    else: print("Bob")

if __name__ == "__main__":
    T = int(input())
    for _ in range(T):
        main()
```

## Operating on A

### Solution 1:  prefix sums

```py
def main():
    N = int(input())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    ans = "YES" if sum(A) == sum(B) else "NO"
    print(ans)

if __name__ == "__main__":
    T = int(input())
    for _ in range(T):
        main()
```

## Manhattan Xor

### Solution 1: 

```py

```

## Lazy Rescue

### Solution 1: 

```py

```

## Painting Rectangles

### Solution 1:  euler tour, tree, sweepline, segment tree with point and range updates, lazy propagation, offline queries, mark active and inactive nodes in tree

```py

```