# Codeforces Round 920 Div 3

## B. Arranging Cats

### Solution 1:  greedy

```py
def main():
    N = int(input())
    start = list(map(int, input()))
    end = list(map(int, input()))
    ns, ne = start.count(1), end.count(1)
    ans = delta = abs(ns - ne)
    if ns > ne:
        for i in range(N):
            if delta == 0: break
            if start[i] == 1 and end[i] == 0: 
                start[i] = 0
                delta -= 1
    if ns > ne:
        for i in range(N):
            if delta == 0: break
            if start[i] == 0 and end[i] == 1: 
                start[i] = 1
                delta -= 1
    ans += sum(1 for i in range(N) if start[i] == 1 and end[i] == 0)
    print(ans)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## C. Sending Messages

### Solution 1:  greedy, simulation

```py
def main():
    # the number of messages, the initial phone's charge, the charge consumption per unit of time, and the consumption when turned off and on sequentially.
    n, f, a, b = map(int, input().split())
    # when messages need to be sent, strictly increasing order.
    arr = [0] + list(map(int, input().split()))
    for i in range(1, n + 1):
        d = arr[i] - arr[i - 1]
        if d * a <= b:
            f -= d * a
        else:
            f -= b
        if f <= 0: return print("NO")
    print("YES")

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## D. Very Different Array

### Solution 1:  dynamic programming, maximize from both sides

```py
def main():
    n, m = map(int, input().split())
    A = sorted(map(int, input().split()))
    B = sorted(map(int, input().split()))
    deltas = [0] * n
    for i, (a, b) in enumerate(zip(A, reversed(B))):
        deltas[i] = abs(a - b)
    for i, (a, b) in enumerate(zip(reversed(A), B)):
        deltas[n - i - 1] = max(deltas[n - i - 1], abs(a - b))
    print(sum(deltas))    

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## E. Eat the Chip

### Solution 1:  math, edge cases

```py
def main():
    h, w, xa, ya, xb, yb = map(int, input().split())
    if xa >= xb: return print("Draw")
    x_delta = xb - xa
    y_delta = abs(yb - ya)
    num_moves = x_delta // 2
    if x_delta & 1: # alice attacks
        if abs(ya - yb) <= 1: return print("Alice")
        wall_dist = w - yb if yb > ya else yb - 1
        if wall_dist <= num_moves - y_delta + 1: return print("Alice")
        else: return print("Draw")
    else: # bob attacks
        if ya == yb: return print("Bob")
        wall_dist = w - ya if ya > yb else ya - 1
        if wall_dist <= num_moves - y_delta: return print("Bob")
        else: return print("Draw")

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## F. Sum of Progression

### Solution 1:  dynamic programming, square root decomposition

```py

```

## G. Mischievous Shooter

### Solution 1:  manhattan distance,  prefix sums

```py

```

