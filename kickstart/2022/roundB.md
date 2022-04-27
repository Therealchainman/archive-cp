# Google Kickstart 2022 Round B

## Summary

Not too bad of a contest, I solved what I am capable of solving

## Infinity Area

### Solution 1: Simulation + Math

```py
from math import pi

def main():
    R, A, B = map(int,input().split())
    sum_area = 0
    while R > 0:
        # RIGHT CIRCLE
        sum_area += pi*R*R
        # LEFT CIRCLE
        R*=A
        sum_area += pi*R*R
        R//=B
    return sum_area


if __name__ == '__main__':
    T = int(input())
    for t in range(1,T+1):
        print(f"Case #{t}: {main()}")
```

## Palindromic Factors

### Solution 1: Efficient find factors + math

```py
from math import sqrt

def is_palindrome(num):
    return str(num) == str(num)[::-1]

def main():
    A = int(input())
    num_pal = 0
    seen = set()
    for i in range(1, int(sqrt(A))+1):
        if A%i==0:
            if i not in seen:
                if is_palindrome(i):
                    num_pal += 1
                seen.add(i)
            if A//i not in seen:
                if is_palindrome(A//i):
                    num_pal += 1
                seen.add(A//i)
    return num_pal


if __name__ == '__main__':
    T = int(input())
    for t in range(1,T+1):
        print(f"Case #{t}: {main()}")
```

## Unlock the Padlock

### Solution 1: BFS + memoization for best paths to unlock

```py
from collections import deque, defaultdict
from math import inf
def main():
    N, D = map(int,input().split())
    V = list(map(int,input().split()))
    compressed = [V[0]]
    best = inf
    for v in V[1:]:
        if compressed[-1] == v: continue
        compressed.append(v)
    queue = deque([(i, i, val, 0) for i, val in enumerate(compressed)])
    memo = defaultdict(lambda: inf)
    while queue:
        left, right, val, ops = queue.popleft()
        if right-left+1 == len(compressed) and val == 0:
            best = min(best, ops)
            continue
        if right-left+1 == len(compressed):
            queue.append((left, right, 0, min(ops+val,ops+D-val)))
            continue
        if left > 0 and right + 1 < len(compressed) and compressed[left-1] == compressed[right+1]:
            dist1 = (val - compressed[left-1])%D
            dist2 = (compressed[left-1] - val)%D
            cost1 = ops+dist1
            cost2 = ops+dist2
            if cost1 < memo[(left-1, right+1, compressed[left-1])]:
                memo[(left-1,right+1,compressed[left-1])] = cost1
                queue.append((left-1, right+1, compressed[left-1], cost1))
            if cost2 < memo[(left-1, right+1, compressed[left-1])]:
                memo[(left-1,right+1,compressed[left-1])] = cost2
                queue.append((left-1,right+1,compressed[left-1], cost2))
            continue
        if left > 0:
            dist1 = (val - compressed[left-1])%D
            dist2 = (compressed[left-1] - val)%D
            cost1 = ops+dist1
            cost2 = ops+dist2
            if cost1 < memo[(left-1,right, compressed[left-1])]:
                memo[(left-1,right,compressed[left-1])] = cost1
                queue.append((left-1, right, compressed[left-1], cost1))
            if cost2 < memo[(left-1,right, compressed[left-1])]:
                memo[(left-1,right,compressed[left-1])] = cost2
                queue.append((left-1, right, compressed[left-1], cost2))
        if right + 1 < len(compressed):
            dist1 = (val - compressed[right+1])%D
            dist2 = (compressed[right+1] - val)%D
            cost1, cost2 = ops+dist1, ops+dist2
            if cost1 < memo[(left, right+1,compressed[right+1])]:
                memo[(left,right+1,compressed[right+1])] = cost1
                queue.append((left, right+1,compressed[right+1], cost1))
            if cost2 < memo[(left, right+1,compressed[right+1])]:
                memo[(left,right+1,compressed[right+1])] = cost2
                queue.append((left, right+1,compressed[right+1], cost2))
    return best


if __name__ == '__main__':
    T = int(input())
    for t in range(1,T+1):
        print(f"Case #{t}: {main()}")
```

## Hamiltonian Tour

### Solution 1: 

```py

```