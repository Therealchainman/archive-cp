# Codeforces Educational Round 153 Div 2

## A. Not a Substring

### Solution 1:  brute force + dfs + stack + early pruning of search space

I just perform a brute force, which should work because there are so many ways to perform early pruning which really reduces the search space significantly

```py
def main():
    s = input()
    n = 2 * len(s) 
    stk = [("(", 1)]
    while stk:
        seq, bal = stk.pop()
        if len(seq) == n:
            if bal == 0:
                if s not in seq:
                    print("YES")
                    print(seq)
                    return
            continue
        rem = n - len(seq)
        for ch in "()":
            if ch == "(":
                if rem > bal + 1:
                    stk.append((seq + ch, bal + 1))
            elif bal > 0:
                stk.append((seq + ch, bal - 1))
    print("NO")

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## B. Fancy Coins

### Solution 1:  math + greedy

The trick here is to use as much regular coins with k value over regular coins with 1 value.  Use as many k value coins as you can. 
Then you will be using some fancy, so now see how many fancy k coins you can remove, based on how many regular 1 value coins you have not used yet. 
It is always better to use fancy k value coins over fancy 1 value coins because you always need fewer k value coins than 1 value coin to equal an integer x.

```py
def main():
    m, k, a1, ak = map(int, input().split())
    use_k = m // k
    fancy_k = max(0, use_k - ak)
    if fancy_k == 0:
        return print(max(0, (m % k) - a1))
    rem = a1 - (m % k)
    if rem <= 0:
        return print(fancy_k + (m % k) - a1)
    x = rem // k
    fancy_k -= min(fancy_k, x)
    print(fancy_k)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## C. Game on Permutation

### Solution 1:  greedy + two bounding pointers

given current integer and it's index,
you increment the answer only if every integer to the left of it's index is in a decreasing order.
You can track the feasible range by left and right pointer. 

```py
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    queries = sorted([(p, i) for i, p in enumerate(arr)])
    right = left = n
    res = 0
    for p, i in queries:
        if left < i < right:
            res += 1
        if i > left:
            right = min(right, i)
        left = min(left, i)
    print(res)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
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

## 

### Solution 1: 

```py

```

