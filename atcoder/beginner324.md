# Atcoder Beginner Contest 324

## B - 3-smooth Numbers 

### Solution 1: 

```py
def main():
    N = int(input())
    for x in range(64):
        a = pow(2, x)
        if a > N: continue
        for y in range(40):
            b = pow(3, y)
            if b > N: continue
            if a * b > N: continue
            if a * b == N: return print("Yes")
    print("No")

if __name__ == '__main__':
    main()
```

## C - Error Correction 

### Solution 1: 

```py
def main():
    N, T = input().split()
    N = int(N)
    works = [0] * N
    for i in range(N):
        S = input()
        if len(S) == len(T): # changed 0 or 1 character in string
            hamming_dist = sum(1 for x, y in zip(S, T) if x != y)
            works[i] = int(hamming_dist <= 1)
        elif len(S) == len(T) + 1: # inserted 1 character
            j = 0
            for ch in S:
                if j < len(T) and ch == T[j]:
                    j += 1
            works[i] = int(j == len(T))
        elif len(S) == len(T) - 1: # deleted 1 character
            j = 0
            for ch in T:
                if j < len(S) and ch == S[j]:
                    j += 1
            works[i] = int(j == len(S))
    K = sum(works)
    print(K)
    print(" ".join(map(str, [i + 1 for i in range(N) if works[i]])))

if __name__ == '__main__':
    main()
```

## D - Square Permutation 

### Solution 1: 

precompute all the squares in less than 10^7 operations
The frequency of the digits in the squares should be the same as that in S if it is a permutation
However you can ignore 0s. 

```py

```

## E - Joint Two Strings 

### Solution 1: 

```py
def main():
    N, T = input().split()
    N = int(N)
    arr = [None] * N
    pindex = [0] * N
    for i in range(N):
        arr[i] = input()
        j = 0
        for ch in arr[i]:
            if j == len(T): break
            if ch == T[j]: j += 1
        pindex[i] = j
    pindex.sort(reverse = True)
    def bsearch(target):
        left, right = 0, N
        while left < right:
            mid = (left + right) >> 1
            if pindex[mid] >= target:
                left = mid + 1
            else:
                right = mid
        return left
    res = 0
    for i in range(N):
        j = 0
        for ch in reversed(arr[i]):
            if j == len(T): break
            if ch == T[len(T) - j - 1]: j += 1
        k = bsearch(len(T) - j)
        res += k
    print(res)

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

