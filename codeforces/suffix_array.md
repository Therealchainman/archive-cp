

## A. Suffix Array - 1

### Solution 1:  suffix array algorithm

```py
def suffix_array(s: str) -> str:
    n = len(s)
    p, c = [0]*n, [0]*n
    arr = [None]*n
    for i, ch in enumerate(s):
        arr[i] = (ch, i)
    arr.sort()
    for i, (_, j) in enumerate(arr):
        p[i] = j
    c[p[0]] = 0
    for i in range(1,n):
        c[p[i]] = c[p[i-1]] + (arr[i][0] != arr[i-1][0])
    k = 1
    is_finished = False
    while k < n and not is_finished:
        for i in range(n):
            arr[i] = (c[i], c[(i+k)%n], i)
        arr.sort()
        for i, (_, _, j) in enumerate(arr):
            p[i] = j
        c[p[0]] = 0
        is_finished = True
        for i in range(1,n):
            c[p[i]] = c[p[i-1]] + (arr[i][:2] != arr[i-1][:2])
            is_finished &= (c[p[i]] != c[p[i-1]])
        k <<= 1
    return ' '.join(map(str, p))

def main():
    s = input() + '$'
    return suffix_array(s)

if __name__ == '__main__':
    print(main())
```

## A. Suffix Array - 2

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