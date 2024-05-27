# Codeforces Round 947 div1

## 

### Solution 1: 

```py

```

## E. Chain Queries

### Solution 1: 

```py

```

## F. Set

### Solution 1:  set theory, bitmasks, binary encoding

```py
n = 3
V = [0] + [15, 15, 15, 15, 15, 15, 12]
arr = []
for S in range(1, 1 << n):
    flag = True
    for T in range(1, 1 << n):
        cnt = (S & T).bit_count()
        if not ((V[T] >> cnt) & 1): 
            flag = False
            break
    if flag: arr.append(S)
print(*arr)
```

## G. Zimpha Fan Club

### Solution 1: 

```py

```

## H. 378QAQ and Core

### Solution 1: 

```py

```

## I. Mind Bloom

### Solution 1: 

```py

```