# Atcoder Regular Contest 169

## A - Please Sign 

### Solution 1:  

```py
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    P = list(map(lambda x: int(x) - 1, input().split()))
    depth = [0] * n
    for i in range(1, n):
        depth[i] = depth[P[i - 1]] + 1
    values = [0] * n
    for i in range(n):
        values[depth[i]] += arr[i]
    while values and values[-1] == 0:
        values.pop()
    if values and values[-1] > 0: return "+"
    if values and values[-1] < 0: return "-"
    return "0"

if __name__ == '__main__':
    print(main())
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

