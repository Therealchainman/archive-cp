# Atcoder Beginner Contest 333

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

## 

### Solution 1:  Stern Brocot tree, farey tree, binary search, gcd to compute irriducible fraction

This is a solution that has worst case time complexity of O(p + q) I believe,  And it happens when you have something like 1/q,  a really large number, it will take q + 1 iterations to get the result for this one. 

Since N can be 10^10 this will time out.

```py
import math

def main():
    r = input()
    N = int(input())
    p, q = int(r[2:]), 10 ** (len(r) - 2)
    # convert to irreducible fraction
    g = math.gcd(p, q)
    while g != 1:
        p //= g
        q //= g
        g = math.gcd(p, q)
    if p <= N and q <= N: return print(p, q)
    # find the closest fraction
    a, b = 0, 1
    c, d = 1, 1
    while a + c <= N and b + d <= N:
        pm, qm = a + c, b + d
        if p * qm < q * pm:
            c, d = pm, qm
        else:
            a, b = pm, qm
    if b * q * (c * q - p * d) >= d * q * (p * b - a * q): print(a, b)
    else: print(c, d)

if __name__ == '__main__':
    main()
```

The logarithmic solution
binary search to find mediants


```py
import math

def main():
    r = input()
    N = int(input())
    p, q = int(r[2:]), 10 ** (len(r) - 2)
    # convert to irreducible fraction
    g = math.gcd(p, q)
    while g != 1:
        p //= g
        q //= g
        g = math.gcd(p, q)
    if p <= N and q <= N: return print(p, q)
    # find the closest fraction
    a, b = 0, 1
    c, d = 1, 1
    while a + c <= N and b + d <= N:
        pm, qm = a + c, b + d
        if p * qm < q * pm:
            left, right = 0, N
            while left < right:
                mid = (left + right + 1) >> 1
                if pm + mid * a > N or qm + mid * b > N: right = mid - 1; continue
                if q * (pm + mid * a) <= p * (qm + mid * b):
                    right = mid - 1
                else:
                    left = mid
            c = pm + left * a
            d = qm + left * b
        else:
            left, right = 0, N
            while left < right:
                mid = (left + right + 1) >> 1
                if pm + mid * c > N or qm + mid * d > N: right = mid - 1; continue
                if q * (pm + mid * c) <= p * (qm + mid * d):
                    left = mid
                else:
                    right = mid - 1
            a = pm + left * c
            b = qm + left * d
    if b * q * (c * q - p * d) >= d * q * (p * b - a * q): print(a, b)
    else: print(c, d)

if __name__ == '__main__':
    main()
```

