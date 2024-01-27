# Stern Brocot Tree

## Introduction 

An elegant construction that intertwines number theory, fractions, and binary search trees. Named after Moritz Stern and Achille Brocot, this tree provides a unique way of systematically enumerating all positive rational numbers without repetition.

## The Structure of the Tree

The tree grows by finding the mediant of two neighboring fractions. The mediant of two fractions, a/b and c/d, is obtained by adding the numerators and the denominators separately, resulting in (a + c) / (b + d). This process is applied iteratively, starting with 0/1 and 1/0 at the top. The first level of the tree contains the fraction 1/1, the mediant of 0/1 and 1/0. Each subsequent level is formed by taking the mediants of each pair of neighboring fractions on the previous level.

## Properties and Applications

One of the most remarkable properties of the Stern-Brocot Tree is that every positive rational number appears exactly once. This tree can be used to systematically list fractions in their reduced form. It also has applications in algorithms, particularly in areas requiring efficient enumeration or approximation of rational numbers.

## Using it to find the nearest fraction to some rational number r

Provided you can form any rational number as long the numerator and denominator integer is less than 10^10. 

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

## Implmentation to solve many queries on Stern Brocot Tree

Finding the range, encoding and decoding path.  Finding the LCA of two rational numbers.  Finding the ancestor at k depth of a rational number.

```py
UPPERBOUND = 10**9

# p1 / q1 < p2 / q2
def less(p1, q1, p2, q2):
    return p1 * q2 < p2 * q1

# p1 / q1 > p2 / q2
def greater(p1, q1, p2, q2):
    return p1 * q2 > p2 * q1

def encode_path(p, q):
    path = []
    # l = a / b
    a, b = 0, 1
    # m = c / d
    c, d = 1, 1
    # h = e / f
    e, f = 1, 0
    while (c, d) != (p, q):
        if less(p, q, c, d): # left
            left, right = 0, UPPERBOUND
            while left < right:
                mid = (left + right) >> 1
                nc, nd = c + mid * a, d + mid * b
                # if nc / nd > p / q
                if less(p, q, nc, nd):
                    left = mid + 1
                else:
                    right = mid
            c, d = c + left * a, d + left * b
            e, f = c - a, d - b
            path.append(("L", left))
        else: # right
            left, right = 0, UPPERBOUND
            while left < right:
                mid = (left + right) >> 1
                nc, nd = c + mid * e, d + mid * f
                # if nc / nd < p / q
                if greater(p, q, nc, nd):
                    left = mid + 1
                else:
                    right = mid
            c, d = c + left * e, d + left * f
            a, b = c - e, d - f
            path.append(("R", left))
    return path
   
def decode_path(path):
    # l = a / b
    a, b = 0, 1
    # m = c / d
    c, d = 1, 1
    # h = e / f
    e, f = 1, 0
    for ch, n in path:
        if ch == "L":
            c, d = c + n * a, d + n * b
            e, f = c - a, d - b
        else:
            c, d = c + n * e, d + n * f
            a, b = c - e, d - f
    return (c, d)

# TODO: can I create a Fraction class or use python's builtin
def lca(p1, q1, p2, q2):
    # l = a / b
    a, b = 0, 1
    # m = c / d
    c, d = 1, 1
    # h = e / f
    e, f = 1, 0
    while (less(p1, q1, c, d) and less(p2, q2, c, d)) or (greater(p1, q1, c, d) and greater(p2, q2, c, d)):
        if less(p1, q1, c, d) and less(p2, q2, c, d): # left
            left, right = 0, UPPERBOUND
            while left < right:
                mid = (left + right) >> 1
                nc, nd = c + mid * a, d + mid * b
                if less(p1, q1, nc, nd) and less(p2, q2, nc, nd):
                    left = mid + 1
                else:
                    right = mid
            c, d = c + left * a, d + left * b
            e, f = c - a, d - b
        else:
            left, right = 0, UPPERBOUND
            while left < right:
                mid = (left + right) >> 1
                nc, nd = c + mid * e, d + mid * f
                if greater(p1, q1, nc, nd) and greater(p2, q2, nc, nd):
                    left = mid + 1
                else:
                    right = mid
            c, d = c + left * e, d + left * f
            a, b = c - e, d - f
    return (c, d)

def ancestor(k, p, q):
    path = encode_path(p, q)
    depth = sum(n for _, n in path)
    if depth < k: return print(-1)
    while depth > k:
        ch, n = path.pop()
        delta = min(n, depth - k)
        n -= delta
        depth -= delta
        if n > 0:
            path.append((ch, n))
    print(*decode_path(path))

def range_(p, q):
    # l = a / b
    a, b = 0, 1
    # m = c / d
    c, d = 1, 1
    # h = e / f
    e, f = 1, 0
    while (c, d) != (p, q):
        if less(p, q, c, d): # left
            left, right = 0, UPPERBOUND
            while left < right:
                mid = (left + right) >> 1
                nc, nd = c + mid * a, d + mid * b
                # if nc / nd > p / q
                if less(p, q, nc, nd):
                    left = mid + 1
                else:
                    right = mid
            c, d = c + left * a, d + left * b
            e, f = c - a, d - b
        else: # right
            left, right = 0, UPPERBOUND
            while left < right:
                mid = (left + right) >> 1
                nc, nd = c + mid * e, d + mid * f
                # if nc / nd < p / q
                if greater(p, q, nc, nd):
                    left = mid + 1
                else:
                    right = mid
            c, d = c + left * e, d + left * f
            a, b = c - e, d - f
    return (a, b, e, f)

def main():
    T = int(input())
    for _ in range(T):
        query = input().split()
        # TODO: convert these to enums
        if query[0] == "ENCODE_PATH":
            p, q = map(int, query[1:])
            path = encode_path(p, q)
            ans = [len(path)]
            for c, n in path:
                ans.extend([c, n])
            print(*ans)
        elif query[0] == "DECODE_PATH":
            path = query[2:]
            path = [(path[i], int(path[i + 1])) for i in range(0, len(path), 2)]
            print(*decode_path(path))
        elif query[0] == "LCA":
            p1, q1, p2, q2 = map(int, query[1:])
            print(*lca(p1, q1, p2, q2))
        elif query[0] == "ANCESTOR":
            k, p, q = map(int, query[1:])
            ancestor(k, p, q)
        elif query[0] == "RANGE":
            p, q = map(int, query[1:])
            print(*range_(p, q))

if __name__ == '__main__':
    main()
```