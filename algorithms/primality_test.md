# Primality Test

## Deterministic Miller-Rabin Primality Test

This miller rabin primality test is deterministic for any 64 bit integer.  It has a good choice of bases for the primality test.  Really useful for determining large prime integers with up to 18 digits. 

This is something log(n) time, maybe not exactly but close to that for sure. 

```py
def check_composite(n, a, d, s):
    x = pow(a, d, n)
    if x == 1 or x == n - 1: return False
    for r in range(1, s):
        x = x * x % n
        if x == n - 1: return False
    return True

def miller_rabin(n):
    if n < 4: return n == 2 or n == 3
    r = 0
    d = n - 1
    while d % 2 == 0:
        r += 1
        d >>= 1
    bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for a in bases:
        if n == a: return True
        if check_composite(n, a, d, r): return False
    return True
```