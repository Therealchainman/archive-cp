# Primality Test

## Deterministic Miller-Rabin Primality Test

This miller rabin primality test is deterministic for any 64 bit integer.  It has a good choice of bases for the primality test.  Really useful for determining large prime integers with up to 18 digits. 

This is something log(n) time, maybe not exactly but close to that for sure. 

Don't try to do this in C++, very hard to avoid 64 bit integer overflow. Python is much better for this.  The code is simple and easy to understand.  It is a probabilistic test, but it is deterministic for the bases used in the test.

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

## Simple prime test

Determines if n is a prime integer,  This is a slow one that runs in sqrt(n) time, so you can get away with integers with up to 16 digits or so.  It is slower than the method above.  So to test a lot of large integers this one quickly becomes too slow.

```py
def is_prime(n):
    if n < 2: return False
    if n == 2: return True
    if n % 2 == 0: return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0: return False
    return True
```