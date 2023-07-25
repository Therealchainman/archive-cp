# binary exponentiation

## iterative implementation without modulus

```py
def exponentiation(b, p):
    res = 1
    while p > 0:
        if p & 1:
            res *= b
        b *= b
        p >>= 1
    return res
```