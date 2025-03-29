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

int b is the base, int p is the power, and int m is the modulus. 
return b^p % m

```cpp
int64 exponentiation(int64 b, int64 p, int64 m) {
    int64 res = 1;
    while (p > 0) {
        if (p & 1) res = (res * b) % m;
        b = (b * b) % m;
        p >>= 1;
    }
    return res;
}
```

without modulus

```cpp
int exponentiation(int b, int p) {
    int res = 1;
    while (p > 0) {
        if (p & 1) res = (res * b);
        b = (b * b);
        p >>= 1;
    }
    return res;
}
```