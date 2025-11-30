# sum of prime factors

This problem is finding sum of prime factors until a fixed pivot is reached which will either be 0, 4 or a prime factor

```py
def sopf(n: int) -> int:
    sum_ = 0
    for i in range(2, int(math.sqrt(n)) + 1):
        if n < i: break
        while n%i == 0:
            sum_ += i
            n //= i
    return sum_ + (n if n > 1 else 0)

def pivot_sopf(n: int) -> int:
    while n != (n := sopf(n)): pass
    return n
```