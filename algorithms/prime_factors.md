# PRIME FACTORS

Get all the prime factors of a number n in sqrt(num) time

```py
def prime_factors(num: int) -> List[int]:
    factors = []
    while num % 2 == 0:
        factors.append(2)
        num //= 2
    for i in range(3, math.isqrt(num) + 1, 2):
        while num % i == 0:
            factors.append(i)
            num //= i
    if num > 2:
        factors.append(num)
    return factors
```