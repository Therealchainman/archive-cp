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

Count all the prime factors of a number in sqrt(num) time

```py
def prime_count(num):
    cnt = 0
    i = 2
    while i * i <= num:
        cnt += num % i == 0
        while num % i == 0:
            num //= i
        i += 1
    cnt += num > 1
    return cnt
```