# PRIME SIEVE

prime sieve to generate prime numbers

primes holds the prime integers for the lim integer

sieve holds the prime factors for each integer from 0, 1, 2, ..., lim
So it gives you the prime factorization of all numbers in the range. 

the time complexity is O(n log log n) most likely

## prime sieve that gets the prime factorization

This prime seive constructs the prime factorization of the composite integers 

```py
def prime_sieve(lim):
    sieve,primes = [[] for _ in range(lim)], []
    for integer in range(2,lim):
        if not len(sieve[integer]):
            primes.append(integer)
            for possibly_divisible_integer in range(integer,lim,integer):
                current_integer = possibly_divisible_integer
                while not current_integer%integer:
                    sieve[possibly_divisible_integer].append(integer)
                    current_integer //= integer
    return primes
```

## fast prime sieve

This prime sieve will just return if prime or not in an arrays.  So an array will represent the status of each integer in a continuous range whether it is prime or not prime (composite)

```py
def prime_sieve(lim):
    primes = [1] * lim
    primes[0] = primes[1] = 0
    p = 2
    while p * p <= lim:
        if primes[p]:
            for i in range(p * p, lim, p):
                primes[i] = 0
        p += 1
    return primes
```