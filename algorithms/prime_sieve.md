# PRIME SIEVE

prime sieve to generate prime numbers

primes holds the prime integers for the lim integer

sieve holds the prime factors for each integer from 0, 1, 2, ..., lim
So it gives you the prime factorization of all numbers in the range. 

the time complexity is O(n log log n) most likely

## prime sieve that gets the prime factorization

This prime seive constructs the prime factorization of the composite integers 
This algorithm is rather slow actually, because it is storing the prime factorization of each integer.  There is a heavy constant factor overhead most likely. 

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

## fast prime sieve for prime factorizations of each integer

precomputes the prime factorization for each integer from 0 to upper_bound (inclusive).  This one is a bit worrisome, it may be too slow in actually.  A faster approach is below for how to get the prime factorization of integer queries. 

```py
def prime_sieve(upper_bound):
    prime_factorizations = [[] for _ in range(upper_bound + 1)]
    for i in range(2, upper_bound + 1):
        if len(prime_factorizations[i]) > 0: continue # not a prime
        for j in range(i, upper_bound + 1, i):
            prime_factorizations[j].append(i)
    return prime_factorizations
```

## prime sieve for smallest prime factor and fast prime factorization of integers

If you calculate the smallest prime factor for each integer, you can use that to speed up prime factorization of each integer from O(sqrt(n)) to O(log(n)).  While the prime sieve here is really fast and just nlog(log(n))

Just remember you want to call the sieve at the appropriate location, don't want to recompute it over and over it is a precomputation step that should only be done once. 

```py
def sieve(n):
    spf = [i for i in range(n + 1)]
    for i in range(2, n + 1):
        if spf[i] != i: continue
        for j in range(i * i, n + 1, i):
            spf[j] = i
    return spf

# log(n) factorize
def factorize(x):
    factors = []
    while x > 1:
        factors.append(spf[x])
        x //= spf[x]
    return factors
```

You can also count the number of prime integers in the prime factorization of an integer, excluding 1, which is not prime anyways. 

```py
def factorize(n):
    cnt = 0
    while n > 1:
        cnt += 1
        n //= spf[n]
    return cnt
```