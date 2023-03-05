"""
prime sieve to generate prime numbers

primes holds the prime integers for the lim integer

sieve holds the prime factors for each integer from 0, 1, 2, ..., lim
So it gives you the prime factorization of all numbers in the range. 

the time complexity is O(n log log n) most likely
"""
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