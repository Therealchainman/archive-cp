"""
prime sieve to generate prime numbers
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