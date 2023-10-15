# BINOMIAL DISTRIBUTION


probabilty of getting exactly m successes in n trials with a probability of success p
binomial distribution for when you have independent events
solve for probability of getting m heads from n coin flips with probability of heads p = 0.5

```py
def pmf_binomial_distribution(n, m, p):
    return math.comb(n, m) * pow(p, m) * pow(1 - p, n - m)
```

alternative for when math.comb is not availabe with a class


```py
mod = int(1e9) + 7

def mod_inverse(v):
    return pow(v, mod - 2, mod)

def factorials(n):
    fact, inv_fact = [1] * (n + 1), [0] * (n + 1)
    for i in range(2, n + 1):
        fact[i] = (fact[i - 1] * i) % mod
    inv_fact[-1] = mod_inverse(fact[-1])
    for i in reversed(range(n)):
        inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % mod
    return fact, inv_fact

def nCr(n, r):
    return (fact[n] * inv_fact[r] * inv_fact[n - r]) % mod if n >= r else 0
```

alternative for when math.comb is not available with a class, although appears to be slower than implementation above

```py
def mod_inverse(num, mod):
    return pow(num, mod - 2, mod)

"""
inverse factorials depends on factorials, so that needs to be called first
"""
class Factorials:
    def __init__(self, size, mod):
        self.fact = []
        self.inv_fact = []
        self.size = size
        self.mod = mod
    
    def factorials(self):
        self.fact = [1]*(self.size + 1)
        for i in range(1, self.size + 1):
            self.fact[i] = (self.fact[i - 1] * i) % self.mod
        return self.fact
    
    def inverse_factorials(self):
        assert len(self.fact) > 0, "Factorials need to be calculated first"
        self.inv_fact = [1]*(self.size + 1)
        self.inv_fact[-1] = mod_inverse(self.fact[-1], self.mod)
        for i in range(self.size - 1, -1, -1):
            self.inv_fact[i] = (self.inv_fact[i + 1] * (i + 1)) % self.mod
        return self.inv_fact
    
class Combinations(Factorials):
    def __init__(self, size, mod):
        super().__init__(size, mod)
        self.fact = self.factorials()
        self.inv_fact = self.inverse_factorials()
        self.mod = mod
    
    def nCr(self, n, r):
        return (self.fact[n] * self.inv_fact[r] * self.inv_fact[n - r]) % self.mod

class BinomialDistribution(Combinations):
    def __init__(self, size, mod):
        super().__init__(size, mod)
        self.mod = mod

    def pmf_binomial_distribution(self, n, m, p):
        return self.nCr(n, m) * pow(p, m, self.mod) * pow(1 - p, n - m, self.mod)
```