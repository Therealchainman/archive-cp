# Catalan's Numbers

So how can you identify if a problem can be solved with catalan's numbers? 
Find if you can split the problem into two parts of size k and n - k

## iterative dp + O(n^2) time

```py
dp = [0]*(n + 1)
dp[0] = 1
for i in range(1, n + 1):
    for j in range(i):
        dp[i] += dp[j]*dp[i - j - 1]
```

## binomial coefficient formula

```py
math.comb(2*n, n)//(n + 1)
```

## analytical formula + O(n) time

```py
cn = 1
for i in range(1, numPeople//2 + 1):
    cn = (2*(2*i - 1)*cn)//(i + 1)
```

## Catalan numbers in python

fasest enough so far implementation to get catalan numbers

```py
from collections import Counter
 
MOD = 998244353
 
def singleton(__init__):
    def __new__(cls, *args, **kwargs):
        if hasattr(cls, 'instance'):
            return getattr(cls, 'instance')
        instance = super(cls, cls).__new__(cls)
        __init__(instance, *args, **kwargs)
        setattr(cls, 'instance', instance)
        return instance
    return __new__
 
 
class Inverse:
    @singleton
    def __new__(cls):
        cls.inverse = {1: 1}
 
    def __getitem__(self, item):
        for i in range(len(self.inverse) + 1, item + 1):
            self.inverse[i] = self.inverse[MOD % i] * (MOD - MOD // i) % MOD
        return self.inverse[item]
 
 
class Catalan:
    @singleton
    def __new__(cls):
        cls.catalan = [1]
 
    def __getitem__(self, item):
        for i in range(len(self.catalan), item + 1):
            self.catalan.append(self.catalan[-1] * (4 * i - 2) * Inverse()[i + 1] % MOD)
        return self.catalan[item]
```

To create the catalan number

```py
Catalan()[count >> 1]
```