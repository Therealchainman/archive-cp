# Infinite Geometric Series

## Example derivative of infinite geometric series

This can be found by differentiating an infinite geometric series.  this may be unrelated but interesting result.

```py
# p + 2*p^2 + 3*p^3 + 4*p^4 + ... = p/(1-p)^2
def inf_geometric_series(p):
    return p / (1 - p) ** 2
```


## Geometric Distribution

Often times this is useful for when you are dealing with geometric distribution.  Which is the tells you when something will happen? For example, if you are flipping a coin, how many times will you have to flip it before you get a head?  Or shooting a basketball, how many times you have to shoot to make a baseket given some probability per shot.

```py
def mod_inverse(v):
    return pow(v, MOD - 2, MOD)

def infinite_geometric_series(a, r):
    return (a * mod_inverse(1 - r)) % MOD

geo = infinite_geometric_series(1, p) # 1 + p^ + p^2, a + ar + ar^2 + ...
```


Suppose p is the probability of success, the probability of getting your first success on kth trial is P(x = k) = (1 - p)^(k - 1)*p
And the probability of eventually getting a success is obviously 1

But we are more interested in the expected number of trials to eventually get a success.

The expected value of geometric distribution is 1/p

important to realize that p represents the probability of a success at each instance of taking an action.




