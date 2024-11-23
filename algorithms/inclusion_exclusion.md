# Principle of Inclusion Exclusion

|A U B| = |A| + |B| - |A ∩ B|
size of union when size of intersection is known. 

## An example

suppose you want to count the number of people with a cat or dog or fish as pet.  Suppose you have these elements, c, d, f, cd, cf, df, cdf.  Then if you split that into the sets of cat A_cat = {c, cd, cf, cdf}, and sets of dogs A_dog = {d, cd, df, cdf}, and sets containing fish A_fish = {f, cf, df, cdf}.  Then you want to use principle of inclusion and exclusion because the first time you can realize you are going to end up counting each group of elements, including intersections (yeah)
(c, 1), (d, 1), (f, 1), (cd, 2), (cf, 2), (df, 2), (cdf, 3),  so you double counted the intersections of size 2 and triple counted the intersections of size 3.  So you correct for this by subtracting intersections of size 2.  and so on. 

## Counting the number of coprime integers

This is an example problem you can use PIE. 

This counts the number of integers in the range [1, x] that are coprime with m.  Also you can easily calculate the inverse as well by switching the signs and everything.  That is you can calculate the number of integers in range [1, x] that are not coprime with integer m.  or just take x - num_coprimes(x, m) to get that. 

```cpp
// count number of coprimes of x to m
int num_coprimes(int x, int m) {
    int ans = x;
    int endMask = 1 << primes[m].size();
    for (int mask = 1; mask < endMask; mask++) {
        int numBits = 0, val = 1;
        for (int i = 0; i < primes[m].size(); i++) {
            if ((mask >> i) & 1) {
                numBits++;
                val *= primes[m][i];
            }
        }
        if (numBits % 2 == 0) { // even add
            ans += x / val;
        } else {
            ans -= x / val;
        }
    }
    return ans;
}
```

## dynamic programming implementation of technique

In this case you can use the inclusion and exclusion technique to remove duplicates coming in the form of if a factor is divisible by another factor such as x is divisibly by y, then x will contain duplicates from y.  

```py
factors = []
for i in range(1, n):
    # i is factor if n is divisible by i
    if n % i == 0: factors.append(i)
m = len(factors)
dp = [[0] * m for _ in range(m)]
for i in range(m):
    dp[i][i] = 1 # i is divisible by i
    for j in range(i):
        if factors[i] % factors[j] == 0: dp[i][j] = 1 # factor_i is divisible by factor_j
# count the ways
counts = [0] * m
for i in range(m):
    # finds position that must be fixed, that is takahashi doesn't work that day so that '.'
    fixed = [0] * factors[i]
    for j in range(n):
        if s[j] == '.': fixed[j % factors[i]] = 1
    unset = factors[i] - sum(fixed)
    counts[i] = pow(2, unset, mod)
# dynamic programming to remove the duplicates
for i in range(m):
    for j in range(i):
        if dp[i][j]:
            counts[i] -= counts[j]
```