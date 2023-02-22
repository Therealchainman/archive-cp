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