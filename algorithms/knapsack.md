# Knapsack Algorithms

## knapsack algorithm for counting number of ways to form subset sums

```py
for i in range(k, x - 1, -1):
    dp[i] += dp[i - x]
```

## knapsack algorithm for removing element from subset sums

```py
for i in range(x, k + 1):
    dp[i] -= dp[i - x]
```