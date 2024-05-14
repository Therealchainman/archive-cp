# Dynamic Knapsack

This is an implementation of an advanced dynamic programming algorithm for the knapsack problem.  This is beyond the knapsack, and making it dynamic and gives you the ability to remove an element from the knapsack

This is also like a dynamic subset sum algorithm.

This has the ability to remove an element from a multiset, and then add it back, so you are able to rollback the knapsack for as if this element was not included in any of the subsets.  You then can determine if it is possible if the dp[i] > 0 still.  You do the modulus and hopefully it is never divisibley by MOD.  You can combat that even, by using randomness, and using rng() in C++, and getting a random modulus probably.

This is a reversible knapsack algorithm

```cpp
const int MOD = 1e9 + 7;
vector<int> arr = {3, 5, 8, 1};
int n = accumulate(arr.begin(), arr.end(), 0);
vector<int> dp(n + 1, 0);
dp[0] = 1;
for (int x : arr) {
    for (int i = n; i >= x; i--) {
        dp[i] = (dp[i] + dp[i - x]) % MOD;
    }
}
// rollback
for (int x : arr) {
    // remove
    for (int i = 0; i <= n; i++) {
        dp[i] = (dp[i] - dp[i - x] + MOD) % MOD;
    }

    // add
    for (int i = n; i >= x; i--) {
        dp[i] = (dp[i] + dp[i - x]) % MOD;
    }
}
```


```py
arr = [3, 5, 8, 1]
n = sum(arr) 
dp = [0] * (n + 1)
dp[0] = 1
for x in arr:
    for i in range(n, x - 1, -1):
        dp[i] += dp[i - x]
for x in arr:
    for i in range(x, n + 1):
        dp[i] -= dp[i- x]
    for i in range(n, x - 1, -1):
        dp[i] += dp[i - x]
```
