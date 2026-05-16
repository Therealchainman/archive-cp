# Longest Increasing Subsequence

Here is an algorithm that solves it using binary search and solves it in O(NlogN).  And it also has the ability to recover the index of the elements picked for the LIS.

This version computes the longest strictly increasing subsequence, where every
next value must be greater than the previous one. The important detail is the
binary search:

- Strictly increasing (`a1 < a2 < ...`): use `lower_bound`, the first position
  with value `>= arr[i]`. Equal values replace an existing tail, so they do not
  extend the subsequence.
- Weakly increasing / non-decreasing (`a1 <= a2 <= ...`): use `upper_bound`,
  the first position with value `> arr[i]`. Equal values are allowed to extend
  the subsequence.

So the only change for longest weakly increasing subsequence is:

```cpp
int idx = upper_bound(pool.begin(), pool.end(), arr[i]) - pool.begin();
```

For example, `[2, 2, 2]` has strict LIS length `1`, but weakly increasing
subsequence length `3`.

```cpp
vector<int> longestIncreasing(const vector<int>& arr) {
    int N = arr.size();
    vector<int> pool;
    vector<int> dp(N, 0);
    for (int i = 0; i < N; i++) {
        int idx = lower_bound(pool.begin(), pool.end(), arr[i]) - pool.begin();
        if (idx == pool.size()) {
            pool.emplace_back(arr[i]);
        } else {
            pool[idx] = arr[i];
        }
        dp[i] = idx + 1;
    }
    return dp;
}
```

Here `pool[len - 1]` stores the smallest possible ending value of a subsequence
with length `len`. `dp[i]` stores the length of the best subsequence ending at
`arr[i]`; if only the final LIS length is needed, it is `pool.size()` after the
loop.

Just need to track current coin_id with length j, and also update the previous pointer, so that it can traverse backwards and recover the elements that make up the LIS.

This needs to be modified, but it has the hard outline out for it. 

```cpp
vector<int> coin_id(N + 2, -1), prv(N + 2, -1);
vector<int> dp(N + 2, INF);
for (int i = 0; i <= N + 1; i++) {
    int j = upper_bound(dp.begin(), dp.end(), coins[i].c) - dp.begin();
    dp[j] = coins[i].c;
    coin_id[j] = i;
    if (j > 0) prv[i] = coin_id[j - 1];
    else prv[i] = -1;
}
int ans = upper_bound(dp.begin(), dp.end(), INF - 1) - dp.begin() - 1;
cout << ans - 1 << endl;
int u = coin_id[ans];
while (u != -1) {
    u = prv[u];
}
vector<int> path;
int v = coin_id[ans];
while (v != 0) {
    path.push_back(v);
    v = prv[v];
}
```
