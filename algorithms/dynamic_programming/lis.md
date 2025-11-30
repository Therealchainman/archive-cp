# Longest Increasing Subsequence

Here is an algorithm that solves it using binary search and solves it in O(NlogN).  And it also has the ability to recover the index of the elements picked for the LIS.

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
        dp[i] = pool.size();
    }
    return dp;
}
```

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