# Leetcode BiWeekly Contest 133

## 3193. Count the Number of Inversions

### Solution 1:  dynamic programming, pointer, window sum optimization, space optimization

```cpp
class Solution {
public:
    int numberOfPermutations(int n, vector<vector<int>>& requirements) {
        sort(requirements.begin(), requirements.end());
        int m = requirements.size();
        const int COUNT = 400, MOD = 1e9 + 7;
        vector<int> dp(COUNT + 1, 0), dp1(COUNT + 1);
        dp[0] = 1;
        for (int i = 0, j = 0; i < n; i++) {
            int mark = -1, wsum = 0;
            if (j < m && requirements[j][0] == i) {
                mark = requirements[j][1];
                j++;
            }
            for (int k = 0; k <= COUNT; k++) {
                wsum = (wsum + dp[k]) % MOD;
                dp1[k] = wsum;
                if (mark != -1 && k != mark) dp1[k] = 0;
                if (k >= i) wsum = (wsum - dp[k - i] + MOD) % MOD;
            }
            swap(dp, dp1);
        }
        return dp[requirements.end()[-1][1]];
    }
};
```