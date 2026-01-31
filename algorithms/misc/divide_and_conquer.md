# Divide and Conquer

This algorithm works based on dividing the problem in midpoints.  And basically imagine it creates a binary tree.  And these are nodes, but the height of this tree is logN and the work that is done at each level in the tree needs to be N, for this to be super useful and give you an NlogN time complexity.


Interesting implementation for a problem.

```cpp
using int64 = long long;
const int64 INF = numeric_limits<int64>::max();
class Solution {
private:
    vector<int64> pre;
    int64 f(int64 n) {
        return n * (n + 1) / 2;
    }
    int64 cost(int l, int r) {
        return f(pre[r] - pre[l - 1]);
    }
    void dfs(const vector<int64>& src, vector<int64>& tgt, int l, int r, int pl, int pr) {
        if (l > r) return;
        int m = l + (r - l) / 2;
        int n = pl;
        int64 best = INF;
        for (int i = pl; i <= pr && i <= m; ++i) {
            if (src[i - 1] == INF) continue;
            int64 cand = src[i - 1] + cost(i, m);
            if (cand <= best) {
                best = cand;
                n = i;
            }
        }
        tgt[m] = best;
        dfs(src, tgt, l, m - 1, pl, n);
        dfs(src, tgt, m + 1, r, n, pr);
    }
public:
    int64 minPartitionScore(vector<int>& nums, int k) {
        int N = nums.size();
        pre.assign(N + 1, 0);
        for (int i = 0; i < N; ++i) {
            pre[i + 1] = pre[i] + nums[i];
        }
        vector<int64> dp(N + 1, INF), ndp(N + 1, 0);
        dp[0] = 0;
        while (k--) {
            ndp.assign(N + 1, INF);
            dfs(dp, ndp, 1, N, 1, N);
            swap(dp, ndp);
        }
        return dp[N];
    }
};
```