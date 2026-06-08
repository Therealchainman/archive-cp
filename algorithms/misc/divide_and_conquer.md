# Divide and Conquer

This algorithm works based on dividing the problem in midpoints.  And basically imagine it creates a binary tree.  And these are nodes, but the height of this tree is logN and the work that is done at each level in the tree needs to be N, for this to be super useful and give you an NlogN time complexity.


## Divide and Conquer for DP


Generally we are looking for this type of dp, where we can make a divide and conquer optimization

dp[g][m] = min(dp[g-1][j] + cost(j+1, m)) for 0 <= j < m

dp[g][i] = best cost to split the first i items into exactly g groups.

You are “picking k of something” in the sense that you are making k partitions/groups/segments/choices.

Brute force way to solve the DP.

```cpp
for (int g = 1; g <= K; g++) {
    for (int i = 1; i <= N; i++) {
        dp[g][i] = INF;
        for (int j = 0; j < i; j++) {
            dp[g][i] = min(dp[g][i], dp[g - 1][j] + cost(j + 1, i));
        }
    }
}
```

This has time complexity O(KN^2) which is too slow for large N.  But if the cost function has some properties, we can optimize this to O(KNlogN) using divide and conquer.

For each fixed layer g, define:
opt[g][i] = argmin_j (dp[g-1][j] + cost(j+1, i))

Divide and conquer optimization applies when:
opt[g][i] <= opt[g][i+1]

So as i increases, the best split point never moves left.

That monotonicity lets you compute a whole DP row in:

O(NlogN)

or per layer: O(KNlogN) instead of O(KN^2).

**The recursive setup**

You compute one row g at a time.

Suppose you want to compute dp_cur[l...r], and you know all optimal split points must lie in [optL...optR].

pick the middle:
mid = (l + r) / 2;

Find the best j for dp_cur[mid], but only search:
j in [optL, optR]
Then recurse:
left half:  [l, mid - 1] with opt range [optL, bestJ]
right half: [mid + 1, r] with opt range [bestJ, optR]

Because of monotonicity, the best split point for the left half can’t be to the right of bestJ, and the best split point for the right half can’t be to the left of bestJ.

When i moves right, the best j also moves right or stays the same.

If confused about monotonicity:

Instead, it says the argmin location moves monotonically as i changes.

Wrong idea:
For fixed i, the cost is monotonic over j,
so best j is at an endpoint.

Correct idea:
For each i, the best j may be in the middle,
but as i increases, that best j does not move left.

```cpp
const long long INF = 4e18;

vector<long long> dp_prev, dp_cur;

long long cost(int l, int r) {
    // cost of grouping items l...r together
}

void compute(int l, int r, int optL, int optR) {
    if (l > r) return;

    int mid = (l + r) / 2;

    pair<long long, int> best = {INF, -1};

    int upper = min(mid - 1, optR);

    for (int j = optL; j <= upper; j++) {
        long long val = dp_prev[j] + cost(j + 1, mid);
        if (val < best.first) {
            best = {val, j};
        }
    }

    dp_cur[mid] = best.first;
    int opt = best.second;

    compute(l, mid - 1, optL, opt);
    compute(mid + 1, r, opt, optR);
}

dp_prev.assign(N + 1, INF);
dp_prev[0] = 0;

for (int g = 1; g <= K; g++) {
    dp_cur.assign(N + 1, INF);

    compute(1, N, 0, N - 1);

    dp_prev = dp_cur;
}

cout << dp_prev[N] << endl;
```

## Example problem


Split nums[0...N-1] into exactly k contiguous groups, minimizing the sum of each group’s score.

The group score: cost(l, r) = f(sum(nums[l-1...r-1]))

Your dp means:

dp[x] = minimum cost to partition the first x elements
        using the number of groups already processed

At the beginning:

dp[0] = 0;
dp[i] = INF for i > 0;

That means:

With 0 groups, I can cover 0 elements with cost 0.
With 0 groups, I cannot cover positive elements.

Then every while (k--) adds one more group.

So after one iteration, dp[i] means best cost for first i elements using 1 group.

After two iterations, dp[i] means best cost for first i elements using 2 groups.

After k iterations, dp[N] is the answer.

So the transition is:

ndp[m]=min(dp[i−1]+cost(i,m)) for 1 <= i <= m

This is the same as the usual form:

dp[g][m] = min(dp[g-1][j] + cost(j+1, m)) for 0 <= j < m

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