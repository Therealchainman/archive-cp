# Lagrangian Relaxation (Alien's Trick)

Exact count constraint+additive cost+DP-able relaxed problem

Lagrangian relaxation is an optimization technique used to solve complex constrained problems by moving difficult constraints into the objective function with a penalty. It simplifies the problem into an "easy" subproblem that is efficiently solvable, generating tight mathematical bounds for integer or linear programming

You replace "must use exactly K" with "using one more costs λ", then tune λ until the unconstrained optimum behaves like the constrained optimum.


## Implementation example

Just note you usually tune λ using binary search, and the unconstrained optimum is often found using a greedy or dynamic programming algorithm.

```cpp
using int64 = long long;
const int64 INF = numeric_limits<int64>::max();
class Solution {
private:
    vector<int64> pref;
    pair<int64, int> solve_penalty(int l, int r, int64 lambda) {
        int N = pref.size() - 1;
        vector<int64> dp(N + 1, -INF);
        vector<int> cnt(N + 1, 0);
        deque<tuple<int, int64, int>> dq;
        for (int i = 1; i <= N; ++i) {
            dp[i] = dp[i - 1];
            cnt[i] = cnt[i - 1];
            int j = i - l;
            if (j >= 0) {
                int64 cand = dp[j];
                int count = cnt[j];
                if (cand < 0) {
                    cand = 0;
                    count = 0;
                }
                cand -= pref[j];
                while (!dq.empty()) {
                    auto [x, y, z] = dq.back();
                    if (make_pair(cand, -count) < make_pair(y, -z)) break;
                    dq.pop_back();
                }
                dq.emplace_back(j, cand, count);
            }
            // remove those that are no longer valid from front of dq
            while (!dq.empty() && get<0>(dq.front()) < i - r) {
                dq.pop_front();
            }
            if (!dq.empty()) { // choose subarray ending at i - 1
                auto [j, ncost, ncount] = dq.front();
                ncost += pref[i] - lambda;
                ncount++;
                if (ncost > dp[i] || (ncost == dp[i] && ncount < cnt[i])) {
                    dp[i] = ncost;
                    cnt[i] = ncount;
                }
            }
        }
        return {dp[N], cnt[N]};
    }
public:
    int64 maximumSum(vector<int>& nums, int m, int l, int r) {
        int N = nums.size();
        pref.assign(N + 1, 0);
        for (int i = 0; i < N; ++i) {
            pref[i + 1] += pref[i] + nums[i];
        }
        int64 lo = 0, hi = 1e18;
        while (lo < hi) {
            int64 mid = lo + (hi - lo) / 2;
            auto [penalizedCost, cnt] = solve_penalty(l, r, mid);
            if (cnt > m) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        auto [penalizedCost, cnt] = solve_penalty(l, r, lo);
        int64 ans = penalizedCost + 1LL * lo * m;
        return ans;
    }
};
```