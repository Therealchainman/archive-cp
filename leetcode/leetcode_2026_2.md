# Leetcode 2026 Part 2

# Leetcode Biweekly Contest 169

## 3738. Longest Non-Decreasing Subarray After Replacing at Most One Element

### Solution 1: dynamic programming, prefix/suffix run length

Find the maximum length of a contiguous subarray that can be made weakly increasing by modifying at most one element.

```cpp
class Solution {
public:
    int longestSubarray(vector<int>& nums) {
        int N = nums.size();
        vector<int> left(N, 1), right(N, 1);
        for (int i = 1; i < N; ++i) {
            if (nums[i] >= nums[i - 1]) left[i] = left[i - 1] + 1;
        }
        for (int i = N - 2; i >= 0; --i) {
            if (nums[i] <= nums[i + 1]) right[i] = right[i + 1] + 1;
        }
        int ans = min(N, *max_element(left.begin(), left.end()) + 1);
        for (int i = 1; i + 1 < N; ++i) {
            if (nums[i - 1] <= nums[i + 1]) ans = max(ans, left[i - 1] + 1 + right[i + 1]);
        }
        return ans;
    }
};
```

## 3739. Count Subarrays With Majority Element II
 
### Solution 1: fenwick tree, coordinate compressions, inversion counting, prefix sum

```cpp
using int64 = long long;
template <typename T>
struct FenwickTree {
    vector<T> nodes;
    T neutral;

    FenwickTree() : neutral(T(0)) {}

    void init(int n, T neutral_val = T(0)) {
        neutral = neutral_val;
        nodes.assign(n + 1, neutral);
    }

    void update(int idx, T val) {
        while (idx < (int)nodes.size()) {
            nodes[idx] += val;
            idx += (idx & -idx);
        }
    }

    T query(int idx) {
        T result = neutral;
        while (idx > 0) {
            result += nodes[idx];
            idx -= (idx & -idx);
        }
        return result;
    }

    T query(int left, int right) {
        return right >= left ? query(right) - query(left - 1) : T(0);
    }
};
class Solution {
public:
    int64 countMajoritySubarrays(vector<int>& nums, int target) {
        int N = nums.size();
        for (int i = 0; i < N; ++i) {
            if (nums[i] == target) nums[i] = 1;
            else nums[i] = -1;
            if (i > 0) nums[i] += nums[i - 1];
        }
        nums.insert(nums.begin(), 0);
        vector<int> values = nums;
        sort(values.begin(), values.end());
        values.erase(unique(values.begin(), values.end()), values.end());
        FenwickTree<int> seg;
        seg.init(N);
        int64 ans = 0;
        for (int x : nums) {
            int i = lower_bound(values.begin(), values.end(), x) - values.begin() + 1;
            ans += seg.query(i - 1);
            seg.update(i, 1);
        }
        return ans;
    }
};
```

# Leetcode Weekly Contest 508

## Q1. Maximum Total Sum of K Selected Elements

### Solution 1: greedy, sorting

Sort descending and take the k largest. The multiplier starts at `mul` and decreases by one per pick, so for each chosen element add `max(nums[i], mul * nums[i])` — applying the multiplier only when it helps (it can hurt once `mul` drops to 0 or for negative values), since the biggest values pair with the biggest multipliers.

```cpp
class Solution {
public:
    long long maxSum(vector<int>& nums, int k, int mul) {
        sort(nums.rbegin(), nums.rend());
        long long ans = 0;
        for (int i = 0; i < k; ++i, --mul) {
            ans += max<long long>(nums[i], 1LL * mul * nums[i]);
        }
        return ans;
    }
};
```

## Q2. Filter Occupied Intervals

### Solution 1: line sweep, events, intervals

Sweep over coordinate events: `+1` coverage at each occupied interval's left endpoint and `-1` just past its right endpoint (encoded so opening events sort before closing at the same coordinate). Clip everything to `[freeStart, freeEnd]` and emit a result interval for every maximal run where the active coverage count is zero, i.e. the gaps inside the free window not covered by any occupied interval.

```cpp
class Solution {
public:
    vector<vector<int>> filterOccupiedIntervals(vector<vector<int>>& occupiedIntervals, int freeStart, int freeEnd) {
        vector<vector<int>> ans;
        vector<pair<int, int>> events;
        for (const auto &p : occupiedIntervals) {
            int l = p[0], r = p[1];
            events.emplace_back(l, -1);
            events.emplace_back(r + 1, 1);
        }
        events.emplace_back(freeStart, -1);
        events.emplace_back(freeEnd + 1, 1);
        sort(events.begin(), events.end());
        int cnt = 0, start = -1;
        for (const auto &[e, i] : events) {
            if (e == freeStart) {
                if (cnt > 0 && start < e) {
                    ans.push_back({start, e - 1});
                }
                start = freeEnd + 1;
            }
            if (cnt == 0 && start < e) {
                start = e;
            }
            cnt -= i;
            if (cnt == 0 && start < e) {
                ans.push_back({start, e - 1});
            }
        }
        return ans;
    }
};
```

## Q3. Maximum Subarray Sum After Multiplier

### Solution 1: dynamic programming, Kadane's algorithm

A modified Kadane where exactly one contiguous middle segment of the chosen subarray is transformed (each element `x` becomes `k*x` in one pass, `x/k` in the other), and the two outer parts use the original values. Three rolling DP states track the best subarray sum ending at the current index that is, respectively, entirely *before* the transformed segment, currently *inside* it, or already *after* it; transitions only move forward through these phases. Run it once for multiply and once for divide and take the max.

```cpp
using int64 = long long;
const int64 INF = numeric_limits<int64>::max();
class Solution {
private:
    int64 calc(bool mult, int k, const vector<int> &nums) {
        int64 before = -INF, cur = -INF, after = -INF, ans = -INF;
        for (int x : nums) {
            int64 y = mult ? 1LL * k * x : x / k;
            int64 new_before = max<int64>(x, before == -INF ? -INF : before + x);
            int64 new_cur = max({y, before == -INF ? -INF : before + y, cur == -INF ? -INF : cur + y});
            int64 new_after = max(cur == -INF ? -INF : cur + x, after == -INF ? -INF : after + x);
            swap(before, new_before);
            swap(cur, new_cur);
            swap(after, new_after);
            ans = max({ans, cur, after});
        }
        return ans;
    }
public:
    int64 maxSubarraySum(vector<int>& nums, int k) {
        return max(calc(true, k, nums), calc(false, k, nums));
    }
};
```

## Q4. Minimum Time to Reach Target With Limited Power

### Solution 1: Dijkstra, shortest path, state-space search

Dijkstra over an expanded state `(node, power_remaining)`. Each traversal out of `u` spends `cost[u]` power (prune if it would go negative) and `w` time; `dp[node][power]` stores the best time to reach that state. The priority queue is ordered by time so the first time we settle the target gives the minimum time, and among equal-time arrivals we keep the one with the most remaining power as the tie-break.

```cpp
using int64 = long long;
const int64 INF = numeric_limits<int64>::max();
class Solution {
public:
    vector<int64> minTimeMaxPower(int n, vector<vector<int>>& edges, int power, vector<int>& cost, int source, int target) {
        vector<vector<int64>> dp(n, vector<int64>(power + 1, INF));
        priority_queue<tuple<int64, int, int>, vector<tuple<int64, int, int>>, greater<tuple<int64, int, int>>> minheap;
        vector<vector<pair<int, int>>> adj(n, vector<pair<int, int>>());
        for (const auto &edge : edges) {
            int u = edge[0], v = edge[1], w = edge[2];
            adj[u].emplace_back(v, w);
        }
        minheap.emplace(0, power, source);
        int64 minTime = INF, maxPower = -INF;
        while (!minheap.empty()) {
            auto [time, pow, u] = minheap.top();
            minheap.pop();
            if (u == target) {
                if (make_pair(time, -pow) < make_pair(minTime, -maxPower)) {
                    minTime = time;
                    maxPower = pow;
                }
            }
            for (const auto &[v, w] : adj[u]) {
                int npower = pow - cost[u];
                if (npower < 0) continue;
                int64 ncost = time + w;
                if (ncost < dp[v][npower]) {
                    dp[v][npower] = ncost;
                    minheap.emplace(ncost, npower, v);
                }
            }
        }
        if (minTime == INF) return {-1, -1};
        return {minTime, maxPower};
    }
};
```