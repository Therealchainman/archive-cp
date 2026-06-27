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