# Leetcode Weekly Contest 407

## Minimum Operations to Make Array Equal to Target

### Solution 1:  min segment tree, point updates, range queries, PURQ, difference array

```cpp
const int INF = 1e9;
struct SegmentTree {
    int size;
    vector<int> nodes, index;

    void init(int num_nodes) {
        size = 1;
        while (size < num_nodes) size *= 2;
        nodes.assign(size * 2, 0);
        index.assign(size * 2, 0);
    }

    int func(int x, int y) {
        return min(x, y);
    }

    void ascend(int segment_idx) {
        while (segment_idx > 0) {
            int left_segment_idx = 2 * segment_idx, right_segment_idx = 2 * segment_idx + 1;
            int left_min = nodes[left_segment_idx], right_min = nodes[right_segment_idx];
            if (left_min < right_min) index[segment_idx] = index[left_segment_idx];
            else index[segment_idx] = index[right_segment_idx];
            nodes[segment_idx] = func(nodes[left_segment_idx], nodes[right_segment_idx]);
            segment_idx >>= 1;
        }
    }

    void update(int segment_idx, int idx, int val) {
        segment_idx += size;
        nodes[segment_idx] = val;
        index[segment_idx] = idx;
        segment_idx >>= 1;
        ascend(segment_idx);
    }

    pair<int, int> query(int left, int right) {
        left += size, right += size;
        int res = INF, idx = 0;
        while (left <= right) {
            if (left & 1) {
                if (nodes[left] < res) {
                    idx = index[left];
                    res = nodes[left];
                }
                left++;
            }
            if (~right & 1) {
                if (nodes[right] < res) {
                    res = nodes[right];
                    idx = index[right];
                }
                right--;
            }
            left >>= 1, right >>= 1;
        }
        return make_pair(res, idx);
    }
};

SegmentTree seg;

long long calc(int l, int r, long long p) {
    long long res = 0;
    if (l > r) return res;
    auto [val, idx] = seg.query(l, r);
    long long delta = val - p;
    res += delta;
    res += calc(l, idx - 1, p + delta) + calc(idx + 1, r, p + delta);
    return res;
}

class Solution {
public:
    long long minimumOperations(vector<int>& nums, vector<int>& target) {
        int N = nums.size();
        seg.init(N);
        long long ans = 0;
        vector<int> diff(N, 0);
        for (int i = 0; i < N; i++) {
            diff[i] = target[i] - nums[i];
        }
        vector<pair<int, int>> ranges;
        int sign = 0, last = 0;
        for (int i = 0; i < N; i++) {
            int d = diff[i];
            seg.update(i, i, abs(d));
            if (d > 0) {
                if (sign <= 0) {
                    if (sign < 0) ranges.emplace_back(last, i - 1);
                    last = i;
                    sign = 1;
                }
            } else if (d < 0) {
                if (sign >= 0) {
                    if (sign > 0) ranges.emplace_back(last, i - 1);
                    last = i;
                    sign = -1;
                }
            } else {
                if (sign != 0) ranges.emplace_back(last, i - 1);
                sign = 0;
                last = i;
            }
        }
        if (sign != 0) ranges.emplace_back(last, N - 1);
        for (auto [l, r] : ranges) {
            ans += calc(l, r, 0);
        }
        return ans;
    }
};
```