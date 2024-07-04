# Leetcode Weekly Contest 401

## 3181. Maximum Total Reward Using Operations II

### Solution 1:  bit manipulation, bitsets, dynamic programming, reachability, sorting

```cpp
class Solution {
public:
    int maxTotalReward(vector<int>& rewards) {
        const int MAXN = 1e5;
        int N = rewards.size();
        sort(rewards.begin(), rewards.end());
        bitset<MAXN> dp, mask;
        dp.set(0);
        int x = 0;
        for (int v : rewards) {
            while (x < v) {
                mask.set(x++);
            }
            dp |= (dp & mask) << v;
        }
        for (int x = MAXN - 1; x >= 0; x--) {
            if (dp.test(x)) return x;
        }
        return 0;
    }
};
```

```py
class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        MAXN = int(1e5)
        x = mask = 0
        dp = 1
        for r in sorted(rewardValues):
            while x < r:
                mask |= (1 << x)
                x += 1
            dp |= (dp & mask) << r
        for i in reversed(range(MAXN)):
            if (dp >> i) & 1: return i
```