# Leetcode Weekly Contest 406

## Minimum Cost for Cutting Cake II

### Solution 1:  greedy, sorting

```cpp
class Solution {
public:
    long long minimumCost(int m, int n, vector<int>& horizontalCut, vector<int>& verticalCut) {
        long long vcount = 1, hcount = 1, ans = 0;
        vector<pair<long long, int>> queries;
        for (int x : horizontalCut) {
            queries.emplace_back(x, 0);
        }
        for (int x : verticalCut) {
            queries.emplace_back(x, 1);
        }
        sort(queries.begin(), queries.end());
        reverse(queries.begin(), queries.end());
        for (auto [cost, t] : queries) {
            if (t == 0) {
                ans += vcount * cost;
                hcount++;
            } else {    
                ans += hcount * cost;
                vcount++;
            }
        }
        return ans;
    }
};
```