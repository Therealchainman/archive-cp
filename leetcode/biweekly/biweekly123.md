# Leetcode BiWeekly  Contest 123

## Maximum Good Subarray Sum

### Solution 1:  prefix sum, kadane's algorithm, last occurrence

```py
import math
class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        psum = list(accumulate(nums))
        vsum = Counter()
        last = {}
        ans = -math.inf
        for i in range(n):
            if nums[i] in last:
                l, r = last[nums[i]], i - 1
                vsum[nums[i]] += psum[r]
                if l > 0:
                    vsum[nums[i]] -= psum[l - 1]
                vsum[nums[i]] = max(0, vsum[nums[i]])
            for cand in [nums[i] - k, nums[i] + k]:
                if cand in last:
                    l, r = last[cand], i
                    segsum = vsum[cand] + psum[r]
                    if l > 0: segsum -= psum[l - 1]
                    ans = max(ans, segsum)
            last[nums[i]] = i
        return ans if ans > -math.inf else 0
```

## Find the Number of Ways to Place People II

### Solution 1:  sort, fenwick tree

```cpp
long long neutral = 0;
struct FenwickTree {
    vector<long long> nodes;
    
    void init(int n) {
        nodes.assign(n + 1, neutral);
    }

    void update(int idx, long long val) {
        while (idx < (int)nodes.size()) {
            nodes[idx] += val;
            idx += (idx & -idx);
        }
    }

    int query(int left, int right) {
        return query(right) - query(left - 1);
    }

    long long query(int idx) {
        long long result = neutral;
        while (idx > 0) {
            result += nodes[idx];
            idx -= (idx & -idx);
        }
        return result;
    }
};
class Solution {
public:
    int numberOfPairs(vector<vector<int>>& points) {
        int n = points.size();
        sort(points.begin(), points.end(), [](const vector<int>& a, const vector<int>& b) {
                  if (a[0] == b[0]) // If the x-coordinates are equal
                      return a[1] > b[1]; // Sort by y-coordinate in descending order
                  return a[0] < b[0]; // Otherwise, sort by x-coordinate in ascending order
              });
        int ans = 0;
        unordered_map<int, int> y_coords;
        int idx = 1;
        vector<int> y_values;
        for (int i = 0; i < n; i++) {
            y_values.push_back(points[i][1]);
        }
        sort(y_values.begin(), y_values.end());
        for (int y : y_values) {
            if (y_coords.find(y) != y_coords.end()) continue;
            y_coords[y] = idx++;
        }
        FenwickTree fenwick;
        fenwick.init(idx);
        vector<int> between;
        for (int i = 0; i < n; i++) {
            int x1 = points[i][0], y1 = points[i][1];
            between.clear();
            for (int j = i + 1; j < n; j++) {
                int x2 = points[j][0], y2 = points[j][1];
                if (x1 <= x2 && y1 >= y2) {
                    int l = y_coords[y2], r = y_coords[y1];
                    if (fenwick.query(l, r) == 0) ans++;
                    between.push_back(y2);
                    fenwick.update(y_coords[y2], 1);
                }
            }
            for (int y : between) {
                fenwick.update(y_coords[y], -1);
            }
        }
        return ans;
    }
};
```

```py
class FenwickTree:
    def __init__(self, N):
        self.sums = [0 for _ in range(N+1)]

    def update(self, i, delta):
        while i < len(self.sums):
            self.sums[i] += delta
            i += i & (-i)

    def query(self, i):
        res = 0
        while i > 0:
            res += self.sums[i]
            i -= i & (-i)
        return res

    def query_range(self, i, j):
        return self.query(j) - self.query(i - 1)

    def __repr__(self):
        return f"array: {self.sums}"
    
class Solution:
    def numberOfPairs(self, points: List[List[int]]) -> int:
        n = len(points)
        # x1 <= x2 and y1 >= y2
        points.sort(key = lambda point: (point[0], -point[1]))
        ans = 0
        # get index for y value
        y_coordinates = {}
        for _, y in sorted(points, key = lambda p: p[1]):
            if y in y_coordinates: continue
            y_coordinates[y] = len(y_coordinates) + 1
        m = len(y_coordinates)
        fenwick = FenwickTree(m)
        # x increasing, y decreasing
        for i in range(n):
            x1, y1 = points[i]
            between = []
            for j in range(i + 1, n):
                x2, y2 = points[j]
                if x1 <= x2 and y1 >= y2:
                    r, l = y_coordinates[y1], y_coordinates[y2]
                    if fenwick.query_range(l, r) == 0: 
                        ans += 1
                    between.append(y2)
                    y_index = y_coordinates[y2]
                    fenwick.update(y_index, 1)
            for y in between:
                y_index = y_coordinates[y]
                fenwick.update(y_index, -1)
        return ans
```

### Solution 2:  sort, binary search, sortedlist data structure

```py
from sortedcontainers import SortedList 
class Solution:
    def numberOfPairs(self, points: List[List[int]]) -> int:
        n = len(points)
        # x1 <= x2 and y1 >= y2
        points.sort(key = lambda point: (point[0], -point[1]))
        ans = 0
        # x increasing, y decreasing
        for i in range(n):
            x1, y1 = points[i]
            between = SortedList()
            for j in range(i + 1, n):
                x2, y2 = points[j]
                if x1 <= x2 and y1 >= y2:
                    idx = between.bisect_left(y2)
                    if idx == len(between): ans += 1
                    between.add(y2)
        return ans
```

### Solution 3:  sort, binary search, set

```cpp
class Solution {
public:
    int numberOfPairs(vector<vector<int>>& points) {
        int n = points.size();
        sort(points.begin(), points.end(), [](const vector<int>& a, const vector<int>& b) {
            if (a[0] == b[0]) return a[1] > b[1]; // sort y coordinate in descending order
            return a[0] < b[0]; // sort x coordinate in ascending order
        });
        int ans = 0;
        set<int> between;
        for (int i = 0; i < n; i++) {
            int x1 = points[i][0], y1 = points[i][1];
            between.clear();
            for (int j = i + 1; j < n; j++) {
                int x2 = points[j][0], y2 = points[j][1];
                if (x1 <= x2 && y1 >= y2) {
                    auto it = between.lower_bound(y2);
                    if (it == between.end()) ans++;
                    between.insert(y2);
                }
            }
        }
        return ans;
    }
};
```

### Solution 4:  sort, track max y value between two pairs, O(n^2)

```py
class Solution:
    def numberOfPairs(self, points: List[List[int]]) -> int:
        n = len(points)
        # x1 <= x2 and y1 >= y2
        points.sort(key = lambda point: (point[0], -point[1]))
        ans = 0
        # x weakly increasing, y weakly decreasing
        for i in range(n):
            x1, y1 = points[i]
            max_y = -math.inf
            for j in range(i + 1, n):
                x2, y2 = points[j]
                if x1 <= x2 and y1 >= y2: # max_y <= y1
                    if max_y < y2: ans += 1
                    max_y = max(max_y, y2)
        return ans
```

