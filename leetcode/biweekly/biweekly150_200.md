# Leetcode Biweekly Rounds 150-199

# Leetcode Biweekly Contest 150

## 3453. Separate Squares I

### Solution 1:  binary search with tolerance, floating point numbers, greedy binary search

```cpp
class Solution {
private:
    vector<vector<int>> squares;
    int N;
    long double totalArea;
    bool possible(long double target) {
        long double area = 0;
        for (const vector<int>& square : squares) {
            int x = square[0], y = square[1], l = square[2];
            long double top = y + l;
            if (top <= target) {
                area += static_cast<long double>(l) * l;
            } else if (static_cast<long double>(y) <= target) {
                long double dy = target - y;
                area += dy * l;
            }
        }
        return area < (totalArea / 2.0);
    }
public:
    double separateSquares(vector<vector<int>>& sq) {
        squares = sq;
        N = squares.size();
        totalArea = 0;
        for (const vector<int>& square : squares) {
            int x = square[0], y = square[1], l = square[2];
            totalArea += static_cast<long double>(l) * l;
        }
        long double lo = 0, hi = 1e10;
        while (hi - lo > 1e-7) {
            long double mid = lo + (hi - lo) / 2.0;
            if (possible(mid)) lo = mid;
            else hi = mid;
        }
        return lo;
    }
};
```

## 3454. Separate Squares II

### Solution 1: line sweep, segment tree for union of rectangles, area covered by squares on 2d plane, segment tree with coordinate compression

1. segment tree calculates the width of each region covered by squares as you use line sweep over the y-value events. 
1. median y-coordinate, can find this with a little math, it is like interpolation

```cpp
using int64 = long long;
const int64 INF = 1e9;

struct SegmentTree {
    int N;
    vector<int64> count, total;
    vector<int64> xs;
    SegmentTree(vector<int64>& arr) {
        xs = vector<int64>(arr.begin(), arr.end());
        sort(xs.begin(), xs.end());
        xs.erase(unique(xs.begin(), xs.end()), xs.end());
        N = xs.size();
        count.assign(4 * N + 1, 0);
        total.assign(4 * N + 1, 0);
    }
    void update(int segmentIdx, int segmentLeftBound, int segmentRightBound, int64 l, int64 r, int64 val) {
        if (l >= r) return;
        if (l == xs[segmentLeftBound] && r == xs[segmentRightBound]) {
            count[segmentIdx] += val;
        } else {
            int mid = (segmentLeftBound + segmentRightBound) / 2;

            if (l < xs[mid]) {
                update(2 * segmentIdx, segmentLeftBound, mid, l, min(r, xs[mid]), val);
            }
            if (r > xs[mid]) {
                update(2 * segmentIdx + 1, mid, segmentRightBound, max(l, xs[mid]), r, val);
            }
        }
        if (count[segmentIdx] > 0) {
            total[segmentIdx] = xs[segmentRightBound] - xs[segmentLeftBound];
        } else {
            total[segmentIdx] = 2 * segmentIdx + 1 < total.size() ? total[2 * segmentIdx] + total[2 * segmentIdx + 1] : 0;
        }
    }
    void update(int l, int r, int val) {
        update(1, 0, N - 1, l, r, val);
    }
    int64 query() {
        return total[1];
    }
};

struct Event {
    int v, t, l, r;
    Event() {}
    Event(int v, int t, int l, int r) : v(v), t(t), l(l), r(r) {}
    bool operator<(const Event& other) const {
        if (v != other.v) return v < other.v;
        return t < other.t;
    }
};

class Solution {
public:
    double separateSquares(vector<vector<int>>& squares) {
        int N = squares.size();
        vector<int64> xs;
        for (const vector<int>& square : squares) {
            int x = square[0], y = square[1], l = square[2];
            xs.emplace_back(x);
            xs.emplace_back(x + l);
        }
        SegmentTree st(xs);
        vector<Event> events;
        for (const vector<int>& square : squares) {
            int x = square[0], y = square[1], l = square[2];
            events.emplace_back(y, 1, x, x + l);
            events.emplace_back(y + l, -1, x, x + l);
        }
        sort(events.begin(), events.end());
        int64 prevY = -INF;
        long double totalArea = 0;
        for (const Event& event : events) {
            long double dh = event.v - prevY;
            long double dw = st.query();
            totalArea += dh * dw;
            st.update(event.l, event.r, event.t);
            prevY = event.v;
        }
        prevY = -INF;
        long double curSumArea = 0;
        for (const Event& event : events) {
            long double dh = event.v - prevY;
            long double dw = st.query();
            long double area = dh * dw;
            if (2 * (area + curSumArea) >= totalArea) {
                return (totalArea / 2.0 - curSumArea) / dw + prevY;
            }
            curSumArea += area;
            st.update(event.l, event.r, event.t);
            prevY = event.v;
        }
        return curSumArea;
    }
};
```

## 3455. Shortest Matching Substring

### Solution 1:  kmp, string matching, dynamic programming, 

```cpp
const int INF = 1e9;
class Solution {
private:
    vector<int> kmp(const string& s) {
        int N = s.size();
        vector<int> pi(N, 0);
        for (int i = 1; i < N; i++) {
            int j = pi[i - 1];
            while (j > 0 && s[i] != s[j]) {
                j = pi[j - 1];
            }
            if (s[j] == s[i]) j++;
            pi[i] = j;
        }
        return pi;
    }
    vector<string> process(const string& s, char delimiter = ' ') {
        vector<string> ans;
        istringstream iss(s);
        string word;
        while (getline(iss, word, delimiter)) ans.emplace_back(word);
        return ans;
    }
public:
    int shortestMatchingSubstring(string s, string p) {
        int N = s.size(), M = p.size();
        vector<string> patterns = process(p, '*');
        while (patterns.size() < 3) {
            patterns.emplace_back("");
        }
        vector<vector<int>> pi(3);
        for (int i = 0; i < 3; i++) {
            vector<int> ret = kmp(patterns[i] + "#" + s);
            pi[i] = vector<int>(ret.begin() + patterns[i].size(), ret.end());
        }
        vector<vector<int>> dp(4, vector<int>(N + 1, -INF));
        iota(dp[0].begin(), dp[0].end(), 0);
        for (int i = 0; i < 3; i++) {
            for (int j = 1; j <= N; j++) {
                if (pi[i][j] == patterns[i].size()) {
                    dp[i + 1][j] = dp[i][j - pi[i][j]];
                }
                dp[i + 1][j] = max(dp[i + 1][j], dp[i + 1][j - 1]);
            }
        }
        int ans = INF;
        for (int i = 0; i <= N; i++) {
            if (dp[3][i] != -INF) {
                ans = min(ans, i - dp[3][i]);
            }
        }
        
        return ans < INF ? ans : -1;
    }
};
```

# Leetcode Biweekly Contest 151

## 3468. Find the Number of Copy Arrays

### Solution 1: interval, greedy, lower and upper bounds

```py
class Solution:
    def countArrays(self, original: List[int], bounds: List[List[int]]) -> int:
        N = len(original)
        lower, upper = bounds[0]
        delta = 0
        for i in range(1, N):
            l, r = bounds[i]
            delta += original[i] - original[i - 1]
            lower += max(0, l - lower - delta)
            upper -= max(0, upper + delta - r)
        return max(0, upper - lower + 1)
```

## 3469. Find Minimum Cost to Remove Array Elements

### Solution 1: prefix, dynamic programming

1. The trick is that there can only be one element that remains in array that is before the current prefix. 
1. map out the possiblities that is enough to solve this one. 

```py
class Solution:
    def minCost(self, nums: List[int]) -> int:
        N = len(nums)
        dp = [[math.inf] * N for _ in range(N)]
        def dfs(last, idx):
            if idx >= N:
                return nums[last] if last < N else 0
            if dp[last][idx] != math.inf: return dp[last][idx]
            ans = math.inf
            ans = min(ans, dfs(idx + 1, idx + 2) + max(nums[last], nums[idx]))
            if idx + 1 < N:
                ans = min(ans, dfs(idx, idx + 2) + max(nums[last], nums[idx + 1]))
                ans = min(ans, dfs(last, idx + 2) + max(nums[idx], nums[idx + 1]))
            dp[last][idx] = ans
            return dp[last][idx]
        return dfs(0, 1)
```

## 3470. Permutations IV

### Solution 1: recursion, factorial, counting, combinatorics, parity

```py
def ceil(x, y):
    return (x + y - 1) // y
def floor(x, y):
    return x // y
class Solution:
    def permute(self, n: int, k: int) -> List[int]:
        ans = []
        odd = [False for _ in range(ceil(n, 2))]
        even = [False for _ in range(floor(n, 2))]
        def calc(N, rem, parity):
            if not N: return
            cnt = 0
            if not parity:
                for i in range(len(even)):
                    if even[i]: continue
                    nxt = math.factorial(floor(N - 1, 2)) * math.factorial(ceil(N - 1, 2))
                    if rem <= cnt + nxt:
                        ans.append(2 * i + 2)
                        even[i] = True
                        calc(N - 1, rem - cnt, parity ^ 1)
                        return
                    cnt += nxt
            else:
                for i in range(len(odd)):
                    if odd[i]: continue
                    nxt = math.factorial(floor(N - 1, 2)) * math.factorial(ceil(N - 1, 2))
                    if rem <= cnt + nxt:
                        ans.append(2 * i + 1)
                        odd[i] = True
                        calc(N - 1, rem - cnt, parity ^ 1)
                        return
                    cnt += nxt
        cnt = 0
        if n % 2 == 0:
            for i in range(n):
                nxt = math.factorial(floor(n - 1, 2)) * math.factorial(ceil(n - 1, 2))
                if k <= cnt + nxt:
                    ans.append(i + 1)
                    if i % 2 == 0: odd[i // 2] = True
                    else: even[i // 2] = True
                    calc(n - 1, k - cnt, i % 2)
                    break
                cnt += nxt
        else:
            for i in range(len(odd)):
                nxt = math.factorial(floor(n - 1, 2)) * math.factorial(ceil(n - 1, 2))
                if k <= cnt + nxt:
                    ans.append(2 * i + 1)
                    odd[i] = True
                    calc(n - 1, k - cnt, 0)
                    break
                cnt += nxt
        return ans if len(ans) == n else []©leetcode
```

# Leetcode Biweekly Contest 152

## 

### Solution 1: 

```cpp

```

## 

### Solution 1: 

```cpp

```

## 

### Solution 1: 

```cpp

```

# Leetcode Biweekly Contest 153

## 3500. Minimum Cost to Divide Array Into Subarrays

### Solution 1: dynamic programming, mathematics, prefix sums, range queries, 

```cpp
using int64 = long long;
const int64 INF = (1LL << 63) - 1;
class Solution {
private:
    int N, K;
    vector<int64> A, C, prefA, prefC;
    vector<vector<int64>> dp;
    int64 query(const vector<int64>& psum, int l, int r) {
        if (l > r) return 0;
        int64 res = psum[r];
        if (l > 0) res -= psum[l - 1];
        return res;
    }
    int64 dfs(int l, int r) {
        if (r == N) {
            if (l == N) return 0;
            return INF;
        }
        if (dp[l][r] != -1) return dp[l][r];
        int64 costSubarray = (query(prefA, l, r) + K) * query(prefC, l, N - 1);
        int64 nextSubarrayCost = dfs(r + 1, r + 1);
        int64 cutSubarrayCost = INF;
        if (nextSubarrayCost != INF) cutSubarrayCost = costSubarray + nextSubarrayCost;
        int64 extendSubarrayCost = dfs(l, r + 1);
        return dp[l][r] = min(extendSubarrayCost, cutSubarrayCost);
    }
public:
    int64 minimumCost(vector<int>& nums, vector<int>& cost, int k) {
        N = nums.size();
        K = k;
        A = vector<int64>(nums.begin(), nums.end());
        C = vector<int64>(cost.begin(), cost.end());
        prefA.assign(N, 0);
        prefC.assign(N, 0);
        for (int i = 0; i < N; i++) {
            prefA[i] = A[i];
            prefC[i] = C[i];
            if (i > 0) {
                prefA[i] += prefA[i - 1];
                prefC[i] += prefC[i - 1];
            }
        }
        dp.assign(N + 1, vector<int64>(N + 1, -1));
        return dfs(0, 0);
    }
};
```

## 3501. Maximize Active Section with Trade II

### Solution 1: sparse table, range maximum query, consecutive blocks, remap index to zero blocks

```cpp
class Solution {
private:
    const int LOG = 24;
    vector<int> nums;
    vector<vector<int>> st;

    int query(int L, int R) {
        int k = log2(R - L + 1);
        return max(st[k][L], st[k][R - (1 << k) + 1]);
    }
public:
    vector<int> maxActiveSectionsAfterTrade(string s, vector<vector<int>>& queries) {
        int N = s.size(), M = queries.size(), baseActive = 0;
        vector<pair<int, int>> zblocks;
        vector<int> indexMap(N, 0);
        for (int i = 0; i < N; i++) {
            if (s[i] == '0') {
                if (i > 0 && s[i - 1] == '0') zblocks.back().second++;
                else zblocks.emplace_back(i, 1);
            } else {
                baseActive++;
            }
            indexMap[i] = zblocks.size() - 1;
        }
        int NZ = zblocks.size();
        st.assign(LOG, vector<int>(N, 0));
        for (int i = 0; i + 1 < NZ; i++) {
            int gain = zblocks[i].second + zblocks[i + 1].second;
            st[0][i] = gain;
        }
        for (int i = 1; i < LOG; i++) {
            for (int j = 0; j + (1 << (i - 1)) < N; j++) {
                st[i][j] = max(st[i - 1][j], st[i - 1][j + (1 << (i - 1))]);
            }
        }
        vector<int> ans(M, 0);
        for (int i = 0; i < M; i++) {
            int l = queries[i][0], r = queries[i][1];
            int zl = indexMap[l] + 1, zr = indexMap[r] - (s[r] == '0');
            int cntLeft = indexMap[l] >= 0 ? zblocks[indexMap[l]].second - (l - zblocks[indexMap[l]].first): -1;
            int cntRight = indexMap[r] >= 0 ? r - zblocks[indexMap[r]].first + 1: -1;
            if (zr > zl) {
                ans[i] = max(ans[i], query(zl, zr - 1));
            }
            if (s[l] == '0' && s[r] == '0' && indexMap[l] + 1 == indexMap[r]) {
                ans[i] = max(ans[i], cntLeft + cntRight);
            }
            if (s[l] == '0' && indexMap[r] - indexMap[l] + (s[r] == '1') > 1) {
                ans[i] = max(ans[i], cntLeft + zblocks[indexMap[l] + 1].second);
            }
            if (s[r] == '0' && indexMap[r] - indexMap[l] > 1) {
                ans[i] = max(ans[i], cntRight + zblocks[indexMap[r] - 1].second);
            }
            ans[i] += baseActive;
        }
        return ans;
    }
};
```

# Leetcode Biweekly Contest 154

## Number of Unique XOR Triplets I

### Solution 1: powers of 2, binary search

```cpp
class Solution {
public:
    int uniqueXorTriplets(vector<int>& nums) {
        int N = nums.size();
        vector<int> powers(20, 1);
        for (int i = 1; i < 20; i++) {
            powers[i] = powers[i - 1] * 2;
        }
        auto it = upper_bound(powers.begin(), powers.end(), N);
        if (N < 3) it--;
        int ans = *it;
        return ans;
    }
};
```

## Number of Unique XOR Triplets II

### Solution 1: set, preocompute the first pair, then add the third

1. Because the number of possible unique xors of just a pair of numbers is relatively small this gets the time complexity to roughly O(N^2)

```cpp
class Solution {
public:
    int uniqueXorTriplets(vector<int>& nums) {
        int N = nums.size();
        sort(nums.begin(), nums.end());
        nums.erase(unique(nums.begin(), nums.end()), nums.end());
        unordered_set<int> doubles, triplets;
        for (int i = 0; i < N; i++) {
            for (int j = i; j < N; j++) {
                doubles.insert(nums[i] ^ nums[j]);
            }
        }
        for (int x : nums) {
            for (int v : doubles) {
                triplets.insert(x ^ v);
            }
        }
        int ans = triplets.size();
        return ans;
    }
};
```

## Shortest Path in a Weighted Tree

### Solution 1: fenwick tree, euler tour for path queries, tree

```cpp
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
private:
    int N, timer;
    vector<int> start, end_, values;
    vector<vector<pair<int, int>>> adj;
    void dfs(int u, int p = -1) {
        for (auto &[v, w] : adj[u]) {
            if (v == p) continue;
            values[v] = w;
            dfs(v, u);
        }
    }
    void dfs1(int u, int p = -1) {
        start[u] = ++timer;
        for (auto &[v, w] : adj[u]) {
            if (v == p) continue;
            dfs1(v, u);
        }
        end_[u] = ++timer;
    }
public:
    vector<int> treeQueries(int n, vector<vector<int>>& edges, vector<vector<int>>& queries) {
        N = n;
        adj.assign(N, vector<pair<int, int>>());
        values.assign(N, 0);
        start.resize(N);
        end_.resize(N);
        for (const auto &edge : edges) {
            int u = edge[0], v = edge[1], w = edge[2];
            u--, v--;
            adj[u].emplace_back(v, w);
            adj[v].emplace_back(u, w);
        }
        dfs(0);
        dfs1(0);
        FenwickTree<int> ft;
        ft.init(timer);
        for (int i = 0; i < N; i++) {
            ft.update(start[i], values[i]);
            ft.update(end_[i], -values[i]);
        }

        vector<int> ans;
        for (const auto &query : queries) {
            int t = query[0];
            if (t == 1) {
                int u = query[1], v = query[2], nw = query[3];
                u--, v--;
                if (start[v] > start[u]) swap(u, v);
                int delta = nw - values[u];
                ft.update(start[u], delta);
                ft.update(end_[u], -delta);
                values[u] = nw;
            } else {
                int u = query[1];
                u--;
                ans.emplace_back(ft.query(start[u]));
            }
        }
        return ans;
    }
};
```

# Leetcode Biweekly Contest 155

## 

### Solution 1: 

```cpp

```

## 

### Solution 1: 

```cpp

```

## 

### Solution 1: 

```cpp

```