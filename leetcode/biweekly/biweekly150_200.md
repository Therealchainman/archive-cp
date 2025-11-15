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

## 3485. Longest Common Prefix of K Strings After Removal

### Solution 1: trie with multiset, trie with erase

trie with multiset,
store frequency of each prefix

that is the count represents how many words have share that prefix, so we know the length just based on the index.  So if that count is >= k then add it to a multiset, so a specific prefix if it has count = k + 3, it will be added to multiset 4 times, so you'd have to remove 4 words to remove it you know.  

```cpp
int K;
multiset<int, greater<int>> lengths; // max multiset, descending order
struct Node {
    int children[26];
    int cnt;
    void init() {
        memset(children, 0, sizeof(children));
        cnt = 0;
    }
};
struct Trie {
    vector<Node> trie;
    void init() {
        Node root;
        root.init();
        trie.emplace_back(root);
    }
    void insert(const string& s) {
        int cur = 0;
        for (int i = 0; i < s.size(); ++i) {
            int cv = s[i] - 'a';
            if (trie[cur].children[cv]==0) {
                Node root;
                root.init();
                trie[cur].children[cv] = trie.size();
                trie.emplace_back(root);
            }
            cur = trie[cur].children[cv];
            trie[cur].cnt++;
            if (trie[cur].cnt >= K) {
                lengths.insert(i + 1);
            }
        }
    }
    void erase(const string& s) {
        int cur = 0;
        for (int i = 0; i < s.size(); ++i) {
            int cv = s[i] - 'a';
            cur = trie[cur].children[cv];
            trie[cur].cnt--;
            if (trie[cur].cnt == K - 1) {
                auto it = lengths.find(i + 1);
                lengths.erase(it);
            }
        }
    }
};
class Solution {
public:
    vector<int> longestCommonPrefix(vector<string>& words, int k) {
        int N = words.size();
        K = k;
        Trie trie;
        trie.init();
        lengths.clear();
        for (const string& s : words) {
            trie.insert(s);
        }
        vector<int> ans(N, 0);
        for (int i = 0; i < N; ++i) {
            trie.erase(words[i]);
            auto it = lengths.begin();
            if (it != lengths.end()) {
                ans[i] = *it;
            }
            trie.insert(words[i]);
        }
        return ans;
    }
};
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

## 3528. Unit Conversion I

### Solution 1: DAG, tree with directed graph, topological order, dfs

```cpp
using int64 = long long;
const int64 MOD = 1e9 + 7;
class Solution {
private:
    vector<int> ans;
    vector<vector<pair<int, int64>>> adj;
    void dfs(int u) {
        for (auto &[v, w]: adj[u]) {
            ans[v] = ans[u] * w % MOD;
            dfs(v);
        }
    }
public:
    vector<int> baseUnitConversions(vector<vector<int>>& conversions) {
        int N = 0;
        for (const vector<int> &edge : conversions) {
            N = max({N, edge[0] + 1, edge[1] + 1});
        }
        adj.assign(N, vector<pair<int, int64>>());
        for (const vector<int> &edge : conversions) {
            int u = edge[0], v = edge[1], w = edge[2];
            adj[u].emplace_back(v, w);
        }
        ans.assign(N, 1);
        dfs(0);
        return ans;
    }
};
```

## 3529. Count Cells in Overlapping Horizontal and Vertical Substrings

### Solution 1: z alborithm, transpose, grid, string matching

```cpp
class Solution {
private:
    int R, C, N;
    std::vector<int> z_algorithm(const string& s) {
        int n = s.length();
        vector<int> z(n, 0);
        int left = 0, right = 0;
        for (int i = 1; i < n; ++i) {
            if (i > right) {
                left = right = i;
                while (right < n && s[right-left] == s[right]) {
                    right++;
                }
                z[i] = right - left;
                right--;
            } else {
                int k = i - left;
                if (z[k] < right - i + 1) {
                    z[i] = z[k];
                } else {
                    left = i;
                    while (right < n && s[right-left] == s[right]) {
                        right++;
                    }
                    z[i] = right - left;
                    right--;
                }
            }
        }
        return z;
    }
    vector<bool> calc(const vector<int> &zarr) {
        vector<bool> vis(R * C, false);
        int cnt = 0;
        for (int i = 0; i < zarr.size(); i++) {
            if (zarr[i] == N) {
                cnt = N;
            }
            if (cnt) {
                vis[i - N - 1] = true;
                cnt--;
            }
        }
        return vis;
    }
    vector<vector<char>> transpose(const vector<vector<char>>& mat) {
        vector<vector<char>> ans(C, vector<char>(R));
        for (int i = 0; i < R; i++) {
            for (int j = 0; j < C; j++) {
                ans[j][i] = mat[i][j];
            }
        }
        return ans;
    }
    pair<int, int> map1dTo2d(int idx) {
        return {idx / R, idx % R};
    }
    int map2dTo1d(int r, int c) {
        return r * C + c;
    }
public:
    int countCells(vector<vector<char>>& grid, string pattern) {
        R = grid.size(), C = grid[0].size(), N = pattern.size();
        string hs = pattern + '$';
        for (int r = 0; r < R; r++) {
            for (int c = 0; c < C; c++) {
                hs += grid[r][c];
            }
        }
        vector<int> zarr = z_algorithm(hs);
        vector<bool> vis = calc(zarr);
        string vs = pattern + '$';
        vector<vector<char>> tgrid = transpose(grid);
        for (int r = 0; r < C; r++) {
            for (int c = 0; c < R; c++) {
                vs += tgrid[r][c];
            }
        }
        vector<int> zarr2 = z_algorithm(vs);
        vector<bool> transVis = calc(zarr2);
        int ans = 0;
        for (int i = 0; i < R * C; i++) {
            auto [r, c] = map1dTo2d(i);
            int j = map2dTo1d(c, r);
            ans += transVis[i] & vis[j];
        }
        return ans;
    }
};
```

## 3530. Maximum Profit from Valid Topological Order in DAG

### Solution 1: DAG, topological ordering, bitmask dp, prerequisite bitmask, bit manipulation

1. prerequisite bitmask can determine if from some state of nodes visited, if you can visit the next node, has all prerequisites been satisfied by current mask/state.

```cpp
using int64 = long long;
const int64 INF = (1LL << 63) - 1;
class Solution {
private:
    bool isSet(int mask, int i) {
        return (mask >> i) & 1;
    }
public:
    int maxProfit(int n, vector<vector<int>>& edges, vector<int>& score) {
        vector<vector<int>> adj(n, vector<int>());
        vector<int> prereq(n, 0);
        for (const vector<int> &edge : edges) {
            int u = edge[0], v = edge[1];
            adj[u].emplace_back(v);
            prereq[v] |= (1 << u);
        }
        vector<int64> dp(1 << n, -INF);
        dp[0] = 0;
        for (int mask = 0; mask < (1 << n); mask++) {
            int64 pos = __builtin_popcount(mask) + 1;
            if (dp[mask] == -INF) continue; // unreachable state
            for (int i = 0; i < n; i++) {
                if (isSet(mask, i) || (mask & prereq[i]) != prereq[i]) continue;
                int nmask = mask | (1 << i);
                dp[nmask] = max(dp[nmask], dp[mask] + pos * score[i]);
            }
        }
        return dp[(1 << n) - 1];
    }
};
```

# Leetcode Biweekly Contest 156

## 3542. Minimum Operations to Convert All Elements to Zero

### Solution 1: monotonic stack, greedy

```cpp
class Solution {
public:
    int minOperations(vector<int>& nums) {
        stack<int> stk;
        int ans = 0;
        for (int x : nums) {
            while (!stk.empty() && stk.top() >= x) {
                int v = stk.top();
                if (v > x) ans++;
                stk.pop();
            }
            if (x) stk.emplace(x);
        }
        ans += stk.size();
        return ans;
    }
};
```

## 

### Solution 1: 

```cpp

```

## 

### Solution 1: 

```cpp

```

# Leetcode Biweekly Contest 157

## Sum of Largest Prime Substrings

### Solution 1: primality test, miller rabin, string to number, set

```py
def check_composite(n, a, d, s):
    x = pow(a, d, n)
    if x == 1 or x == n - 1: return False
    for r in range(1, s):
        x = x * x % n
        if x == n - 1: return False
    return True

def miller_rabin(n):
    if n < 4: return n == 2 or n == 3
    r = 0
    d = n - 1
    while d % 2 == 0:
        r += 1
        d >>= 1
    bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for a in bases:
        if n == a: return True
        if check_composite(n, a, d, r): return False
    return True
class Solution:
    def sumOfLargestPrimes(self, s: str) -> int:
        n = len(s)
        primes = set()
        for i in range(n):
            for j in range(i + 1, n + 1):
                val = int(s[i : j])
                if (miller_rabin(val)): primes.add(val)
        ans = 0
        primes = sorted(primes, reverse = True)
        for i in range(min(3, len(primes))):
            ans += primes[i]
        return ans
```

## Find Maximum Number of Non Intersecting Substrings

### Solution 1: dynamic programming, greedy

```cpp
class Solution {
private:
    int decode(char ch) {
        return ch - 'a';
    }
public:
    int maxSubstrings(string word) {
        int N = word.size();
        vector<int> dp(N + 1, 0);
        vector<vector<int>> last(26, vector<int>());
        for (int i = 1; i <= N; i++) {
            dp[i] = dp[i - 1];
            int u = decode(word[i - 1]);
            for (int idx = last[u].size() - 1; idx >= 0; idx--) {
                int j = last[u][idx];
                int d = i - j + 1;
                if (d >= 4) {
                    dp[i] = max(dp[i], dp[i - d] + 1);
                    break;
                }
            }
            last[u].emplace_back(i);
        } 
        return dp[N];
    }
};
```

## Number of Ways to Assign Edge Weights II

### Solution 1: binary lifting, tree, lca, distance, dynamic programming

```cpp
const int MOD = 1e9 + 7;
struct Tree {
    int N, LOG;
    vector<vector<int>> adj;
    vector<int> depth, parent, dist;
    vector<vector<int>> up;

    Tree(int n) : N(n) {
        LOG = 20;
        adj.assign(N, vector<int>());
        depth.assign(N, 0);
        parent.assign(N, -1);
        dist.assign(N, 0);
        up.assign(LOG, vector<int>(N, -1));
    }
    void addEdge(int u, int v, int w = 1) {
        adj[u].emplace_back(v);
        adj[v].emplace_back(u);
    }
    void preprocess(int root = 0) {
        dfs(root);
        buildLiftingTable();
    }
    int kthAncestor(int u, int k) const {
        for (int i = 0; i < LOG && u != -1; i++) {
            if ((k >> i) & 1) {
                u = up[i][u];
            }
        }
        return u;
    }
    int lca(int u, int v) const {
        if (depth[u] < depth[v]) swap(u, v);
        // Bring u up to the same depth as v
        u = kthAncestor(u, depth[u] - depth[v]);
        if (u == v) return u;
        // Binary lift both
        for (int i = LOG - 1; i >= 0; i--) {
            if (up[i][u] != up[i][v]) {
                u = up[i][u];
                v = up[i][v];
            }
        }
        // Now parents are equal
        return parent[u];
    }
    int distance(int u, int v) const {
        int a = lca(u, v);
        return dist[u] + dist[v] - 2 * dist[a];
    }
private:
    void dfs(int u, int p = -1) {
        parent[u] = p;
        up[0][u] = p;
        for (int v : adj[u]) {
            if (v == p) continue;
            depth[v] = depth[u] + 1;
            dist[v] = dist[u] + 1;
            dfs(v, u);
        }
    }
    void buildLiftingTable() {
        for (int i = 1; i < LOG; i++) {
            for (int j = 0; j < N; j++) {
                if (up[i - 1][j] == -1) continue;
                up[i][j] = up[i - 1][up[i - 1][j]];
            }
        }
    }
};
class Solution {
public:
    vector<int> assignEdgeWeights(vector<vector<int>>& edges, vector<vector<int>>& queries) {
        int N = edges.size() + 1, M = queries.size();
        vector<vector<int>> dp(N + 1, vector<int>(2, 0));
        dp[0][0] = 1;
        for (int i = 1; i <= N; i++) {
            int add = (dp[i - 1][0] + dp[i - 1][1]) % MOD;
            dp[i][0] = (dp[i][0] + add) % MOD;
            dp[i][1] = (dp[i][1] + add) % MOD;
        }
        Tree tree(N);
        for (const auto &edge : edges) {
            int u = edge[0], v = edge[1];
            u--; v--;
            tree.addEdge(u, v);
        }
        tree.preprocess(0);
        vector<int> ans(M, 0);
        for (int i = 0; i < M; i++) {
            int u = queries[i][0], v = queries[i][1];
            u--; v--;
            int d = tree.distance(u, v);
            ans[i] = dp[d][1];
        }
        return ans;
    }
};
```

# Leetcode Biweekly Contest 158

## Best Time to Buy and Sell Stock V

### Solution 1: dynamic programming, state transition, maximum profit

```cpp
using int64 = int64_t;
const int64 INF = numeric_limits<int64>::max();
class Solution {
public:
    int64 maximumProfit(vector<int>& prices, int k) {
        vector<vector<int64>> dp(k + 1, vector<int64>(3, -INF)), ndp(k + 1, vector<int64>(3, -INF));
        dp[0][0] = 0;
        for (int x : prices) {
            for (int i = 0; i <= k; i++) {
                ndp[i] = dp[i];
                if (i > 0 && dp[i - 1][0] != -INF) {
                    ndp[i][1] = max(ndp[i][1], dp[i - 1][0] - x);
                    ndp[i][2] = max(ndp[i][2], dp[i - 1][0] + x);
                }
                if (dp[i][1] != -INF) ndp[i][0] = max(ndp[i][0], dp[i][1] + x);
                if (dp[i][2] != -INF) ndp[i][0] = max(ndp[i][0], dp[i][2] - x);
            }
            swap(dp, ndp);
        }
        int64 ans = 0;
        for (int i = 0; i <= k; i++) {
            ans = max(ans, dp[i][0]);
        }
        return ans;
    }
};
```

## Maximize Subarray GCD Score

### Solution 1: prefix gcd, powers of 2

```cpp
using int64 = int64_t;
class Solution {
public:
    int64 maxGCDScore(vector<int>& nums, int k) {
        int N = nums.size();
        int64 ans = 0;
        vector<int64> pow2(N, 1);
        for (int i = 0; i < N; i++) {
            while (nums[i] % (2 * pow2[i]) == 0) pow2[i] *= 2;
        }
        for (int i = 0; i < N; i++) {
            int64 g = 0;
            int cnt = 0, minPower = numeric_limits<int>::max();
            for (int j = i; j < N; j++) {
                g = gcd(g, nums[j]);
                if (pow2[j] < minPower) {
                    minPower = pow2[j];
                    cnt = 0;
                }
                if (pow2[j] == minPower) cnt++;
                int64 cand = cnt <= k ? 2 * (j - i + 1) * g : (j - i + 1) * g;
                ans = max(ans, cand);
            }
        }
        return ans;
    }
};
```

### Solution 2: sparse table, range gcd query, powers of 2

```cpp
using int64 = int64_t;
const int LOG = 30;
struct SparseGCD {
    int N;
    vector<vector<int64>> st;
    SparseGCD(const vector<int> &arr) : N(arr.size()), st(LOG, vector<int64>(N, 0)) {
        for (int i = 0; i < N; i++) {
            st[0][i] = arr[i];
        }
        for (int i = 1; i < LOG; i++) {
            for (int j = 0; j + (1LL << (i - 1)) < N - 1; j++) {
                st[i][j] = gcd(st[i - 1][j], st[i - 1][j + (1LL << (i - 1))]);
            }
        }
    }
    int64 query(int l, int r) const {
        int k = log2(r - l + 1);
        return gcd(st[k][l], st[k][r - (1LL << k) + 1]);
    }
};
class Solution {
public:
    int64 maxGCDScore(vector<int>& nums, int k) {
        int N = nums.size();
        SparseGCD sparseGCD(nums);
        int64 ans = 0;
        vector<int64> pow2(N, 1);
        for (int i = 0; i < N; i++) {
            while (nums[i] % (2 * pow2[i]) == 0) pow2[i] *= 2;
        }
        for (int len = 1; len <= N; len++) {
            map<int64, int> freq; // value of power of 2 -> count
            for (int i = 0; i < N; i++) {
                freq[pow2[i]]++;
                int j = i - len + 1;
                if (j >= 0) {
                    pair<int64, int> p = *freq.begin();
                    int64 cand = sparseGCD.query(j, i);
                    if (p.second <= k) {
                        cand *= 2;
                    }
                    ans = max(ans, len * cand);
                    freq[pow2[j]]--;
                    if (!freq[pow2[j]]) {
                        freq.erase(pow2[j]);
                    }
                }
            }
        }
        return ans;
    }
};
```

## Maximum Good Subtree

### Solution 1: bitmask dp, tree-dp, dfs, post order merging of children values

Tree‐DP with bitmask‐based state merging

```cpp
const int MOD = 1e9 + 7;
class Solution {
private:
    int ans = 0;
    vector<int> vals;
    vector<map<int, int>> dp;
    vector<vector<int>> adj;
    bool isValid(int x) {
        vector<int> freq(10, 0);
        while (x > 0) {
            freq[x % 10]++;
            x /= 10;
        }
        return all_of(freq.begin(), freq.end(), [](const int x) {
            return x <= 1;
        });
    }
    int buildMask(int x) {
        int ans = 0;
        while (x > 0) {
            int v = x % 10;
            ans |= (1 << v);
            x /= 10;
        }
        return ans;
    }
    void dfs(int u, int p = -1) {
        dp[u][0] = 0;
        if (isValid(vals[u])) {
            int mask = buildMask(vals[u]);
            dp[u][mask] = vals[u];
        }
        for (int v : adj[u]) {
            if (v == p) continue;
            dfs(v, u);
            vector<pair<int, int>> values(dp[u].begin(), dp[u].end());
            for (auto [mu, vu] : values) {
                for (auto [mv, vv] : dp[v]) {
                    if ((mu & mv) != 0) continue;
                    dp[u][mu | mv] = max(dp[u][mu | mv], vu + vv);
                }
            }            
        }
        int best = 0;
        for (auto [mu, vu] : dp[u]) {
            best = max(best, vu);
        }
        ans = (ans + best) % MOD;
    }
public:
    int goodSubtreeSum(vector<int>& vals, vector<int>& par) {
        int N = vals.size();
        this -> vals = vals;
        dp.assign(N, map<int, int>());
        adj.assign(N, vector<int>());
        for (int i = 1; i < N; i++) {
            adj[i].emplace_back(par[i]);
            adj[par[i]].emplace_back(i);
        }
        dfs(0);
        return ans;
    }
};
```

# Leetcode Biweekly Contest 159

## Count Prime-Gap Balanced Subarrays

### Solution 1: sliding window, monotonic deque, detect prime integers

```cpp
class Solution {
private:
    bool isPrime(int n) {
        if (n <= 1) return false;
        if (n <= 3) return true;
        if (n % 2 == 0 || n % 3 == 0) return false;
        int limit = static_cast<int>(sqrt(n));
        for (int i = 5; i <= limit; i += 2) {
            if (n % i == 0) return false;
        }
        return true;
    }
public:
    int primeSubarray(vector<int>& nums, int k) {
        int N = nums.size();
        vector<bool> primeArr(N, false);
        for (int i = 0; i < N; i++) {
            primeArr[i] = isPrime(nums[i]);
        }
        int lastPrime = -1, lastPrimeTwo = -1, ans = 0;
        deque<int> mindq, maxdq;
        for (int i = 0, j = 0; i < N; i++) {
            if (primeArr[i]) {
                lastPrimeTwo = lastPrime;
                lastPrime = i;
                while (!mindq.empty() && nums[i] <= nums[mindq.back()]) mindq.pop_back();
                while (!maxdq.empty() && nums[i] >= nums[maxdq.back()]) maxdq.pop_back();
                mindq.emplace_back(i);
                maxdq.emplace_back(i);
            }
            while (!mindq.empty() && !maxdq.empty() && nums[maxdq.front()] - nums[mindq.front()] > k) {
                j++;
                if (mindq.front() < j) mindq.pop_front();
                if (maxdq.front() < j) maxdq.pop_front();
            }
            if (lastPrimeTwo < j) continue;
            ans += lastPrimeTwo - j + 1;
        }
        return ans;
    }
};
```

## Kth Smallest Path XOR Sum

### Solution 1: tree, small-to-large-merging, ordered set, binary search, self balanced binary search tree

```cpp
template <typename T, typename Compare = std::less<T>>
class OrderedSet {
    enum Color { RED, BLACK };
    struct Node {
        T key;
        Color color;
        size_t size;
        Node *left, *right, *parent;
        Node(const T& k)
            : key(k), color(RED), size(1), left(nullptr), right(nullptr), parent(nullptr) {}
    };

    Node* root_;
    Compare cmp_;

    // Utility: get size of node (0 if null)
    size_t node_size(Node* x) const {
        return x ? x->size : 0;
    }
    void update_size(Node* x) {
        if (x)
            x->size = 1 + node_size(x->left) + node_size(x->right);
    }

    // Left rotate around x
    void left_rotate(Node* x) {
        Node* y = x->right;
        x->right = y->left;
        if (y->left) y->left->parent = x;
        y->parent = x->parent;
        if (!x->parent)
            root_ = y;
        else if (x == x->parent->left)
            x->parent->left = y;
        else
            x->parent->right = y;
        y->left = x;
        x->parent = y;
        // update sizes
        update_size(x);
        update_size(y);
    }

    // Right rotate around x
    void right_rotate(Node* x) {
        Node* y = x->left;
        x->left = y->right;
        if (y->right) y->right->parent = x;
        y->parent = x->parent;
        if (!x->parent)
            root_ = y;
        else if (x == x->parent->right)
            x->parent->right = y;
        else
            x->parent->left = y;
        y->right = x;
        x->parent = y;
        // update sizes
        update_size(x);
        update_size(y);
    }

    // BST insert + RB fixup
    void insert_fixup(Node* z) {
        while (z->parent && z->parent->color == RED) {
            Node* gp = z->parent->parent;
            if (z->parent == gp->left) {
                Node* y = gp->right;
                if (y && y->color == RED) {
                    z->parent->color = BLACK;
                    y->color = BLACK;
                    gp->color = RED;
                    z = gp;
                } else {
                    if (z == z->parent->right) {
                        z = z->parent;
                        left_rotate(z);
                    }
                    z->parent->color = BLACK;
                    gp->color = RED;
                    right_rotate(gp);
                }
            } else {
                Node* y = gp->left;
                if (y && y->color == RED) {
                    z->parent->color = BLACK;
                    y->color = BLACK;
                    gp->color = RED;
                    z = gp;
                } else {
                    if (z == z->parent->left) {
                        z = z->parent;
                        right_rotate(z);
                    }
                    z->parent->color = BLACK;
                    gp->color = RED;
                    left_rotate(gp);
                }
            }
        }
        root_->color = BLACK;
    }

    // Transplant u -> v
    void transplant(Node* u, Node* v) {
        if (!u->parent)
            root_ = v;
        else if (u == u->parent->left)
            u->parent->left = v;
        else
            u->parent->right = v;
        if (v)
            v->parent = u->parent;
    }

    Node* minimum(Node* x) {
        while (x->left) x = x->left;
        return x;
    }

    Node* find_node(const T& key) const {
        Node* x = root_;
        while (x) {
            if (cmp_(key, x->key))
                x = x->left;
            else if (cmp_(x->key, key))
                x = x->right;
            else
                return x;
        }
        return nullptr;
    }

    // Delete fixup
    void delete_fixup(Node* x, Node* x_parent) {
        while ((x != root_) && (!x || x->color == BLACK)) {
            if (x_parent && x == x_parent->left) {
                Node* w = x_parent->right;
                if (w && w->color == RED) {
                    w->color = BLACK;
                    x_parent->color = RED;
                    left_rotate(x_parent);
                    w = x_parent->right;
                }
                if ((!(w->left) || w->left->color == BLACK) &&
                    (!(w->right) || w->right->color == BLACK)) {
                    if(w) w->color = RED;
                    x = x_parent;
                    x_parent = x_parent->parent;
                } else {
                    if (!(w->right) || w->right->color == BLACK) {
                        if(w->left) w->left->color = BLACK;
                        w->color = RED;
                        right_rotate(w);
                        w = x_parent->right;
                    }
                    if(w) w->color = x_parent->color;
                    x_parent->color = BLACK;
                    if(w->right) w->right->color = BLACK;
                    left_rotate(x_parent);
                    x = root_;
                    break;
                }
            } else if(x_parent) {
                Node* w = x_parent->left;
                if (w && w->color == RED) {
                    w->color = BLACK;
                    x_parent->color = RED;
                    right_rotate(x_parent);
                    w = x_parent->left;
                }
                if ((!(w->left) || w->left->color == BLACK) &&
                    (!(w->right) || w->right->color == BLACK)) {
                    if(w) w->color = RED;
                    x = x_parent;
                    x_parent = x_parent->parent;
                } else {
                    if (!(w->left) || w->left->color == BLACK) {
                        if(w->right) w->right->color = BLACK;
                        w->color = RED;
                        left_rotate(w);
                        w = x_parent->left;
                    }
                    if(w) w->color = x_parent->color;
                    x_parent->color = BLACK;
                    if(w->left) w->left->color = BLACK;
                    right_rotate(x_parent);
                    x = root_;
                    break;
                }
            } else break;
        }
        if (x) x->color = BLACK;
    }

public:
    OrderedSet() : root_(nullptr) {}
    ~OrderedSet() {
        // TODO: recursively delete nodes to avoid memory leak
    }

    size_t size() const {
        return node_size(root_);
    }

    void insert(const T& key) {
        Node* z = new Node(key);
        Node* y = nullptr;
        Node* x = root_;
        while (x) {
            y = x;
            if (cmp_(z->key, x->key))
                x = x->left;
            else if (cmp_(x->key, z->key))
                x = x->right;
            else { // key already exists
                delete z;
                return;
            }
        }
        z->parent = y;
        if (!y)
            root_ = z;
        else if (cmp_(z->key, y->key))
            y->left = z;
        else
            y->right = z;
        // Update sizes up the chain
        Node* p = z;
        while (p) {
            update_size(p);
            p = p->parent;
        }
        insert_fixup(z);
    }

    void erase(const T& key) {
        Node* z = find_node(key);
        if (!z) return;
        Color y_orig = z->color;
        Node* x = nullptr;
        Node* x_parent = nullptr;
        if (!z->left) {
            x = z->right;
            x_parent = z->parent;
            transplant(z, z->right);
        } else if (!z->right) {
            x = z->left;
            x_parent = z->parent;
            transplant(z, z->left);
        } else {
            Node* y = minimum(z->right);
            y_orig = y->color;
            x = y->right;
            if (y->parent == z) {
                x_parent = y;
            } else {
                transplant(y, y->right);
                y->right = z->right;
                y->right->parent = y;
                x_parent = y->parent;
            }
            transplant(z, y);
            y->left = z->left;
            y->left->parent = y;
            y->color = z->color;
            update_size(y);
        }
        // Update sizes up from x_parent
        for (Node* p = x_parent; p; p = p->parent)
            update_size(p);
        if (y_orig == BLACK)
            delete_fixup(x, x_parent);
        delete z;
    }

    // Number of elements strictly less than key
    size_t order_of_key(const T& key) const {
        size_t cnt = 0;
        Node* x = root_;
        while (x) {
            if (cmp_(key, x->key)) {
                x = x->left;
            } else {
                cnt += node_size(x->left) + (cmp_(x->key, key) ? 1 : 0);
                x = (cmp_(x->key, key) ? x->right : nullptr);
            }
        }
        return cnt;
    }

    // 0-based k-th smallest, nullptr if out of range
    const T* find_by_order(size_t k) const {
        Node* x = root_;
        while (x) {
            size_t ls = node_size(x->left);
            if (k < ls)
                x = x->left;
            else if (k == ls)
                return &x->key;
            else {
                k -= ls + 1;
                x = x->right;
            }
        }
        return nullptr;
    }

    class iterator {
        Node* node_;
    public:
        using value_type = T;
        using reference = T&;
        using pointer = T*;
        using iterator_category = forward_iterator_tag;
        using difference_type = ptrdiff_t;

        explicit iterator(Node* n): node_(n) {}
        iterator& operator++() {
            if (!node_) return *this;
            if (node_->right) {
                node_ = node_->right;
                while (node_->left) node_ = node_->left;
            } else {
                Node* p = node_->parent;
                while (p && node_ == p->right) {
                    node_ = p;
                    p = p->parent;
                }
                node_ = p;
            }
            return *this;
        }
        reference operator*() const { return node_->key; }
        pointer operator->() const { return &node_->key; }
        bool operator==(const iterator& o) const { return node_ == o.node_; }
        bool operator!=(const iterator& o) const { return node_ != o.node_; }
    };

    iterator begin() const {
        Node* n = root_;
        while (n && n->left) n = n->left;
        return iterator(n);
    }
    iterator end() const { return iterator(nullptr); }
};

class Solution {
private:
    vector<int> nodeValues, ans;
    vector<vector<int>> adj;
    vector<vector<pair<int, int>>> data;
    OrderedSet<int> dfs(int u, int p = -1, int val = 0) {
        OrderedSet<int> curSet;
        val ^= nodeValues[u];
        curSet.insert(val);
        for (int v : adj[u]) {
            if (v == p) continue;
            OrderedSet<int> childSet = dfs(v, u, val);
            if (childSet.size() > curSet.size()) swap(childSet, curSet);
            for (int val : childSet) {
                curSet.insert(val);
            }
        }
        for (auto [i, k] : data[u]) {
            if (curSet.size() < k) continue;
            ans[i] = *curSet.find_by_order(k - 1);
        }
        return curSet;
    }
public:
    vector<int> kthSmallest(vector<int>& par, vector<int>& vals, vector<vector<int>>& queries) {
        int N = vals.size(), M = queries.size();
        nodeValues = vector<int>(vals.begin(), vals.end());
        adj.assign(N, vector<int>());
        data.assign(N, vector<pair<int, int>>());
        for (int i = 1; i < N; i++) {
            adj[par[i]].emplace_back(i);
            adj[i].emplace_back(par[i]);
        }
        for (int i = 0; i < M; i++) {
            data[queries[i][0]].emplace_back(i, queries[i][1]);
        }
        ans.assign(M, -1);
        dfs(0);
        return ans;
    }
};
```

# Leetcode Biweekly Contest 160

## Minimum Cost Path with Alternating Directions II

### Solution 1: grid, dynamic programming

```cpp
using int64 = int64_t;
const int64 INF = numeric_limits<int64>::max();
class Solution {
public:
    int64 minCost(int R, int C, vector<vector<int>>& waitCost) {
        vector<vector<int64>> dp(R, vector<int64>(C, INF));
        dp[0][0] = 1;
        for (int r = 0; r < R; r++) {
            for (int c = 0; c < C; c++) {
                int64 cost = (r + 1) * (c + 1) + waitCost[r][c];
                if (r > 0) dp[r][c] = dp[r - 1][c] + cost;
                if (c > 0) dp[r][c] = min(dp[r][c], dp[r][c - 1] + cost);
            }
        }
        return dp[R - 1][C - 1] - waitCost[R - 1][C - 1];
    }
};
```

## Minimum Time to Reach Destination in Directed Graph

### Solution 1: directed graph, dijkstra's algorithm, minheap

```cpp
const int INF = numeric_limits<int>::max();
class Solution {
public:
    int minTime(int n, vector<vector<int>>& edges) {
        vector<vector<int>> adj(n, vector<int>());
        for (int i = 0; i < edges.size(); ++i) {
            int u = edges[i][0], v = edges[i][1];
            adj[u].emplace_back(i);
        }
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> minheap;
        minheap.emplace(0, 0);
        vector<int> dist(n, INF);
        dist[0] = 0;
        while (!minheap.empty()) {
            auto [d, u] = minheap.top();
            minheap.pop();
            if (d > dist[u]) continue;
            if (u == n - 1) return d;
            for (int i : adj[u]) {
                int v = edges[i][1], s = edges[i][2], e = edges[i][3];
                if (d > e) continue;
                int nd = max(d + 1, s + 1);
                if (nd < dist[v]) {
                    dist[v] = nd;
                    minheap.emplace(nd, v);
                }
            }
        }
        return -1;
    }
};
```

## Minimum Stability Factor of Array

### Solution 1: binary search, sparse table, gcd, greedy

```cpp
const int LOG = 30;
struct SparseGCD {
    int N;
    vector<vector<int>> st;
    SparseGCD(const vector<int> &arr) : N(arr.size()), st(LOG, vector<int>(N, 0)) {
        for (int i = 0; i < N; i++) {
            st[0][i] = arr[i];
        }
        for (int i = 1; i < LOG; i++) {
            for (int j = 0; j + (1LL << i) <= N; j++) {
                st[i][j] = gcd(st[i - 1][j], st[i - 1][j + (1LL << (i - 1))]);
            }
        }
    }
    int query(int l, int r) const {
        int k = log2(r - l + 1);
        return gcd(st[k][l], st[k][r - (1LL << k) + 1]);
    }
};
class Solution {
private:
    int K, N;
    vector<int> arr;
    bool feasible(int target) {
        int cnt = 0;
        for (int i = 0, j = 0; i < N; ++i) {
            int len = arr[i] - i + 1;
            int splits = len / (target + 1);
            if (splits) i = i + splits * (target + 1) - 1;
            cnt += splits;
        }
        return cnt <= K;
    }
public:
    int minStable(vector<int>& nums, int maxC) {
        K = maxC, N = nums.size();
        int cnt = count_if(nums.begin(), nums.end(), [](int x) {
            return x > 1;
        });
        if (cnt <= maxC) return 0;
        SparseGCD sp(nums);
        arr.assign(N, 0);
        for (int i = 0; i < N; i++) {
            int lo = i, hi = N - 1;
            while (lo < hi) {
                int mid = lo + (hi - lo + 1) / 2;
                int v = sp.query(i, mid);
                if (v > 1) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            arr[i] = lo;
        }
        int lo = 1, hi = N;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (feasible(mid)) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
    }
};
```

# Leetcode Biweekly Contest 161

## 3619. Count Islands With Total Value Divisible by K

### Solution 1: dfs, connected component

```cpp
using int64 = int64_t;
class Solution {
private:
    int R, C;
    vector<vector<int>> grid;
    bool inBounds(int r, int c) {
        return 0 <= r && r < R && 0 <= c && c < C;
    }
    int64 dfs(int r, int c) {
        int64 ans = grid[r][c];
        if (!grid[r][c]) return ans;
        grid[r][c] = 0;
        for (int dr = -1; dr <= 1; ++dr) {
            for (int dc = -1; dc <= 1; ++dc) {
                if (abs(dr) + abs(dc) != 1) continue;
                int nr = r + dr, nc = c + dc;
                if (!inBounds(nr, nc)) continue;
                ans += dfs(nr, nc);
            }
        }
        return ans;
    }
public:
    int countIslands(vector<vector<int>>& A, int k) {
        grid = A;
        R = grid.size(), C = grid[0].size();
        int ans = 0;
        for (int r = 0; r < R; ++r) {
            for (int c = 0; c < C; ++c) {
                if (!grid[r][c]) continue;
                int64 val = dfs(r, c);
                if (val % k == 0) ++ans;
            }
        }
        return ans;
    }
};
```

## 3620. Network Recovery Pathways

### Solution 1: binary search, greedy decision problem, feasibility check, directed graph, dijkstra algorithm, 

```cpp
using int64 = int64_t;
const int INF = 1e9 + 5;
class Solution {
private:
    int N;
    int64 K;
    vector<bool> online;
    vector<vector<pair<int, int>>> adj;
    bool feasible(int target) {
        priority_queue<pair<int64, int>, vector<pair<int64, int>>, greater<pair<int64, int>>> minheap;
        vector<int64> dist(N, K + 1);
        minheap.emplace(0, 0);
        dist[0] = 0;
        while (!minheap.empty()) {
            auto [cost, u] = minheap.top();
            minheap.pop();
            if (cost > dist[u]) continue;
            if (u == N - 1) return true;
            for (auto [v, w] : adj[u]) {
                if (!online[v]) continue;
                if (w < target) continue;
                int64 ncost = cost + w;
                if (ncost > K) continue;
                if (ncost < dist[v]) {
                    dist[v] = ncost;
                    minheap.emplace(ncost, v);
                }
            }
        }
        return false;
    }
public:
    int findMaxPathScore(vector<vector<int>>& edges, vector<bool>& _online, long long k) {
        online = _online;
        N = online.size(), K = k;
        adj.assign(N, vector<pair<int, int>>());
        for (const auto &edge : edges) {
            int u = edge[0], v = edge[1], w = edge[2];
            adj[u].emplace_back(v, w);
        }
        int lo = 0, hi = INF;
        while (lo < hi) {
            int mid = lo + (hi - lo + 1) / 2;
            if (feasible(mid)) lo = mid;
            else hi = mid - 1;
        }
        return feasible(lo) ? lo : -1;
    }
};
```

## 3621. Number of Integers With Popcount-Depth Equal to K I

### Solution 1:  binary number, bit digit dp, popcount, depth, memoization

Basically count the number of binary number less than or equal to n with exactly k bits set.  This can be solved with the following algorithm:

```cpp
using int64 = int64_t;
const int BITS = 64;
class Solution {
private:
    string S;
    int64 dp[BITS][BITS][2];
    int64 dfs(int idx, int rem, bool tight) {
        if (rem < 0) return 0;
        if (idx == S.size()) return rem == 0 ? 1 : 0;
        if (dp[idx][rem][tight] != -1) return dp[idx][rem][tight];
        int64 ans = 0;
        int curBit = S[idx] - '0';
        for (int i = 0; i < 2; ++i) {
            if (tight && i > curBit) continue; 
            ans += dfs(idx + 1, rem - i, tight && curBit == i);
        }
        return dp[idx][rem][tight] = ans;
    }
public:
    int64 popcountDepth(long long n, int k) {
        if (k == 0) return 1;
        vector<int> cnt(BITS, 1);
        for (int i = 2; i < BITS; ++i) {
            cnt[i] = cnt[__builtin_popcount(i)] + 1;
        }
        bool found = false;
        for (int i = BITS - 1; i >= 0; --i) {
            if ((n >> i) & 1LL) found = true;
            if (!found) continue;
            if ((n >> i) & 1LL) S.push_back('1');
            else S.push_back('0');
        }
        int64 ans = 0;
        for (int i = 1; i < BITS; ++i) {
            if (cnt[i] != k) continue;
            cout << i << endl;
            fill(&dp[0][0][0], &dp[0][0][0] + BITS * BITS * 2, -1LL);
            ans += dfs(0, i, true);
            if (i == 1) --ans; // don't count 1
        }
        return ans;
    }
};
```

# Leetcode Biweekly Contest 162

## 3634. Minimum Removals to Balance Array

### Solution 1:  sorting, sliding window, two pointers

```cpp
using int64 = int64_t;
class Solution {
public:
    int minRemoval(vector<int>& nums, int k) {
        int N = nums.size();
        sort(nums.begin(), nums.end());
        int ans = 0;
        for (int i = 0, j = 0; i < N; ++i) {
            while (static_cast<int64>(nums[i]) > static_cast<int64>(k) * nums[j]) ++j;
            ans = max(ans, i - j + 1);
        }
        return N - ans;
    }
};
```

## 3635. Earliest Finish Time for Land and Water Rides II

### Solution 1: sorting, greedy, minheaps

1. just track minheap of min end time of any ride that can start after current ride. 
1. And also track rides that start before and track minimum duration, any of these you can start directly after current ride.
1. inverse the roles of land and water rides and take the minimum of the two, since either one can be first.

```cpp
const int INF = numeric_limits<int>::max();
class Solution {
private:
    int calc(const vector<int> &A, const vector<int> &B, const vector<int> &C, const vector<int> &D) {
        int N = A.size(), M = C.size();
        vector<bool> vis(M, false);
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> minheap, startheap;
        for (int i = 0; i < M; ++i) {
            minheap.emplace(C[i] + D[i], i);
            startheap.emplace(C[i], i);
        }
        int minD = INF, ans = INF;
        vector<pair<int, int>> pool;
        for (int i = 0; i < N; ++i) {
            pool.emplace_back(A[i] + B[i], i);
        }
        sort(pool.begin(), pool.end());
        for (auto [e, i] : pool) {
            while (!startheap.empty() && startheap.top().first <= e) {
                auto [v, idx] = startheap.top();
                startheap.pop();
                minD = min(minD, D[idx]);
                vis[idx] = true;
            }
            while (!minheap.empty() && vis[minheap.top().second]) minheap.pop();
            if (minD != INF) ans = min(ans, e + minD);
            if (!minheap.empty()) ans = min(ans, minheap.top().first);
        }
        return ans;
    }
public:
    int earliestFinishTime(vector<int>& landStartTime, vector<int>& landDuration, vector<int>& waterStartTime, vector<int>& waterDuration) {
        return min(calc(landStartTime, landDuration, waterStartTime, waterDuration), calc(waterStartTime, waterDuration, landStartTime, landDuration));
    }
};
```

## 3636. Threshold Majority Queries

### Solution 1: Mo's algorithm, offline queries, sorting, maxheap, frequency

```cpp
int block_size;

struct Query {
    int l, r, threshold, idx;
    bool operator<(const Query &other) const {
        int b1 = l / block_size, b2 = other.l / block_size;
        if (b1 != b2) return b1 < b2;
        if (b1 & 1) return r > other.r;
        return r < other.r;
    }
};

struct Point {
    int f, v;
    bool operator<(const Point &other) const {
        if (f != other.f) return f < other.f;
        return v > other.v;
    }
};

class Solution {
private:
    int N, M;
    vector<int> arr;
    map<int, int> freq;
    priority_queue<Point> maxheap;
    void add(int pos) {
        maxheap.emplace(++freq[arr[pos]], arr[pos]);
    }
    void remove(int pos) {
        maxheap.emplace(--freq[arr[pos]], arr[pos]);
    }
    int getAnswer(int threshold) {
        int ans = 0;
        while (!maxheap.empty() && maxheap.top().f != freq[maxheap.top().v]) maxheap.pop();
        if (maxheap.empty()) return -1;
        if (maxheap.top().f < threshold) return -1;
        return maxheap.top().v;
    }
    vector<int> mo_s_algorithm(vector<Query> queries) {
        block_size = max(1, (int)(N / max(1.0, sqrt(M))));
        vector<int> answers(M);
        sort(queries.begin(), queries.end());
        int curL = 0, curR = -1;
        for (const Query& q : queries) {
            while (curL > q.l) add(--curL);
            while (curR < q.r) add(++curR);
            while (curL < q.l) { remove(curL); ++curL; }
            while (curR > q.r) { remove(curR); --curR; }
            answers[q.idx] = getAnswer(q.threshold);
        }
        return answers;
    }
public:
    vector<int> subarrayMajority(vector<int>& nums, vector<vector<int>>& queries) {
        N = nums.size(), M = queries.size();
        arr = nums;
        vector<Query> moQuery;
        for (int i = 0; i < M; ++i) {
            int l = queries[i][0], r = queries[i][1], thres = queries[i][2];
            moQuery.emplace_back(l, r, thres, i);
        }
        vector<int> ans = mo_s_algorithm(moQuery);
        return ans;
    }
};
```

# Leetcode Biweekly Contest 163

## 3649. Number of Perfect Pairs

### Solution 1: fenwick tree, binary search, sorting, coordinate compression

```cpp
using int64 = int64_t;
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
    int64 perfectPairs(vector<int>& nums) {
        int N = nums.size();
        vector<int> values;
        for (int x : nums) {
            values.emplace_back(abs(x));
        }
        sort(values.begin(), values.end());
        values.erase(unique(values.begin(), values.end()), values.end());
        int M = values.size();
        FenwickTree<int> ftp, ftn;
        ftp.init(M + 1), ftn.init(M + 1);
        int64 ans = 0;
        for (int a : nums) {
            int a_i = lower_bound(values.begin(), values.end(), abs(a)) - values.begin();
            // find specific b within range a/2 and 2a
            int lb = (abs(a) + 1) / 2, ub = 2 * abs(a);
            int lb_i = lower_bound(values.begin(), values.end(), lb) - values.begin();
            int ub_i = upper_bound(values.begin(), values.end(), ub) - values.begin() - 1;
            ans += ftp.query(lb_i + 1, ub_i + 1) + ftn.query(lb_i + 1, ub_i + 1);
            if (a >= 0) ftp.update(a_i + 1, 1);
            else ftn.update(a_i + 1, 1);
        }
        return ans;
    }
};
```

## 3650. Minimum Cost Path with Edge Reversals

### Solution 1: undirected graph, transpose undirected graph, min heap, dijsktra

```cpp
const int INF = numeric_limits<int>::max();
class Solution {
public:
    int minCost(int n, vector<vector<int>>& edges) {
        vector<int> dist(n, INF);
        vector<vector<pair<int, int>>> adj(n, vector<pair<int, int>>()), tadj(n, vector<pair<int, int>>());
        for (const auto &edge : edges) {
            int u = edge[0], v = edge[1], w = edge[2];
            adj[u].emplace_back(v, w);
            adj[v].emplace_back(u, 2 * w);
        }
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> minheap;
        minheap.emplace(0, 0);
        while (!minheap.empty()) {
            auto [cost, u] = minheap.top();
            minheap.pop();
            if (u == n - 1) return cost;
            if (cost > dist[u]) continue;
            for (auto [v, w] : adj[u]) {
                int ncost = cost + w;
                if (ncost < dist[v]) {
                    dist[v] = ncost;
                    minheap.emplace(ncost, v);
                }
            }
        }
        return -1;
    }
};
```

## 3651. Minimum Cost Path with Teleportations

### Solution 1: dynamic programming, grid, prefix min, bottom-up dp

```cpp
const int INF = numeric_limits<int>::max();
class Solution {
public:
    int minCost(vector<vector<int>>& grid, int k) {
        int R = grid.size(), C = grid[0].size();
        vector<vector<int>> dp(R, vector<int>(C, INF));
        int maxVal = 0;
        for (int r = 0; r < R; ++r) {
            for (int c = 0; c < C; ++c) {
                maxVal = max(maxVal, grid[r][c]);
            }
        }
        vector<int> pref(maxVal + 1, INF);
        for (int i = 0; i <= k; ++i) {
            for (int r = R - 1; r >= 0; --r) {
                for (int c = C - 1 ; c >= 0; --c) {
                    if (r == R - 1 && c == C - 1) {
                        dp[r][c] = 0;
                        continue;
                    }
                    if (r + 1 < R) {
                        dp[r][c] = min(dp[r][c], dp[r + 1][c] + grid[r + 1][c]);
                    }
                    if (c + 1 < C) {
                        dp[r][c] = min(dp[r][c], dp[r][c + 1] + grid[r][c + 1]);
                    }
                    if (i > 0) { // teleportation
                        dp[r][c] = min(dp[r][c], pref[grid[r][c]]);
                    }
                }
            }
            pref.assign(maxVal + 1, INF);
            for (int r = 0; r < R; ++r) {
                for (int c = 0; c < C; ++c) {
                    pref[grid[r][c]] = min(pref[grid[r][c]], dp[r][c]);
                }
            }
            for (int j = 1; j <= maxVal; ++j) {
                pref[j] = min(pref[j], pref[j - 1]);
            }
        }
        return dp[0][0];
    }
};
```

# Leetcode Biweekly Contest 164

## 3664. Two-Letter Card Game

### Solution 1: greedy, frequency, grouping

```cpp
class Solution {
private:
    char X;
    int calc(vector<int> &freq, int doubles) {
        freq[X - 'a'] = doubles;
        int m = 0, total = 0;
        for (int i = 0; i < 10; ++i) {
            total += freq[i];
            if (freq[i] > m) m = freq[i];
        }
        int rem = 2 * m > total ? 2 * m - total : total % 2;
        int consumed = total - rem;
        return consumed / 2;
    }
public:
    int score(vector<string>& cards, char x) {
        X = x;
        int N = cards.size(), doubles = 0, ans = 0;
        vector<int> cnt1(10, 0), cnt2(10, 0);
        for (const string &s : cards) {
            if (s[0] != x && s[1] != x) continue;
            if (s[0] == x && s[1] == x) {
                doubles++;
                continue;
            }
            if (s[0] == x) cnt1[s[1] - 'a']++;
            else cnt2[s[0] - 'a']++;
        }
        for (int i = 0; i <= doubles; ++i) {
            int cand = calc(cnt1, i) + calc(cnt2, doubles - i);
            ans = max(ans, cand);
        }
        return ans;
    }
};
```

## 3665. Twisted Mirror Path Count

### Solution 1: dynamic programming

```cpp
using int64 = int64_t;
const int MOD = 1e9 + 7;
class Solution {
public:
    int uniquePaths(vector<vector<int>>& grid) {
        int R = grid.size(), C = grid[0].size();
        vector<vector<vector<int64>>> dp(R, vector<vector<int64>>(C, vector<int64>(2, 0)));
        dp[0][0][0] = 1;
        for (int r = 0; r < R; ++r) {
            for (int c = 0; c < C; ++c) {
                if (r > 0) {
                    dp[r][c][1] = (dp[r][c][1] + dp[r - 1][c][0]) % MOD;
                    if (!grid[r - 1][c]) dp[r][c][1] = (dp[r][c][1] + dp[r - 1][c][1]) % MOD;
                }
                if (c > 0) {
                    dp[r][c][0] = (dp[r][c][0] + dp[r][c - 1][1]) % MOD;
                    if (!grid[r][c - 1]) dp[r][c][0] = (dp[r][c][0] + dp[r][c - 1][0]) % MOD;
                }
            }
        }
        int ans = (dp[R - 1][C - 1][0] + dp[R - 1][C - 1][1]) % MOD;
        return ans;
    }
};
```

## 3666. Minimum Operations to Equalize Binary String

### Solution 1: bfs, queue, set, undirected graph, binary search

```cpp
const int INF = numeric_limits<int>::max();
class Solution {
public:
    int minOperations(string s, int k) {
        int N = s.size();
        int z0 = 0;
        set<int> neighbors[2];
        for (int i = 0; i < N; ++i) {
            if (s[i] == '0') z0++;
            neighbors[i % 2].insert(i);
        }
        neighbors[N % 2].insert(N);
        vector<int> dist(N + 1, INF);
        dist[z0] = 0;
        neighbors[z0 % 2].erase(z0);
        queue<int> q;
        q.emplace(z0);
        while (!q.empty()) {
            int z = q.front();
            q.pop();
            if (z == 0) return dist[z];
            int lo = max(0, k - (N - z)); // min 0 to flip
            int hi = min(z, k); // max 0 to flip
            int l = z + k - 2 * hi;
            int r = z + k - 2 * lo; // (z - i) + (k - i) decrease z by i, and flipped k - i 1s to 0
            int p = (z + k) % 2;
            auto it = neighbors[p].lower_bound(l);
            for (auto it = neighbors[p].lower_bound(l); it != neighbors[p].end() && *it <= r;) {
                int v = *it;
                q.emplace(v);
                dist[v] = dist[z] + 1;
                ++it; // before erase else iterator is removed
                neighbors[p].erase(v);
            }
        }
        return -1;
    }
};
```

# Leetcode Biweekly Contest 165

## 3679. Minimum Discards to Balance Inventory

### Solution 1: fixed size sliding window, window frequency, greedy

```cpp
class Solution {
public:
    int minArrivalsToDiscard(vector<int>& arrivals, int w, int m) {
        int N = arrivals.size();
        unordered_map<int, int> windowFreq;
        int ans = 0;
        for (int i = 0; i < N; ++i) {
            if (windowFreq[arrivals[i]] == m) {
                ++ans;
                arrivals[i] = 0; // dummy value
            }
            windowFreq[arrivals[i]]++;
            if (i >= w - 1) {
                windowFreq[arrivals[i - w + 1]]--;
            }
        }
        return ans;
    }
};
```

## 3680. Generate Schedule

### Solution 1: random algorithm, random shuffle of array, greedy corrections

```cpp
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

class Solution {
private:
    vector<vector<int>> games;
    bool isValid(int i, int j) {
        return games[i][0] != games[j][0] && games[i][0] != games[j][1] && games[i][1] != games[j][0] && games[i][1] != games[j][1];
    }
public:
    vector<vector<int>> generateSchedule(int n) {
        if (n < 5) return {};
        games.clear();
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                games.push_back({i, j});
            }
        }
        while (true) {
            shuffle(games.begin(), games.end(), rng);
            bool isGood = true;
            for (int i = 1; i < n * (n - 1); ++i) {
                int j = i;
                while (j < n * (n - 1) && !isValid(i - 1, j)) ++j;
                if (j == n * (n - 1)) {
                    isGood = false;
                    break;
                }
                swap(games[i], games[j]);
            }
            if (isGood) return games;
        }
        return {};
    }
};
```

## 3681. Maximum XOR of Subsequences

### Solution 1: linear algebra, linear indepdendence, linear basis, xor operation

```cpp
class Solution {
public:
    int maxXorSubsequences(vector<int>& nums) {
        int N = nums.size();
        vector<int> basis;
        for (int i = 0; i < N; ++i) {
            for (int b : basis) {
                nums[i] = min(nums[i], nums[i] ^ b);
            }
            if (!nums[i]) continue;
            for (int &b : basis) {
                b = min(b, b ^ nums[i]);
            }
            basis.emplace_back(nums[i]);
        }
        sort(basis.rbegin(), basis.rend());
        int ans = 0;
        for (int b : basis) {
            if ((ans ^ b) > ans) ans ^= b;
        }
        return ans;
    }
};
```

# Leetcode Biweekly Contest 166

## 3693. Climbing Stairs II

### Solution 1: dynamic programming

```cpp
const int INF = numeric_limits<int>::max();
class Solution {
public:
    int climbStairs(int n, vector<int>& costs) {
        costs.emplace(costs.begin(), 0);
        vector<int> dp(n + 1, INF);
        dp[0] = 0;
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= 3; ++j) {
                if (i - j < 0) break;
                dp[i] = min(dp[i], dp[i - j] + costs[i] + j * j);
            }
        }
        return dp[n];
    }
};
```

## 3694. Distinct Points Reachable After Substring Removal

### Solution 1: set, sliding window

```cpp
class Solution {
private:
    pair<int, int> getCoords(char ch) {
        if (ch == 'U') return {0, 1};
        if (ch == 'D') return {0, -1};
        if (ch == 'L') return {-1, 0};
        return {1, 0};
    }
public:
    int distinctPoints(string s, int k) {
        int N = s.size();
        set<pair<int, int>> seen;
        int x = 0, y = 0, dx, dy;
        for (int i = k; i < N; ++i) {
            tie(dx, dy) = getCoords(s[i]);
            x += dx;
            y += dy;
        }
        seen.emplace(x, y);
        for (int i = 0; i < N - k; ++i) {
            tie(dx, dy) = getCoords(s[i]);
            x += dx; y += dy;
            tie(dx, dy) = getCoords(s[i + k]);
            x -= dx; y -= dy;
            seen.emplace(x, y);
        }
        return seen.size();
    }
};
```

## 3695. Maximize Alternating Sum Using Swaps

### Solution 1: union find, connected components, sorting, greedy, alternating sum

```cpp
using int64 = long long;
struct UnionFind {
    vector<int> parents, size;
    UnionFind(int n) {
        parents.resize(n);
        iota(parents.begin(),parents.end(),0);
        size.assign(n,1);
    }

    int find(int i) {
        if (i==parents[i]) {
            return i;
        }
        return parents[i]=find(parents[i]);
    }

    void unite(int i, int j) {
        i = find(i), j = find(j);
        if (i!=j) {
            if (size[j]>size[i]) {
                swap(i,j);
            }
            size[i]+=size[j];
            parents[j]=i;
        }
    }

    bool same(int i, int j) {
        return find(i) == find(j);
    }
    
    vector<vector<int>> groups() {
        int n = parents.size();
        unordered_map<int, vector<int>> group_map;
        for (int i = 0; i < n; ++i) {
            group_map[find(i)].emplace_back(i);
        }
        vector<vector<int>> res;
        for (auto& [_, group] : group_map) {
            res.emplace_back(move(group));
        }
        return res;
    }
};
class Solution {
public:
    int64 maxAlternatingSum(vector<int>& nums, vector<vector<int>>& swaps) {
        int N = nums.size();
        UnionFind dsu(N);
        for (const auto &pair : swaps) {
            dsu.unite(pair[0], pair[1]);
        }
        vector<vector<int>> groups = dsu.groups();
        vector<int> arr(N, 0);
        for (auto &g : groups) {
            vector<int> values;
            sort(g.begin(), g.end(), [](const int x, const int y) {
                return x % 2 > y % 2;
            }); // odd is first x % 2 = 1 when odd 
            for (int i : g) {
                values.emplace_back(nums[i]);
            }
            sort(values.begin(), values.end());
            for (int i = 0; i < g.size(); ++i) {
                arr[g[i]] = values[i];
            }
        }
        vector<int> indices(N);
        iota(indices.begin(), indices.end(), 0);
        int64 ans = accumulate(indices.begin(), indices.end(), 0LL, [&arr](const int64 total, const int64 i) {
            return total + (i % 2 ? -arr[i] : arr[i]);
        });
        return ans;
    }
};
```

# Leetcode Biweekly Contest 167

## 3708. Longest Fibonacci Subarray

### Solution 1: fibonacci sequence

```cpp
class Solution {
public:
    int longestSubarray(vector<int>& nums) {
        int ans = 2, cnt = 2, N = nums.size();
        for (int i = 2; i < N; ++i) {
            if (nums[i - 1] + nums[i - 2] != nums[i]) {
                cnt = 1;
            }
            cnt++;
            ans = max(ans, cnt);
        }
        return ans;
    }
};
```

## 3709. Design Exam Scores Tracker

### Solution 1: prefix sum, binary search

```cpp
using int64 = long long;
class ExamTracker {
private:
    vector<int> times;
    vector<int64> psum;
public:
    ExamTracker() {
        
    }
    
    void record(int time, int score) {
        times.emplace_back(time);
        int64 cur = !psum.empty() ? psum.back() : 0;
        psum.emplace_back(cur + score);
    }
    
    int64 totalScore(int startTime, int endTime) {
        int N = times.size();
        int l = lower_bound(times.begin(), times.end(), startTime) - times.begin();
        int r = upper_bound(times.begin(), times.end(), endTime) - times.begin() - 1;
        if (r < l) return 0;
        int64 ans = psum[r];
        if (l > 0) ans -= psum[l - 1];
        return ans;
    }
};

```

## 3710. Maximum Partition Factor

### Solution 1: binary search, bipartite graph, manhattan distance, dfs, stack, 2-coloring

1. binary search on the answer, for each mid value build a graph where an edge exists between two points if their manhattan distance is less than mid.
2. basically all the points that have an edge need to belong to a different set

```cpp
class Solution {
private:
    vector<vector<int>> adj, A;
    vector<int> colors;
    bool bipartite(int source) {
        stack<int> stk;
        stk.push(source);
        colors[source] = 1;
        bool ans = true;
        while (!stk.empty()) {
            int u = stk.top();
            stk.pop();
            for (int v : adj[u]) {
                if (colors[v] == 0) { // unvisited
                    colors[v] = 3 - colors[u]; // needs to be different color of two possible values 1 and 2
                    stk.push(v);
                } else if (colors[u] == colors[v]) {
                    ans = false;
                }
            }
        }
        return ans;
    }
    int manhattan(int x1, int y1, int x2, int y2) {
        return abs(x1 - x2) + abs(y1 - y2);
    }
    bool possible(int target) {
        int N = A.size();
        adj.assign(N, vector<int>());
        // BUILD GRAPH
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                int x1 = A[i][0], y1 = A[i][1];
                int x2 = A[j][0], y2 = A[j][1];
                int dist = manhattan(x1, y1, x2, y2);
                if (dist < target) {
                    adj[i].emplace_back(j);
                    adj[j].emplace_back(i);
                }
            }
        }
        // BIPARTITE
        colors.assign(N, 0);
        for (int i = 0; i < N; ++i) {
            if (colors[i]) continue;
            if (!bipartite(i)) return false;
        }
        return true;
    }
public:
    int maxPartitionFactor(vector<vector<int>>& points) {
        int N = points.size();
        if (N == 2) return 0;
        A = points;
        int lo = 0, hi = 1e9;
        while (lo < hi) {
            int mid = lo + (hi - lo + 1) / 2;
            if (possible(mid)) lo = mid;
            else hi = mid - 1;
        }
        return lo;
    }
};
```

# Leetcode Biweekly Contest 168

## 3723. Maximize Sum of Squares of Digits

### Solution 1: greedy, take largest first

```cpp
class Solution {
public:
    string maxSumOfSquares(int num, int sum) {
        string ans;
        for (int i = 0; i < num; ++i) {
            int dig = min(9, sum);
            ans += dig + '0';
            sum -= dig;
        }
        if (sum) return "";
        return ans;
    }
};
```

## 3724. Minimum Operations to Transform Array

### Solution 1: loop, math, ordering of elements

```cpp
using int64 = long long;
class Solution {
private:
    int sign(int x) {
        return x >= 0 ? 1 : -1;
    }
public:
    int64 minOperations(vector<int>& nums1, vector<int>& nums2) {
        int N = nums1.size();
        int64 ans = 1;
        int app = 1e9;
        for (int i = 0; i < N; ++i) {
            ans += abs(nums1[i] - nums2[i]);
            int d1 = sign(nums2.back() - nums1[i]), d2 = sign(nums2.back() - nums2[i]);
            if (d1 != d2) app = 0;
            app = min({app, abs(nums1[i] - nums2.back()), abs(nums2[i] - nums2.back())});
        }
        ans += app;
        return ans;
    }
};
```

## 3725. Count Ways to Choose Coprime Integers from Rows

### Solution 1: gcd, dynamic programming, grid, counting

```cpp
const int MOD = 1e9 + 7, MAXN = 151;
class Solution {
public:
    int countCoprime(vector<vector<int>>& mat) {
        int R = mat.size(), C = mat[0].size();
        vector<int> dp(MAXN, 0), ndp(MAXN, 0);
        dp[0] = 1;
        for (int r = 0; r < R; ++r) {
            ndp.assign(MAXN, 0);
            for (int c = 0; c < C; ++c) {
                for (int g = 0; g < MAXN; ++g) {
                    int ng = gcd(g, mat[r][c]);
                    ndp[ng] = (ndp[ng] + dp[g]) % MOD;
                }
            }
            swap(dp, ndp);
        }
        return dp[1];
    }
};
```

# Leetcode Biweekly Contest 169

## 

### Solution 1: 

1. LIS problem, dp with binary search patience algorithm
1. that is it and add 1 to it, cause the missing element can be whatever and can extend the LIS by 1

```cpp

```

## 

### Solution 1: 

1. map values to -1, +1 (if it is target)
1. find the number of subarrays with positive sum, so using prefix sums just find for j > i, where pref[j] > pref[i].
1. This can be done with fenwick tree to count number of prefix sums less than current prefix sum (range query)

```cpp

```

# Leetcode Biweekly Contest 170

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