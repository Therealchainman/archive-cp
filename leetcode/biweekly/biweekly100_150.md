# Leetcode Biweekly Rounds 100-149

# Leetcode BiWeekly Contest 126

## 3081. Replace Question Marks in String to Minimize Its Value

### Solution 1:  min heap, greedy, lexicographically smallest

```py
class Solution:
    def minimizeStringValue(self, s: str) -> str:
        n = len(s)
        minheap = []
        for ch in string.ascii_lowercase:
            heappush(minheap, (s.count(ch), ch))
        add = []
        s = list(s)
        for _ in range(s.count("?")):
            x, ch = heappop(minheap)
            add.append(ch)
            heappush(minheap, (x + 1, ch))
        add.sort(reverse = True)
        for i in range(n):
            if s[i] != "?": continue 
            s[i] = add.pop()
        return "".join(s)
```

## 3082. Find the Sum of the Power of All Subsequences

### Solution 1:  combinatorics, dp, counting, subsequences

```py
class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        n = len(nums)
        freq = Counter({(1, 0): 1}) # (size, sum) -> freq
        ans = 0
        MOD = int(1e9) + 7
        for num in nums:
            nfreq = freq.copy()
            for (sz, su), f in freq.items():
                if su + num == k: ans = (ans + f * pow(2, n - sz, MOD)) % MOD
                if su + num >= k: continue 
                nfreq[(sz + 1, su + num)] += f
            freq = nfreq
        return ans
```



# Leetcode BiWeekly Contest 127

## 3096. Minimum Levels to Gain More Points

### Solution 1:  prefix sum, suffix sum

```py
class Solution:
    def minimumLevels(self, arr: List[int]) -> int:
        n = len(arr)
        ssum = 0
        for x in arr:
            if x == 1: ssum += 1
            else: ssum -= 1
        psum = 0
        for i in range(n - 1):
            psum += 1 if arr[i] == 1 else -1
            ssum -= 1 if arr[i] == 1 else -1
            if psum > ssum: return i + 1
        return -1
```

## 3097. Shortest Subarray With OR at Least K II

### Solution 1:  bit manipulation, sliding window, frequency array

```py
class Solution:
    def minimumSubarrayLength(self, nums: List[int], K: int) -> int:
        n = len(nums)
        ans = math.inf
        cur = j = 0
        freq = [0] * 30
        for i in range(n):
            for k in range(30):
                if (nums[i] >> k) & 1:
                    freq[k] += 1
                    if freq[k] == 1: cur |= 1 << k
            while cur >= K and j <= i:
                ans = min(ans, i - j + 1)
                for k in range(30):
                    if (nums[j] >> k) & 1:
                        freq[k] -= 1
                        if freq[k] == 0: cur ^= 1 << k
                j += 1
        return ans if ans < math.inf else -1
```

## 3098. Find the Sum of Subsequence Powers

### Solution 1:  sort, dp, countings

```py
class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        n = len(nums)
        mod = int(1e9) + 7
        nums.sort()
        @cache
        def dp(idx, sz, last, mind):
            if idx == n:
                if sz == k: return mind
                return 0
            ans = 0
            # take
            if sz < k:
                ans = (ans + dp(idx + 1, sz + 1, nums[idx], min(mind, nums[idx] - last))) % mod
            ans = (ans + dp(idx + 1, sz, last, mind)) % mod
            return ans
        return dp(0, 0, -math.inf, math.inf)
```

```py
class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        n = len(nums)
        mod = int(1e9) + 7
        nums.sort()
        dp = Counter({(0, -math.inf, math.inf): 1})
        for x in nums:
            ndp = Counter()
            for (sz, last, mind), cnt in dp.items():
                if sz < k:
                    ndp[(sz + 1, x, min(mind, x - last))] += cnt
                    ndp[(sz + 1, x, min(mind, x - last))] %= mod
                ndp[(sz, last, mind)] += cnt 
                ndp[(sz, last, mind)] %= mod
            dp = ndp
        ans = 0
        for (sz, _, mind), cnt in dp.items():
            if sz == k: ans = (ans + cnt * mind) % mod
        return ans
```

Could use coordinate compression and remove the Counter.


# Leetcode BiWeekly Contest 128

## 3112. Minimum Time to Visit Disappearing Nodes

### Solution 1:  dijkstra, shortest path, undirected graph

```py
def dijkstra(adj, src, dis):
    N = len(adj)
    min_heap = [(0, src)]
    dist = [math.inf] * N
    while min_heap:
        cost, u = heapq.heappop(min_heap)
        if cost >= dis[u]: continue
        if cost >= dist[u]: continue
        dist[u] = cost
        for v, w in adj[u]:
            if cost + w < dist[v]: heapq.heappush(min_heap, (cost + w, v))
    return dist
class Solution:
    def minimumTime(self, n: int, edges: List[List[int]], disappear: List[int]) -> List[int]:
        pedges = defaultdict(lambda: math.inf)
        adj = [[] for _ in range(n)]
        for u, v, w in edges:
            pedges[(u, v)] = min(pedges[(u, v)], w)
            pedges[(v, u)] = min(pedges[(v, u)], w)
        for (u, v), w in pedges.items():
            adj[u].append((v, w))
        dist_arr = dijkstra(adj, 0, disappear)
        ans = [-1] * n
        for u in range(n):
            if dist_arr[u] < disappear[u]: ans[u] = dist_arr[u]
        return ans
```

## 3113. Find the Number of Subarrays Where Boundary Elements Are Maximum

### Solution 1:  groups, sliding window, range maximum query sparse tables

```py

class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        lg = [0] * (n + 1)
        for i in range(2, n + 1):
            lg[i] = lg[i // 2] + 1
        LOG = lg[-1] + 1
        st = [[math.inf] * n for _ in range(LOG)]
        st[0] = nums[:]
        for i in range(1, LOG):
            j = 0
            while (j + (1 << (i - 1))) < n:
                st[i][j] = max(st[i - 1][j], st[i - 1][j + (1 << (i - 1))])
                j += 1
        def query(left, right):
            length = right - left + 1
            i = lg[length]
            return max(st[i][left], st[i][right - (1 << i) + 1])
        pos = defaultdict(list)
        for i in range(n):
            pos[nums[i]].append(i)
        ans = 0
        for val, arr in pos.items():
            delta = 0
            start = arr[0]
            for end in arr:
                if query(start, end) != val:
                    start = end
                    delta = 0
                delta += 1
                ans += delta
        return ans
```

# Leetcode Biweekly Contest 129

## 3129. Find All Possible Stable Binary Arrays I

### Solution 1:  recursive dynamic programming, memoization, counting

```cpp
const int MOD = 1e9 + 7;
class Solution {
private:
    int LIMIT;
    vector<vector<vector<vector<int>>>> dp;
    int64 dfs(int zero, int one, int last, int len) {
        if (zero < 0 || one < 0) return 0;
        if (len > LIMIT) return 0;
        if (zero == 0 && one == 0) return 1;
        if (dp[zero][one][last][len] != -1) return dp[zero][one][last][len];
        int64 ans = dfs(zero - 1, one, 0, last == 0 ? len + 1 : 1);
        ans += dfs(zero, one - 1, 1, last == 1 ? len + 1 : 1);
        ans %= MOD;
        return dp[zero][one][last][len] = ans;
    }
public:
    int numberOfStableArrays(int zero, int one, int limit) {
        LIMIT = limit;
        dp.assign(zero + 1, vector<vector<vector<int>>>(one + 1, vector<vector<int>>(2, vector<int>(LIMIT + 1, -1))));
        return dfs(zero, one, 0, 0);
    }
};
```

# Leetcode BiWeekly Contest 133

## 3190

### solution 1: modulo 3

```cpp
class Solution {
public:
    int minimumOperations(vector<int>& nums) {
        int ans = accumulate(nums.begin(), nums.end(), 0, [](int accum, int x) {
            return accum + (x % 3 != 0 ? 1 : 0);
        });
        return ans;
    }
};
```

## 3193. Count the Number of Inversions

### Solution 1:  dynamic programming, pointer, window sum optimization, space optimization

```cpp
class Solution {
public:
    int numberOfPermutations(int n, vector<vector<int>>& requirements) {
        sort(requirements.begin(), requirements.end());
        int m = requirements.size();
        const int COUNT = 400, MOD = 1e9 + 7;
        vector<int> dp(COUNT + 1, 0), dp1(COUNT + 1);
        dp[0] = 1;
        for (int i = 0, j = 0; i < n; i++) {
            int mark = -1, wsum = 0;
            if (j < m && requirements[j][0] == i) {
                mark = requirements[j][1];
                j++;
            }
            for (int k = 0; k <= COUNT; k++) {
                wsum = (wsum + dp[k]) % MOD;
                dp1[k] = wsum;
                if (mark != -1 && k != mark) dp1[k] = 0;
                if (k >= i) wsum = (wsum - dp[k - i] + MOD) % MOD;
            }
            swap(dp, dp1);
        }
        return dp[requirements.end()[-1][1]];
    }
};
```

# Leetcode BiWeekly Contest 134

## Alternating Groups II

### Solution 1: two pointers

```cpp
class Solution {
public:
    int numberOfAlternatingGroups(vector<int>& colors, int k) {
        int ans = 0, N = colors.size();
        for (int i = 0, j = 0; i < N; j++) {
            if (j > i && colors[(j - 1 + N) % N] == colors[j % N]) {
                i = j;
            }
            if (j - i + 1 == k) {
                ans++;
                i++;
            }
        }
        return ans;
    }
};
```

## Number of Subarrays With AND Value of K

### Solution 1:  sparse table, range AND queries, binary search

```py
class ST_And:
    def __init__(self, nums):
        self.nums = nums
        self.n = len(nums)
        self.LOG = 21 # 10^9
        self.build()

    def op(self, x, y):
        return x & y

    def build(self):
        self.lg = [0] * (self.n + 1)
        for i in range(2, self.n + 1):
            self.lg[i] = self.lg[i // 2] + 1
        self.st = [[0] * self.n for _ in range(self.LOG)]
        for i in range(self.n): 
            self.st[0][i] = self.nums[i]
        # CONSTRUCT SPARSE TABLE
        for i in range(1, self.LOG):
            j = 0
            while (j + (1 << (i - 1))) < self.n:
                self.st[i][j] = self.op(self.st[i - 1][j], self.st[i - 1][j + (1 << (i - 1))])
                j += 1

    def query(self, l, r):
        length = r - l + 1
        i = self.lg[length]
        return self.op(self.st[i][l], self.st[i][r - (1 << i) + 1])
class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        ans = 0
        st = ST_And(nums)
        def upper_bound(start):
            lo, hi = start, n - 1
            while lo < hi:
                mid = (lo + hi + 1) >> 1
                if st.query(start, mid) < k:
                    hi = mid - 1
                else:
                    lo = mid
            return lo
        def lower_bound(start):
            lo, hi = start, n - 1
            while lo < hi:
                mid = (lo + hi) >> 1
                if st.query(start, mid) <= k:
                    hi = mid
                else:
                    lo = mid + 1
            return lo
        for i in range(n):
            l, r = lower_bound(i), upper_bound(i)
            if r == l and st.query(i, l) != k: continue
            ans += r - l + 1
        return ans
```

# Leetcode Biweekly Contest 135

## Minimum Array Changes to Make Differences Equal

### Solution 1:  sort, multiset, greedy

```cpp
class Solution {
public:
    int minChanges(vector<int>& nums, int k) {
        int N = nums.size();
        vector<pair<int, int>> data;
        for (int i = 0; i < N / 2; i++) {
            int a = nums[i], b = nums[N - i - 1];
            if (a > b) swap(a, b);
            int p = max(k - a, b);
            data.emplace_back(b - a, p);
        }
        sort(data.begin(), data.end());
        multiset<int> pool;
        int ans = N / 2, ptr = 0, cur = N / 2;
        for (int x = 0; x <= k; x++) {
            int cnt_equal = 0;
            for (auto it = pool.begin(); it != pool.end(); it++) {
                if (x <= *it) break;
                pool.erase(it);
                cur++;
            }
            while (ptr < N / 2 && data[ptr].first == x) {
                pool.insert(data[ptr++].second);
                cnt_equal++;
            }
            ans = min(ans, cur - cnt_equal);
        }
        return ans;
    }
};
```

## Maximum Score From Grid Operations

### Solution 1:  dynamic programming with 3 states, prefix sums, dp on matrix

```cpp

```

# Leetcode Biweekly Contest 136

## Minimum Number of Flips to Make Binary Grid Palindromic II

### Solution 1:  connected components, counting, greedy

```cpp
class Solution {
public:
    int minFlips(vector<vector<int>>& grid) {
        int R = grid.size(), C = grid[0].size();
        int ans = 0;
        for (int r = 0; r < R / 2; r++) {
            for (int c = 0; c < C / 2; c++) {
                int cnt1 = grid[r][c] + grid[r][C - c - 1] + grid[R - r - 1][c] + grid[R - r - 1][C - c - 1];
                ans += min(cnt1, 4 - cnt1);
            }
        }
        int freq[3];
        memset(freq, 0, sizeof(freq));
        if (R & 1) {
            for (int c = 0; c < C / 2; c++) {
                int cnt = grid[R / 2][c] + grid[R / 2][C - c - 1];
                freq[cnt]++;
            }
        }
        if (C & 1) {
            for (int r = 0; r < R / 2; r++) {
                int cnt = grid[r][C / 2] + grid[R - r - 1][C / 2];
                freq[cnt]++;
            }
        }
        if ((freq[2] & 1) && freq[1] == 0) freq[1] += 2;
        ans += freq[1];
        if ((R & 1) && (C & 1)) {
            ans += grid[R / 2][C / 2];
        }
        return ans;
    }
};
```

## 3241. Time Taken to Mark All Nodes

### Solution 1:  reroot dp on tree, dfs

```cpp
class Solution {
public:
    int N;
    vector<vector<int>> adj;
    vector<int> ans, max_d1, max_d2, max_c1, max_c2, max_p;
    void dfs1(int u, int p = -1) {
        for (int v : adj[u]) {
            if (v == p) continue;
            dfs1(v, u);
            int cd = max_d1[v] + (v % 2 == 0 ? 2 : 1);
            if (cd > max_d1[u]) {
                max_d2[u] = max_d1[u];
                max_c2[u] = max_c1[u];
                max_d1[u] = cd;
                max_c1[u] = v;
            } else if (cd > max_d2[u]) {
                max_d2[u] = cd;
                max_c2[u] = v;
            }
        }
    }
    void dfs2(int u, int p = -1) {
        ans[u] = max(max_p[u], max_d1[u]);
        for (int v : adj[u]) {
            if (v == p) continue;
            if (v != max_c1[u]) {
                max_p[v] = max(max_p[u], max_d1[u]) + (u % 2 == 0 ? 2 : 1);
            } else {
                max_p[v] = max(max_p[u], max_d2[u]) + (u % 2 == 0 ? 2 : 1);
            }
            dfs2(v, u);
        }
    }
    vector<int> timeTaken(vector<vector<int>>& edges) {
        N = edges.size() + 1;
        adj.assign(N, {});
        for (const auto &edge : edges) {
            int u = edge[0], v = edge[1];
            adj[u].push_back(v);
            adj[v].push_back(u);
        }
        max_d1.assign(N, 0);
        max_d2.assign(N, 0);
        max_c1.assign(N, -1);
        max_c2.assign(N, -1);
        dfs1(0);
        ans.resize(N);
        max_p.assign(N, 0);
        dfs2(0);
        return ans;
    }
};
```

# Leetcode Biweekly Contest 137

## 

### Solution 1: 

```cpp

```

## 

### Solution 1: 

```cpp

```

# Leetcode Biweekly Contest 138

## 

### Solution 1: 

```cpp

```

## 3273. Minimum Amount of Damage Dealt to Bob

### Solution 1:  greedy, sorting, exchange argument

```cpp
int ceil(int x, int y) {
    return (x + y - 1) / y;
}
struct Monster {
    long long dmg, turns;
    Monster() {}
    Monster(long long dmg, long long turns) : dmg(dmg), turns(turns) {}
    bool operator<(const Monster &other) const {
        long long cost1 = turns * (dmg + other.dmg) + other.dmg * other.turns;
        long long cost2 = other.turns * (dmg + other.dmg) + dmg * turns;
        return cost1 < cost2;
    }
};
class Solution {
public:
    long long minDamage(int power, vector<int>& damage, vector<int>& health) {
        int N = damage.size();
        long long ans = 0, dmg = 0;
        vector<Monster> arr(N);
        for (int i = 0; i < N; i++) {
            dmg += damage[i];
            arr[i] = Monster(damage[i], ceil(health[i], power));
        }
        sort(arr.begin(), arr.end());
        for (const Monster &m : arr) {
            ans += dmg * m.turns;
            dmg -= m.dmg;
        }
        return ans;
    }
};
```

# Leetcode Biweekly Contest 139

## 

### Solution 1: 

```cpp

```

## 

### Solution 1: 

```cpp

```

# Leetcode Biweekly Contest 140

## 3301. Maximize the Total Height of Unique Towers

### Solution 1:  greedy, sorting

```cpp
class Solution {
public:
    const long long INF = 1e12;
    long long maximumTotalSum(vector<int>& A) {
        long long ans = 0;
        long long cur = INF;
        sort(A.rbegin(), A.rend());
        for (long long x : A) {
            cur = min(cur, x);
            if (cur <= 0) return -1;
            ans += cur;
            cur--;
        }
        return ans;
    }
};
```

## 3302. Find the Lexicographically Smallest Valid Sequence

### Solution 1:  suffix, lexicographically smallest, greedy

1. Track the earliest matching position from some character, that is matching the suffix of the string.
1. Then try matching the prefix, until you get to the point where

```cpp
class Solution {
public:
    vector<int> validSequence(string word1, string word2) {
        int N = word1.size(), M = word2.size();
        vector<int> suffix(N + 1);
        suffix.back() = M;
        for (int i = N - 1, j = M; i >= 0; i--) {
            if (j > 0 && word1[i] == word2[j - 1]) j--;
            suffix[i] = j;
        }
        vector<int> ans;
        bool flag = false;
        for (int i = 0, j = 0; i < N && j < M; i++) {
            if (word1[i] == word2[j]) {
                ans.push_back(i);
                j++;
            } else if (!flag && suffix[i + 1] - j - 1 <= 0) {
                ans.push_back(i);
                j++;
                flag = true;
            }
        }
        if (ans.size() < M) return {};
        return ans;
    }
};
```

## 3303. Find the Occurrence of First Almost Equal Substring

### Solution 1:  z algorithm, string matching, reverse string

```cpp
std::vector<int> z_algorithm(const std::string& s) {
    int n = s.length();
    std::vector<int> z(n, 0);
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
class Solution {
public:
    int minStartingIndex(string s, string pattern) {
        int N = s.size(), M = pattern.size();
        vector<int> temp1 = z_algorithm(pattern + '$' + s);
        reverse(pattern.begin(), pattern.end());
        reverse(s.begin(), s.end());
        vector<int> temp2 = z_algorithm(pattern + '$' + s);
        vector<int> z1(temp1.begin() + M + 1, temp1.end()), z2(temp2.begin() + M + 1, temp2.end());
        for (int i = 0; i <= N - M; i++) {
            if (z1[i] + z2[N - i - M] + 1 >= M) return i;
        }
        return -1;
    }
};
```

### Solution 2:  polynomial hash, rolling hash, string matching

1. This solution is hard to get to work, if you try some smaller prime integers for modulus it doesn't work for me.  But the larger the prime integer it helped. 
1. Non-deterministic solution, but it works for the test cases.

```cpp
class Solution {
public:
    const int INF = 1e9;
    const long long p = 31, MOD1 = pow(2, 43) - 1;
    int coefficient(char ch) {
        return ch - 'a' + 1;
    }
    int minStartingIndex(string s, string pattern) {
        int N = s.size(), M = pattern.size();
        unordered_map<long long, int> hashv;
        vector<long long> POW(M);
        POW[0] = 1;
        for (int i = 1; i < M; i++) {
            POW[i] = (POW[i - 1] * p) % MOD1;
        }
        long long hash = 0;
        for (int i = 0; i < N; i++) {
            hash = (p * hash + coefficient(s[i])) % MOD1;
            if (i >= M - 1) {
                if (!hashv.count(hash)) hashv[hash] = i - M + 1;
                hash = (hash - (POW[M - 1] * coefficient(s[i - M + 1])) % MOD1 + MOD1) % MOD1; 
            }
        }
        long long prefix_hash = 0;
        int ans = INF;
        vector<long long> suffix_hash(M + 1);
        suffix_hash[M] = 0;
        for (int i = M - 1; i >= 0; i--) {
            suffix_hash[i] = POW[M - i - 1] * coefficient(pattern[i]) % MOD1;
            suffix_hash[i] = (suffix_hash[i] + suffix_hash[i + 1]) % MOD1;
        }
        for (int i = 0; i < M; i++) {
            for (int j = 1; j <= 26; j++) {
                long long val = POW[M - i - 1] * j % MOD1;
                long long cand = (prefix_hash + val + suffix_hash[i + 1]) % MOD1;
                if (hashv.count(cand)) {
                    ans = min(ans, hashv[cand]);
                }
            }
            prefix_hash = (prefix_hash + POW[M - i - 1] * coefficient(pattern[i])) % MOD1;
        }
        return ans != INF ? ans : -1;
    }
};
```

# Leetcode Biweekly Contest 141

## 3316. Find Maximum Removals From Source String

### Solution 1:  recursive dynamic programming, 

```cpp
class Solution {
public:
    const int INF = 1e9;
    int N, M, T;
    vector<vector<int>> dp;
    string S, P;
    vector<int> targets;
    int dfs(int i, int j, int k) {
        if (i == N) {
            if (j == M) return 0;
            return -INF;
        }
        if (dp[i][j] != INF) return dp[i][j];
        int ans = -INF;
        if (k < T && targets[k] < i) k++;
        if (j < M && S[i] == P[j]) {
            ans = max(ans, dfs(i + 1, j + 1, k));
        } else {
            ans = max(ans, dfs(i + 1, j, k));
        }
        if (k < T && targets[k] == i) {
            ans = max(ans, dfs(i + 1, j, k) + 1);
        }
        return dp[i][j] = ans;
    }
    int maxRemovals(string source, string pattern, vector<int>& targetIndices) {
        N = source.size(), M = pattern.size(), T = targetIndices.size();
        S = source;
        P = pattern;
        targets = targetIndices;
        dp.assign(N, vector<int>(M + 1, INF));
        return dfs(0, 0, 0);
    }
};
```

## 3317. Find the Number of Possible Ways for an Event

### Solution 1:  dynamic programming, combinatorics, combinations, permutations, stirlings number of the second kind

```cpp
class Solution {
public:
    const long long MOD = 1e9 + 7;
    vector<vector<long long>> s;
    void stirlings(int n) {
        s[0][0] = 1;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= i; j++) {
                s[i][j] = (s[i - 1][j - 1] + j * s[i - 1][j]) % MOD;
            }
        }
    }
    long long inv(long long i) {
        return i <= 1 ? i : MOD - (long long)(MOD/i) * inv(MOD % i) % MOD;
    }

    vector<long long> fact, inv_fact;

    void factorials(int n) {
        fact.assign(n + 1, 1);
        inv_fact.assign(n + 1, 0);
        for (int i = 2; i <= n; i++) {
            fact[i] = (fact[i - 1] * i) % MOD;
        }
        inv_fact.end()[-1] = inv(fact.end()[-1]);
        for (int i = n - 1; i >= 0; i--) {
            inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % MOD;
        }
    }
    long long choose(int n, int r) {
        if (n < r) return 0;
        return (fact[n] * inv_fact[r] % MOD) * inv_fact[n - r] % MOD;
    }
    int numberOfWays(int n, int x, int y) {
        s.assign(n + 1, vector<long long>(n + 1, 0));
        stirlings(n);
        factorials(x);
        long long ans = 0, score_ways = 1, group_perms = 1;
        long long res;
        for (int i = 1; i <= min(x, n); i++) {
            score_ways = (score_ways * y) % MOD;
            group_perms = (group_perms * i) % MOD;
            long long group_ways = choose(x, i);
            long long partition_ways = s[n][i]; 
            res = score_ways * group_perms % MOD;
            res = res * group_ways % MOD;
            res = res * partition_ways % MOD;
            ans = (ans + res) % MOD;
        }
        return ans;
    }
};
```

# Leetcode Biweekly Contest 142

## 3331. Find Subtree Sizes After Changes

### Solution 1: tree, dfs, backtracking

```cpp
class Solution {
public:
    vector<vector<int>> adj;
    vector<int> sz, par;
    int last[26];
    int unicode(char ch) {
        return ch - 'a';
    }
    string S;
    int N;
    void dfs1(int u) {
        int cv = unicode(S[u]);
        int p = last[cv];
        if (p != -1) {
            par[u] = p;
        }
        last[cv] = u;
        for (int v : adj[u]) {
            dfs1(v);
        }
        last[cv] = p;
    }
    void dfs2(int u) {
        sz[u] = 1;
        for (int v : adj[u]) {
            dfs2(v);
            sz[u] += sz[v];
        }
    }
    vector<int> findSubtreeSizes(vector<int>& parent, string s) {
        S = s;
        par = vector(parent.begin(), parent.end());
        N = s.size();
        adj.assign(N, vector<int>());
        for (int i = 0; i < N; i++) {
            if (par[i] == -1) continue;
            adj[par[i]].emplace_back(i);
        }
        fill(last, last + 26, -1);
        dfs1(0);
        adj.assign(N, vector<int>());
        for (int i = 0; i < N; i++) {
            if (par[i] == -1) continue;
            adj[par[i]].emplace_back(i);
        }
        sz.assign(N, 0);
        dfs2(0);
        return sz;
    }
};
```

## 3332. Maximum Points Tourist Can Earn

### Solution 1:  dynamic programming, 2D dp

1. O(N^3) is fast enough for this problem, so that is straightforward, just trying every transition at every point in time.
1. Always have it calculated the maximum value for being at a node at a certain time, dp[time][city].

```cpp
class Solution {
public:
    int maxScore(int n, int k, vector<vector<int>>& stayScore, vector<vector<int>>& travelScore) {
        vector<vector<int>> dp(k + 1, vector<int>(n, 0));
        for (int i = 0; i < k; i++) {
            for (int src = 0; src < n; src++) {
                for (int dst = 0; dst < n; dst++) {
                    dp[i + 1][dst] = max(dp[i + 1][dst], dp[i][src] + travelScore[src][dst] + (src == dst ? stayScore[i][dst] : 0));
                }
            }
        }
        return *max_element(dp[k].begin(), dp[k].end());
    }
};
```

## 3333. Find the Original Typed String II

### Solution 1:  dynamic programming, run length encoding, counting

1. It is easy to count the total number of possible strings taking the product of the run length encoding lengths.
1. Now if k <= N, then we can just return the total number of possible strings.
1. If k > N, there is a problem because taking 1 of every character is not possible, and will only reach size N, which is not at least K.
1. So for this scenario we can calcualte with dynammic programming the number of ways to have a string of length between 0 to k - 1, and then these are the number of ways that are not going to achieve at least length k, so subtract these from final result.
1. Counting these can be done in O(k^2) time complexity, if you make sure you use a fixed sized window sum.  Write out the transitions to understand why this works exactly.

```cpp
class Solution {
public:
    int possibleStringCount(string word, int k) {
        int mod = 1e9 + 7;
        vector<int> rle;
        long long ans = 1;
        int cnt = 1;
        for (int i = 1; i < word.size(); i++) {
            if (word[i - 1] != word[i]) {
                ans = (ans * cnt) % mod;
                rle.emplace_back(cnt);
                cnt = 0;
            }
            cnt++;
        }
        ans = (ans * cnt) % mod;
        rle.emplace_back(cnt);
        int N = rle.size();
        if (k <= N) return ans;
        vector<int> dp(k, 0), ndp(k);
        dp[0] = 1;
        for (int i = 0; i < N; i++) {
            ndp.assign(k, 0);
            int wsum = 0;
            for (int j = i; j < k; j++) {
                ndp[j] = (ndp[j] + wsum) % mod;
                wsum = (wsum + dp[j]) % mod;
                if (j >= rle[i]) wsum = (wsum - dp[j - rle[i]] + mod) % mod;
            }
            swap(ndp, dp);
        }
        for (const int& x : dp) {
            ans = (ans - x + mod) % mod;
        }
        return ans;
    }
};
```

# Leetcode Biweekly Contest 143

## 

### Solution 1: 

```cpp

```

## 3346. Maximum Frequency of an Element After Performing Operations I

### Solution 1:  sorting, binary search, frequency map

```cpp
class Solution {
public:
    int maxFrequency(vector<int>& nums, int k, int numOperations) {
        unordered_map<int, int> freq;
        for (int x : nums) freq[x]++;
        sort(nums.begin(), nums.end());
        int N = nums.size(), ans = 0;
        for (int i = nums.front(); i <= nums.back(); ++i) {
            int l = lower_bound(nums.begin(), nums.end(), i - k) - nums.begin();
            int r = upper_bound(nums.begin(), nums.end(), i + k) - nums.begin();
            int cnt = freq[i], len = r - l;
            ans = max(ans, min(numOperations + cnt, len));
        }
        return ans;
    }
};
```

## 3347. Maximum Frequency of an Element After Performing Operations II

### Solution 1:  sorting, binary search, frequency map, sparse modes

```cpp
using int64 = long long;
class Solution {
public:
    int maxFrequency(vector<int>& nums, int k, int numOperations) {
        unordered_map<int, int> freq;
        sort(nums.begin(), nums.end());
        vector<int> modes;
        for (int x : nums) {
            modes.emplace_back(x - k);
            modes.emplace_back(x);
            modes.emplace_back(x + k);
            freq[x]++;
        }
        sort(modes.begin(), modes.end());
        modes.erase(unique(modes.begin(), modes.end()), modes.end());
        int ans = 0;
        for (int x : modes) {
            // [l, r)
            int l = lower_bound(nums.begin(), nums.end(), x - k) - nums.begin();
            int r = upper_bound(nums.begin(), nums.end(), static_cast<int64>(x) + k) - nums.begin();
            int len = r - l, cnt = freq[x];
            ans = max(ans, min(len, numOperations + cnt));
        }
        return ans;
    }
};
```

# Leetcode Biweekly Contest 144

## 3361. Shift Distance Between Two Strings

### Solution 1:  string, greedy

```cpp
#define int64 long long
class Solution {
private:
    vector<int> nxt, prv;
    int64 calcNext(int s, int e) {
        int64 ans = 0;
        while (s != e) {
            ans += nxt[s];
            s++;
            s %= 26;
        }
        return ans;
    }
    int64 calcPrev(int s, int e) {
        int64 ans = 0;
        while (s != e) {
            ans += prv[s];
            s = (s - 1 + 26) % 26;
        }
        return ans;
    }
    int decode(char ch) {
        return ch - 'a';
    }
public:
    int64 shiftDistance(string s, string t, vector<int>& nextCost, vector<int>& previousCost) {
        int N = s.size();
        nxt = nextCost, prv = previousCost;
        int64 ans = 0;
        for (int i = 0; i < N; i++) {
            int u = decode(s[i]), v = decode(t[i]);
            ans += min(calcNext(u, v), calcPrev(u, v));
        }
        return ans;
    }
};
```

## 3362. Zero Array Transformation III

### Solution 1:  max heap, greedy, sorting

```cpp
class Solution {
public:
    int maxRemoval(vector<int>& nums, vector<vector<int>>& queries) {
        priority_queue<int> maxheap;
        sort(queries.begin(), queries.end());
        int N = nums.size(), M = queries.size(), cur = 0;
        vector<int> end(N + 1, 0);
        for (int i = 0, j = 0; i < N; i++) {
            cur -= end[i];
            while (j < M && queries[j][0] <= i) {
                maxheap.push(queries[j][1]);
                j++;
            }
            while (cur < nums[i]) {
                if (maxheap.empty() || maxheap.top() < i) return -1;
                cur++;
                int e = maxheap.top();
                maxheap.pop();
                end[e + 1]++;
            }
        }
        return maxheap.size();
    }
};
```

## 3363. Find the Maximum Number of Fruits Collected

### Solution 1:  dynamic programming, grid, 

```cpp
class Solution {
private:
    vector<vector<int>> dp1, dp2;
    vector<vector<int>> grid;
    const int INF = 1e9;
    int N;
    bool inBounds(int r, int c) {
        return r >= 0 && r < N && c >= 0 && c < N;
    }
    int dfs1(int r, int c, int x) {
        if (!inBounds(r, c)) return -INF;
        if (x == N - 1) {
            return r == N - 1 && c == N - 1 ? 0 : -INF;
        }
        if (dp1[r][c] != -1) return dp1[r][c];
        int ans = -INF;
        for (int d = -1; d <= 1; d++) {
            ans = max(ans, dfs1(r + 1, c + d, x + 1) + grid[r][c]);
        }
        return dp1[r][c] = ans;
    }
    int dfs2(int r, int c, int x) {
        if (!inBounds(r, c)) return -INF;
        if (x == N - 1) {
            return r == N - 1 && c == N - 1 ? 0 : -INF;
        }
        if (dp2[r][c] != -1) return dp2[r][c];
        int ans = -INF;
        for (int d = -1; d <= 1; d++) {
            ans = max(ans, dfs2(r + d, c + 1, x + 1) + grid[r][c]);
        }
        return dp2[r][c] = ans;
    }
public:
    int maxCollectedFruits(vector<vector<int>>& fruits) {
        grid = fruits;
        N = fruits.size();
        int ans = 0;
        for (int i = 0; i < N; i++) {
            ans += grid[i][i];
            grid[i][i] = 0;
        }
        dp1.assign(N, vector<int>(N, -1));
        dp2.assign(N, vector<int>(N, -1));
        ans += dfs1(0, N - 1, 0);
        ans += dfs2(N - 1, 0, 0);
        return ans;
    }
};
```

```cpp
class Solution {
private:
    vector<vector<int>> dp1, dp2;
    vector<vector<int>> grid;
    const int INF = 1e9;
    int N;
    bool inBounds(int i) {
        return i >= 0 && i < N;
    }
public:
    int maxCollectedFruits(vector<vector<int>>& fruits) {
        grid = fruits;
        N = fruits.size();
        int ans = 0;
        for (int i = 0; i < N; i++) {
            ans += grid[i][i];
            grid[i][i] = 0;
        }
        dp1.assign(N, vector<int>(N, -INF));
        dp1[0][N - 1] = grid[0][N - 1];
        dp2.assign(N, vector<int>(N, -INF));
        dp2[N - 1][0] = grid[N - 1][0];
        for (int r = 1; r < N; r++) {
            for (int c = 0; c < N; c++) {
                for (int d = -1; d <= 1; d++) {
                    if (!inBounds(c + d)) continue;
                    dp1[r][c] = max(dp1[r][c], dp1[r - 1][c + d] + grid[r][c]);
                }
            }
        }
        for (int c = 1; c < N; c++) {
            for (int r = 0; r < N; r++) {
                for (int d = -1; d <= 1; d++) {
                    if (!inBounds(r + d)) continue;
                    dp2[r][c] = max(dp2[r][c], dp2[r + d][c - 1] + grid[r][c]);
                }
            }
        }
        ans = (ans + dp1[N - 1][N - 1] + dp2[N - 1][N - 1]);
        return ans;
    }
};
```

# Leetcode Biweekly Contest 145

## 3376. Minimum Time to Break Locks I

### Solution 1:  permutations, brute force, sorting

```cpp
class Solution {
private:
    const int INF = 1e9;
    int ceil(int x, int y) {
        return (x + y - 1) / y;
    }
public:
    int findMinimumTime(vector<int>& strength, int K) {
        sort(strength.begin(), strength.end());
        int ans = INF;
        do {
            int cost = 0;
            int X = 1;
            for (int s : strength) {
                cost += ceil(s, X);
                X += K;
            }
            ans = min(ans, cost);
        } while (next_permutation(strength.begin(), strength.end()));
        return ans;
    }
};
```

## 3377. Digit Operations to Make Two Integers Equal

### Solution 1:  weighted directed graph, shortest path, min heap, prime sieve

```cpp
const int MAXN = 1e5;
class Solution {
private:
    int decode(char ch) {
        return ch - '0';
    }
    char encode(int d) {
        return d + '0';
    }
    static bool precomputed;
    static bool primes[MAXN];
    void sieve(int n) {
        if (precomputed) return;
        fill(primes, primes + n, true);
        primes[0] = primes[1] = false;
        int p = 2;
        for (int p = 2; p * p <= n; p++) {
            if (primes[p]) {
                for (int i = p * p; i < n; i += p) {
                    primes[i] = false;;
                }
            }
        }
        precomputed = true;
    }
public:
    int minOperations(int n, int m) {
        sieve(MAXN);
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> minheap;
        minheap.emplace(n, n);
        unordered_map<int, int> memo;
        int cost;
        while (!minheap.empty()) {
            tie(cost, n) = minheap.top();
            minheap.pop();
            if (primes[n]) continue;
            if (n == m) return cost;
            string ni = to_string(n);
            for (int i = 0; i < ni.size(); i++) {
                int dig = decode(ni[i]);
                if (dig < 9) {
                    string node = ni;
                    node[i] = encode(dig + 1);
                    int u = stoi(node);
                    if (!memo.contains(u) || cost + u < memo[u]) {
                        minheap.emplace(cost + u, u);
                        memo[u] = cost + u;
                    }
                }
                if (dig > 0) {
                    string node = ni;
                    node[i] = encode(dig - 1);
                    int u = stoi(node);
                    if (!memo.contains(u) || cost + u < memo[u]) {
                        minheap.emplace(cost + u, u);
                        memo[u] = cost + u;
                    }
                }
            }
        }
        return -1;
    }
};

bool Solution::precomputed = false;
bool Solution::primes[MAXN];
```

## 3378. Count Connected Components in LCM Graph

### Solution 1:  disjoint sets, union find algorithm, factorization sieve, set, connected components, undirected graph

```cpp
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

    bool same(int i, int j) {
        i = find(i), j = find(j);
        if (i!=j) {
            if (size[j]>size[i]) {
                swap(i,j);
            }
            size[i]+=size[j];
            parents[j]=i;
            return false;
        }
        return true;
    }
};
class Solution {
private:
    static const int MAXN = 2e5 + 5;
    static bool precomputed;
    static vector<int> factors[MAXN];
    void precomputeFactors(int n) {
        if (precomputed) return;
        for (int i = 1; i < MAXN; i++) {
            for (int j = i; j < MAXN; j += i) {
                factors[j].emplace_back(i);
            }
        }
        precomputed = true;
    }
public:
    int countComponents(vector<int>& nums, int threshold) {
        precomputeFactors(MAXN);
        UnionFind dsu(threshold + 1);
        unordered_set<int> numbersSet(nums.begin(), nums.end());
        int ans = 0;
        for (int l = 2; l <= threshold; l++) {
            int last = -1;
            for (int f : factors[l]) {
                if (numbersSet.contains(f)) {
                    if (last != -1) {
                        dsu.same(last, f);
                    }
                    last = f;
                }
            }
        }
        unordered_set<int> connectedComponents;
        for (int x : nums) {
            if (x > threshold) {
                ans++;
            } else {
                connectedComponents.insert(dsu.find(x));
            }
        }
        ans += connectedComponents.size();
        return ans;
    }
};

bool Solution::precomputed = false;
vector<int> Solution::factors[MAXN];
```

# Leetcode Biweekly Contest 146

## 3393. Count Paths With the Given XOR Value

### Solution 1:  dynamic programming, bitwise operations, 

```cpp
#define int64 long long
class Solution {
public:
    int countPathsWithXorValue(vector<vector<int>>& grid, int k) {
        int R = grid.size(), C = grid[0].size();
        int MOD = 1e9 + 7;
        vector<vector<vector<int64>>> dp(R, vector<vector<int64>>(C, vector<int64>(32, 0)));
        dp[0][0][grid[0][0]] = 1;
        for (int r = 0; r < R; r++) {
            for (int c = 0; c < C; c++) {
                if (r == 0 && c == 0) continue;
                for (int i = 0; i < 32; i++) {
                    if (r > 0) dp[r][c][i] += dp[r - 1][c][i ^ grid[r][c]];
                    if (c > 0) dp[r][c][i] += dp[r][c-1][i ^ grid[r][c]];
                    dp[r][c][i] %= MOD;
                }
            }
        }
        return dp[R-1][C-1][k];
    }
};
```

## 3394. Check if Grid can be Cut into Sections

### Solution 1:  sorting, line sweep

```cpp
class Solution {
private:
    bool check(vector<pair<int, int>> &arr) {
        sort(arr.begin(), arr.end());
        int cnt = 0, ans = 0;
        for (int i = 0; i < arr.size(); i++) {
            cnt += arr[i].second;
            if (cnt == 0 && i + 1 < arr.size()) ans++;
        }
        return ans >= 2;
    }
public:
    bool checkValidCuts(int n, vector<vector<int>>& rectangles) {
        int N = rectangles.size();
        vector<pair<int, int>> X, Y;
        for (const vector<int> &rect : rectangles) {
            X.emplace_back(rect[0], 1);
            X.emplace_back(rect[2], -1);
            Y.emplace_back(rect[1], 1);
            Y.emplace_back(rect[3], -1);
        }
        return check(X) || check(Y);
    }
};
```

## 3395. Subsequences with a Unique Middle Mode I

### Solution 1:  combinatorics, math, counting, linear algebra, 

```cpp

#define int64 long long
const int64 MOD = 1e9 + 7;

class Solution {
private:
    int64 chooseTwo(int64 n) {
        return n * (n - 1) / 2;
    }
public:
    int subsequencesWithMiddleMode(vector<int>& nums) {
                int N = nums.size();
        int64 ans = 0;
        unordered_map<int, int> lfreq, rfreq;
        int64 rightSquaredSum = 0, leftSquaredSum = 0;
        for (int i = 0; i < N; i++) {
            if (i < 2) {
                lfreq[nums[i]]++;
            } else {
                rfreq[nums[i]]++;
            }
        }
        for (auto [_, f] : rfreq) {
            rightSquaredSum += f * f;
        }
        for (auto [_, f] : lfreq) {
            leftSquaredSum += f * f;
        }
        for (int i = 2; i < N - 2; i++) {
            rightSquaredSum -= rfreq[nums[i]] * rfreq[nums[i]];
            rfreq[nums[i]]--;
            int lf = lfreq[nums[i]], rf = rfreq[nums[i]];
            rightSquaredSum += rf * rf;
            // left: count(nums[i]]) = 1
            // right: count(nums[i]) = 1
            int64 leftTerm = lf * (i - lf);
            int64 rightTerm = rf * (N - i - 1 - rf);
            ans = (ans + leftTerm * rightTerm) % MOD;
            // left: count(nums[i]) = 2
            // right: count(nums[i]) < 2
            int64 coef = chooseTwo(lf);
            int64 val = chooseTwo(N - i - 1) - chooseTwo(rf);
            ans = (ans + coef * val) % MOD;
            // left: count(nums[i]) < 2
            // right: count(nums[i]) = 2
            coef = chooseTwo(rf);
            val = chooseTwo(i) - chooseTwo(lf);
            ans = (ans + coef * val) % MOD;
            // left: count(nums[i]) = 2
            // right: count(nums[i]) = 2
            int64 cand = chooseTwo(lf) * chooseTwo(rf) % MOD;
            ans = (ans + cand) % MOD;
            // side 1: count(nums[i]) = 1
            // side 2: count(nums[i]) = 0
            for (int j = 0; j < N; j++) {
                if (nums[i] == nums[j]) continue;
                int lfj = lfreq[nums[j]], rfj = rfreq[nums[j]];
                if (j < i) {
                    int64 rightSum = N - i - 1 - rf - rfj;
                    int64 squaredSumCorrected = rightSquaredSum - rf * rf - rfj * rfj;
                    int64 cand = ((rightSum * rightSum - squaredSumCorrected) / 2) % MOD;
                    int64 totalCand = lf * cand % MOD;
                    ans = (ans + totalCand) % MOD;
                } else {
                    int64 leftSum = i - lf - lfj;
                    int64 squaredSumCorrected = leftSquaredSum - lf * lf - lfj * lfj;
                    int64 cand = ((leftSum * leftSum - squaredSumCorrected) / 2) % MOD;
                    int64 totalCand = rf * cand % MOD;
                    ans = (ans + totalCand) % MOD;
                }
            }
            leftSquaredSum -= lf * lf;
            lfreq[nums[i]]++;
            leftSquaredSum += lfreq[nums[i]] * lfreq[nums[i]];
        }
        return ans;
    }
};
```

# Leetcode Biweekly Contest 147

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

# Leetcode Biweekly Contest 148

## 3424. Minimum Cost to Make Arrays Identical

### Solution 1:  sorting, greedy

```cpp
#define int64 long long

class Solution {
public:
    int64 minCost(vector<int>& arr, vector<int>& brr, int64 k) {
        int N = arr.size();
        int64 ans = 0;
        for (int i = 0; i < N; i++) {
            int64 cost = abs(arr[i] - brr[i]);
            ans += cost;
        }
        int64 cand = k;
        sort(arr.begin(), arr.end());
        sort(brr.begin(), brr.end());
        for (int i = 0; i < N; i++) {
            int64 cost = abs(arr[i] - brr[i]);
            cand += cost;
        }
        ans = min(ans, cand);
        return ans;
    }
};
```

## 3425. Longest Special Path

### Solution 1:  unweighted tree, dfs, monotonic stack, depth, fenwick tree

```cpp
const int INF = 1e9;
struct FenwickTree {
    vector<int> nodes;
    int neutral = 0;

    void init(int n) {
        nodes.assign(n + 1, neutral);
    }

    void update(int idx, int val) {
        while (idx < (int)nodes.size()) {
            nodes[idx] += val;
            idx += (idx & -idx);
        }
    }

    int query(int left, int right) {
        return right >= left ? query(right) - query(left - 1) : 0;
    }

    int query(int idx) {
        int result = neutral;
        while (idx > 0) {
            result += nodes[idx];
            idx -= (idx & -idx);
        }
        return result;
    }
};

class Solution {
private:
    FenwickTree ft;
    int N;
    vector<int> ans;
    vector<vector<pair<int, int>>> adj;
    stack<int> monoStack;
    vector<int> values;
    map<int, vector<int>> path;
    void dfs(int u, int p = -1, int d = 1) {
        int lastDepth = path.find(values[u]) == path.end() ? 0 : path[values[u]].back();
        if (monoStack.empty() || monoStack.top() < lastDepth) {
            monoStack.push(lastDepth);
        }
        int prevDepth = monoStack.top();
        int pathLength = ft.query(prevDepth + 1, d - 1);
        int numberNodes = d - prevDepth;
        if (pathLength >= ans[0]) {
            ans[1] = pathLength > ans[0] ? numberNodes : min(ans[1], numberNodes);
            ans[0] = pathLength;
        }
        path[values[u]].emplace_back(d);
        for (auto &[v, w] : adj[u]) {
            if (v == p) continue;
            ft.update(d, w);
            dfs(v, u, d + 1);
            ft.update(d, -w);
        }
        path[values[u]].pop_back();
        if (path[values[u]].empty()) path.erase(values[u]);
        else {
            if (monoStack.top() == lastDepth) monoStack.pop();
        }
    }
public:
    vector<int> longestSpecialPath(vector<vector<int>>& edges, vector<int>& nums) {
        N = nums.size();
        ft.init(N);
        values = nums;
        adj.assign(N, vector<pair<int, int>>());
        for (auto& edge : edges) {
            int u = edge[0], v = edge[1], w = edge[2];
            adj[u].emplace_back(v, w);
            adj[v].emplace_back(u, w);
        }
        ans.assign(2, 0);
        ans[1] = INF;
        monoStack.push(0);
        dfs(0);
        return ans;
    }
};
```

## 3426. Manhattan Distances of All Arrangements of Pieces

### Solution 1: 

```cpp

```

# Leetcode Biweekly Contest 149

## 3439. Reschedule Meetings for Maximum Free Time I

### Solution 1: intervals, fixed sized sliding window

```cpp
class Solution {
public:
    int maxFreeTime(int eventTime, int k, vector<int>& startTime, vector<int>& endTime) {
        int N = startTime.size();
        vector<int> gaps;
        int prv = 0;
        for (int i = 0; i < N; i++) {
            int gap = startTime[i] - prv;
            gaps.emplace_back(gap);
            prv = endTime[i];
        }
        gaps.emplace_back(eventTime - prv);
        int ans = 0, wgap = gaps[0];
        for (int i = 0; i < N; i++) {
            wgap += gaps[i + 1];
            if (i >= k) {
                wgap -= gaps[i - k];
            }
            ans = max(ans, wgap);
        }
        return ans;
    }
};
```

## 3440. Reschedule Meetings for Maximum Free Time II

### Solution 1:  greedy, sorting, intervals

```cpp
class Solution {
public:
    int maxFreeTime(int eventTime, vector<int>& startTime, vector<int>& endTime) {
        int N = startTime.size();
        vector<int> gaps;
        int prv = 0;
        for (int i = 0; i < N; i++) {
            int gap = startTime[i] - prv;
            gaps.emplace_back(gap);
            prv = endTime[i];
        }
        gaps.emplace_back(eventTime - prv);
        vector<int> sortedGaps = gaps;
        sort(sortedGaps.begin(), sortedGaps.end());
        int ans = 0;
        for (int i = 0; i < N; i++) {
            int gl = gaps[i], gr = gaps[i + 1], len = endTime[i] - startTime[i];
            ans = max(ans, gl + gr);
            int g = 0;
            if (sortedGaps.end()[-1] != max(gl, gr)) {
                g = sortedGaps.end()[-1];
            } else if (sortedGaps.end()[-2] != min(gl, gr)) {
                g = sortedGaps.end()[-2];
            } else {
                g = sortedGaps.end()[-3];
            }
            if (g < len) continue;
            ans = max(ans, gl + gr + len);
        }
        return ans;
    }
};
```

## 3441. Minimum Cost Good Caption

### Solution 1:  recursive dynamic programming, greedy, lexicographical sorting

1. The tricky part of this dynamic programming problem is the following
1. I think recursion makes this one a lot easier
1. The first observation is that you want to minimize the cost, for the current segment of characters independent of later.  And also it makes sense it will be one of the characters in the current segment you are looking at.  Which also means these segments will be length 3 to 5.
1. Then you have the recursive dfs calculate the best answer for the next state, and return that. 
1. The reason you need this is because you want to minimize total operations of current distance to the end. 
1. But also there is a tie breaker for when cost is the same and that is when the next character picked for the next segment is lexicographically smaller. 

```cpp
struct State {
    int cost, cnt;
    char ch;
    State() {}
    State(int cost, int cnt, char ch) : cost(cost), cnt(cnt), ch(ch) {}
};

const int INF = 1e9;
class Solution {
private:
    int N;
    string S;
    vector<State> dp;
    State dfs(int idx) {
        if (idx == N) return State(0, 0, 'a');
        if (dp[idx].cost != -1) return dp[idx];
        State res = State(INF, 0, 'a');
        vector<char> seen;
        string current = "";
        for (int i = idx; i < idx + 5; ++i) {
            if (i >= N) break;
            seen.emplace_back(S[i]);
            if (i < idx + 2) continue;
            int cost = INF;
            char character = 'a';
            int cnt = i - idx + 1;
            for (const char &ch : seen) {
                int newCost = 0;
                for (const char &c : seen) {
                    newCost += abs(c - ch);
                }
                if (newCost < cost || (newCost == cost && ch < character)) {
                    cost = newCost;
                    character = ch;
                }
            }
            State nextState = dfs(i + 1);
            string cand = string(cnt, character) + string(nextState.cnt, nextState.ch);
            if (cost + nextState.cost < res.cost || (cost + nextState.cost == res.cost && cand < current)) {
                res = State(cost + nextState.cost, cnt, character);
                current = cand;
            }
        }
        return dp[idx] = res;
    }
public:
    string minCostGoodCaption(string caption) {
        N = caption.size();
        if (N < 3) return "";
        string ans = "";
        int i = 0;
        S = caption;
        dp.assign(N, State(-1, -1, -1));
        while (i < N) {
            State nxtState = dfs(i);
            ans += string(nxtState.cnt, nxtState.ch);
            i += nxtState.cnt;
        }
        return ans;
    }   
};
```

