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

## 

### Solution 1: 

```cpp

```

## 

### Solution 1: 

```cpp

```

# Leetcode Biweekly Contest 144

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