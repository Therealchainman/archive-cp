# Practice

## 1400. Construct K Palindrome Strings

### Solution 1:  counting, parity, palindromes

```cpp
class Solution {
private:
    int encode(char ch) {
        return ch - 'a';
    }
public:
    bool canConstruct(string s, int k) {
        int N = s.size();
        vector<int> freq(26, 0);
        for (const char ch : s) {
            freq[encode(ch)]++;
        }
        int cnt = accumulate(freq.begin(), freq.end(), 0, [](int accum, int x) {
            return accum + (x & 1);
        });
        return k >= cnt && k <= N;
    }
};
```

## 2116. Check if a Parentheses String Can Be Valid

### Solution 1:  stack

```cpp
class Solution {
public:
    bool canBeValid(string s, string locked) {
        int N = s.size();
        if (N & 1) return false;
        stack<int> openBracket, unlocked;
        for (int i = 0; i < N; i++) {
            if (locked[i] == '0') {
                unlocked.push(i);
            } else if (s[i] == '(') {
                openBracket.push(i);
            } else {
                if (!openBracket.empty()) {
                    openBracket.pop();
                } else if (!unlocked.empty()) {
                    unlocked.pop();
                } else {
                    return false;
                }
            }
        }
        while (!openBracket.empty() && !unlocked.empty() && openBracket.top() < unlocked.top()) {
            openBracket.pop();
            unlocked.pop();
        }
        if (!openBracket.empty()) return false;
        return true;
    }
};
```

## 3223. Minimum Length of String After Operations

### Solution 1:  counting, parity, string

```cpp
class Solution {
private:
    int encode(char ch) {
        return ch - 'a';
    }
public:
    int minimumLength(string s) {
        vector<int> freq(26, 0);
        for (const char ch : s) {
            freq[encode(ch)]++;
        }
        int ans = 0;
        for (int x : freq) {
            if (!x) continue;
            ans++;
            if (x % 2 == 0) ans++;
        }
        return ans;
    }
};
```

## 2429. Minimize XOR

### Solution 1:  bit manipulation

```cpp
class Solution {
private:
    bool isSet(int mask, int i) {
        return (mask >> i) & 1;
    }
public:
    int minimizeXor(int num1, int num2) {
        int cnt = __builtin_popcount(num2);
        int x = 0;
        for (int i = 31; i >= 0 && cnt > 0; i--) {
            if (isSet(num1, i)) {
                x |= (1 << i);
                cnt--;
            }
        }
        for (int i = 0; i < 32 && cnt > 0; i++) {
            if (!isSet(x, i)) {
                x |= (1 << i);
                cnt--;
            }
        }
        return x;
    }
};
```

## 407. Trapping Rain Water II

### Solution 1:  greedy, min heap, grid

```cpp
struct Cell {
    int r, c, h;
    Cell(int r, int c, int h) : r(r), c(c), h(h) {}
    Cell(){}
    bool operator<(const Cell &other) const {
        return h > other.h;
    }
};

class Solution {
private:
    int R, C;
    vector<pair<int, int>> neighborhood(int r, int c) {
        return {{r - 1, c}, {r + 1, c}, {r, c - 1}, {r, c + 1}};
    }
    bool in_bounds(int r, int c) {
        return r >= 0 && r < R && c >= 0 && c < C;
    }
public:
    int trapRainWater(vector<vector<int>>& heightMap) {
        R = heightMap.size(), C = heightMap[0].size();
        vector<vector<bool>> vis(R, vector<bool>(C, false));
        priority_queue<Cell> minheap;
        for (int r = 0; r < R; r++) {
            minheap.emplace(r, 0, heightMap[r][0]);
            minheap.emplace(r, C - 1, heightMap[r][C - 1]);
            vis[r][0] = vis[r][C - 1] = true;
        }
        for (int c = 0; c < C; c++) {
            minheap.emplace(0, c, heightMap[0][c]);
            minheap.emplace(R - 1, c, heightMap[R - 1][c]);
            vis[0][c] = vis[R - 1][c] = true;
        }
        int ans = 0;
        while (!minheap.empty()) {
            auto [r, c, h] = minheap.top();
            minheap.pop();
            for (const auto &[nr, nc] : neighborhood(r, c)) {
                if (!in_bounds(nr, nc) || vis[nr][nc]) continue;
                vis[nr][nc] = true;
                ans += max(0, h - heightMap[nr][nc]);
                minheap.emplace(nr, nc, max(h, heightMap[nr][nc]));
            }
        }
        return ans;
    }
};
```

## 2017. Grid Game

### Solution 1:  prefix sums, greedy

1. Basically let index i represent when the first robot moves down, and to the bottom row, in that scenario there are only two options for robot 2 that are optimal. 
2. The first option is to move and collect all the values of the remaining element in the top row, which are only set to 0 for up to index i. And then the robot moves down.
3. The second option is to move down at the start and collect all the values of the bottom row, which are only set to 0 for everything after and including index i. 
4. draw it out and it becomes obvious, but the trick is the second robot will either move down right away or stay on the top row.  Cause, when it reaches the index that robot 1 moves down, there is no reason to move down anymore, so if you stayed up you didn't collect any points, but now you will. And to get points in bottom row you should move down right away.

```cpp
#define int64 long long
class Solution {
public:
    int64 gridGame(vector<vector<int>>& grid) {
        int N = grid[0].size();
        int64 firstRowSum = accumulate(grid[0].begin(), grid[0].end(), 0LL);
        int64 secondRowSum = 0, ans = 1e18;
        for (int i = 0; i < N; i++) {
            firstRowSum -= grid[0][i];
            ans = min(ans, max(firstRowSum, secondRowSum));
            secondRowSum += grid[1][i];
        }
        return ans;
    }
};
```

## 1267. Count Servers that Communicate

### Solution 1:  counting, grid, matrix

```cpp
class Solution {
public:
    int countServers(vector<vector<int>>& grid) {
        int R = grid.size(), C = grid[0].size(), ans = 0;
        vector<int> rowCount(R, 0), colCount(C, 0);
        for (int r = 0; r < R; ++r) {
            for (int c = 0; c < C; ++c) {
                rowCount[r] += grid[r][c];
                colCount[c] += grid[r][c];
            }
        }
        for (int r = 0; r < R; ++r) {
            for (int c = 0; c < C; ++c) {
                ans += grid[r][c];
                if (grid[r][c] && rowCount[r] == 1 && colCount[c] == 1) {
                    ans--;
                }
            }
        }
        return ans;
    }
};
```

## 2493. Divide Nodes Into the Maximum Number of Groups

### Solution 1:  bipartite, 2-coloring, undirected graph, connected components, bfs

```cpp
class Solution {
private:
    vector<int> colors;
    vector<bool> vis;
    vector<vector<int>> adj;
    bool bipartite_dfs(int source) {
        stack<int> stk;
        stk.push(source);
        colors[source] = 1;
        bool ans = true;
        while (!stk.empty()) {
            int u = stk.top();
            stk.pop();
            for (int v: adj[u]) {
                if (colors[v] == 0) {
                    colors[v] = 3 - colors[u];
                    stk.push(v);
                } else if (colors[u] == colors[v]) {
                    ans = false;
                }
            }
        }
        return ans;
    }
    void dfs(int u, vector<int>& comp) {
        if (vis[u]) return;
        vis[u] = true;
        comp.emplace_back(u);
        for (int v : adj[u]) {
            dfs(v, comp);
        }
    }
    int bfs(int src) {
        queue<int> dq;
        dq.emplace(src);
        vis[src] = true;
        int ans = 0;
        while (!dq.empty()) {
            int sz = dq.size();
            for (int i = 0; i < sz; i++) {
                int u = dq.front();
                dq.pop();
                for (int v : adj[u]) {
                    if (vis[v]) continue;
                    dq.emplace(v);
                    vis[v] = true;
                }
            }
            ++ans;
        }
        return ans;
    }
public:
    int magnificentSets(int n, vector<vector<int>>& edges) {
        adj.assign(n, vector<int>());
        for (const auto& edge : edges) {
            int u = edge[0], v = edge[1];
            --u, --v;
            adj[u].emplace_back(v);
            adj[v].emplace_back(u);
        }
        colors.assign(n, 0);
        for (int i = 0; i < n; i++) {
            if (colors[i]) continue;
            if (!bipartite_dfs(i)) return -1;
        }
        vis.assign(n, false);
        vector<vector<int>> components;
        for (int i = 0; i < n; i++) {
            if (vis[i]) continue;
            vector<int> comp;
            dfs(i, comp);
            components.emplace_back(comp);
        }
        int ans = 0;
        vis.assign(n, false);
        for (const vector<int>& comp : components) {
            int mx = 0;
            for (int u : comp) {
                for (int v : comp) {
                    vis[v] = false;
                }
                mx = max(mx, bfs(u));
            }
            ans += mx;
        }
        return ans;
    }
};
```

## 3105. Longest Strictly Increasing or Strictly Decreasing Subarray

### Solution 1:  loop

```cpp
class Solution {
public:
    int longestMonotonicSubarray(vector<int>& nums) {
        int ans = 0, cur = 0, prv = 0;
        for (int x : nums) {
            if (x <= prv) cur = 0;
            ++cur;
            prv = x;
            ans = max(ans, cur);
        }
        cur = 0;
        for (int x : nums) {
            if (x >= prv) cur = 0;
            ++cur;
            prv = x;
            ans = max(ans, cur);
        }
        return ans;
    }
};
```

## 1726. Tuple with Same Product

### Solution 1: counting, map, combinatorics

```cpp
class Solution {
private:
    int calc(int n) {
        return n * (n - 1);
    }
public:
    int tupleSameProduct(vector<int>& nums) {
        int N = nums.size();
        map<int, int> freq;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < i; ++j) {
                int x = nums[i] * nums[j];
                ++freq[x];
            }
        }
        int ans = accumulate(freq.begin(), freq.end(), 0, [&](int accum, pair<int, int> item) {
            return accum + calc(item.second);
        });
        ans *= 4;
        return ans;
    }
};
```

## 1352. Product of the Last K Numbers

### Solution 1:  prefix product

```cpp
class ProductOfNumbers {
private:
    vector<int> pre;
public:
    ProductOfNumbers() {
        pre.emplace_back(1);
    }
    
    void add(int num) {
        pre.emplace_back(pre.back() * num);
        if (!num) pre = {1};
    }
    
    int getProduct(int k) {
        if (k >= pre.size()) return 0;
        return pre.back() / pre.end()[-k - 1];
    }
};
```

## 1718. Construct the Lexicographically Largest Valid Sequence

### Solution 1: recursion, backtracking, early pruning, greedy

```cpp
class Solution {
private:
    vector<int> ans;
    vector<bool> used;
    int N;
    bool dfs(int i) {
        if (i == 2 * N - 1) {
            return true;
        }
        if (ans[i] != -1) return dfs(i + 1);
        for (int v = N; v > 1; v--) {
            if (used[v]) continue;
            if (i + v < 2 * N - 1 && ans[i] == -1 && ans[i + v] == -1) {
                ans[i] = ans[i + v] = v;
                used[v] = true;
                if (dfs(i + 1)) return true;
                used[v] = false;
                ans[i] = ans[i + v] = -1;
            }
        }
        if (!used[1]) {
            used[1] = true;
            ans[i] = 1;
            if (dfs(i + 1)) return true;
            used[1] = false;
            ans[i] = -1;
        }
        return false;
    }
public:
    vector<int> constructDistancedSequence(int n) {
        N = n;
        ans.assign(2 * N - 1, -1);
        used.assign(N + 1, false);
        dfs(0);
        return ans;
    }
};
```

## 1054. Distant Barcodes

### Solution 1: greedy, frequency array, sorting

```cpp
class Solution {
public:
    vector<int> rearrangeBarcodes(vector<int>& barcodes) {
        int N = barcodes.size();
        map<int, int> freq;
        for (int x : barcodes) {
            freq[x]++;
        }
        vector<pair<int, int>> events;
        for (const auto [k, v] : freq) {
            events.emplace_back(v, k);
        }
        sort(events.begin(), events.end());
        int cnt = 0, val = 0;
        for (int i = 0; i < N; i += 2) {
            if (!cnt) {
                tie(cnt, val) = events.back();
                events.pop_back();
            }
            barcodes[i] = val;
            cnt--;
        }
        for (int i = 1; i < N; i += 2) {
            if (!cnt) {
                tie(cnt, val) = events.back();
                events.pop_back();
            }
            barcodes[i] = val;
            cnt--;
        }
        return barcodes;
    }
};
```

## 1079. Letter Tile Possibilities

### Solution 1: recursion, backtracking, set, strings

```cpp
class Solution {
private:
    int N;
    set<string> ans;
    vector<bool> vis;
    string cur, tiles;
    void dfs() {
        if (ans.count(cur)) return;
        ans.insert(cur);
        for (int j = 0; j < N; j++) {
            if (vis[j]) continue;
            vis[j] = true;
            cur += tiles[j];
            dfs();
            cur.pop_back();
            vis[j] = false;
        }
    }
public:
    int numTilePossibilities(string S) {
        tiles = S;
        N = tiles.size();
        cur = "";
        vis.assign(N, false);
        dfs();
        return ans.size() - 1;
    }
};
```

## 1415. The k-th Lexicographical String of All Happy Strings of Length n

### Solution 1: recursion, backtracking, strings

```cpp
class Solution {
private:
    int N, K;
    string CHARS = "abc", cur;
    string dfs() {
        if (cur.size() == N) {
            K--;
            if (!K) return cur;
            return "";
        }
        for (char ch : CHARS) {
            if (!cur.empty() && cur.back() == ch) continue;
            cur += ch;
            string resp = dfs();
            if (!resp.empty()) return resp;
            cur.pop_back();
        }
        return "";
    }
public:
    string getHappyString(int n, int k) {
        N = n;
        K = k;
        cur = "";
        return dfs();
    }
};
```

## 1261. Find Elements in a Contaminated Binary Tree

### Solution 1: set, dfs, binary tree, recursion

```cpp
class FindElements {
private:
    set<int> seen;
    void dfs(TreeNode* root, int x) {
        if (root == nullptr) return;
        seen.insert(x);
        dfs(root -> left, 2 * x + 1);
        dfs(root -> right, 2 * x + 2);
    }
public:
    FindElements(TreeNode* root) {
        dfs(root, 0);
    }
    
    bool find(int target) {
        return seen.count(target);
    }
};
```

## 1028. Recover a Tree From Preorder Traversal

### Solution 1:  recursion, preorder tree traversal, dfs, depth

```cpp
class Solution {
private:
    string S;
    int idx;
    int decode(char ch) {
        return ch - '0';
    }
    TreeNode* dfs(int depth) {
        if (idx >= S.size()) return nullptr;
        int dashes = 0;
        while (S[idx] == '-') {
            dashes++;
            idx++;
        }
        if (dashes < depth) {
            idx -= dashes;
            return nullptr;
        }
        int val = 0;
        while (idx < S.size() && isdigit(S[idx])) {
            val = val * 10 + decode(S[idx++]);
        }
        TreeNode* node = new TreeNode(val);
        node -> left = dfs(depth + 1);
        node -> right = dfs(depth + 1);
        return node;
    }
public:
    TreeNode* recoverFromPreorder(string traversal) {
        S = traversal;
        idx = 0;
        return dfs(0);
    }
};
```

## 889. Construct Binary Tree from Preorder and Postorder Traversal

### Solution 1: recursion, two pointers, preorder, postorder, tree traversal, binary tree

```cpp
class Solution {
private:
    int i, j, N;
    vector<int> preorder, postorder;
    TreeNode* dfs() {
        if (i == N) return nullptr;
        TreeNode* root = new TreeNode(preorder[i++]);
        if (root -> val == postorder[j]) {
            j++;
            return root;
        }
        root -> left = dfs();
        if (postorder[j] == root -> val) {
            j++;
            return root;
        }
        root -> right = dfs();
        if (postorder[j] == root -> val) j++;
        return root;
    }
public:
    TreeNode* constructFromPrePost(vector<int>& preorder, vector<int>& postorder) {
        i = 0;
        j = 0;
        N = preorder.size();
        this -> preorder = preorder;
        this -> postorder = postorder;
        return dfs();
    }
};
```

## 873. Length of Longest Fibonacci Subsequence

### Solution 1: dynamic programming, dictionary, fibonacci, subsequence

```py
class Solution:
    def lenLongestFibSubseq(self, arr: List[int]) -> int:
        N = len(arr)
        ans = 0
        dp = [[0] * N for _ in range(N)]
        val_to_idx = {num: i for i, num in enumerate(arr)}
        for i in range(N):
            for j in range(i):
                delta = arr[i] - arr[j]
                k = val_to_idx.get(delta, math.inf)
                dp[j][i] = (
                    dp[k][j] + 1 
                    if delta < arr[j] and k < j 
                    else 2
                )
                ans = max(ans, dp[j][i])
        return ans if ans > 2 else 0
```

## 1123. Lowest Common Ancestor of Deepest Leaves

### Solution 1: recursion, depth, binary tree

```cpp
class Solution {
private:
    pair<TreeNode*, int> dfs(TreeNode* root) {
        if (!root) return {nullptr, 0};
        auto left = dfs(root -> left);
        auto right = dfs(root -> right);
        if (left.second > right.second) {
            return {left.first, left.second + 1};
        }
        if (left.second < right.second) {
            return {right.first, right.second + 1};
        }
        return {root, left.second + 1};
    }
public:
    TreeNode* lcaDeepestLeaves(TreeNode* root) {
        return dfs(root).first;
    }
};
```

## 781. Rabbits in Forest

### Solution 1: frequency array, ceil division

```cpp
class Solution {
private:
    int ceil(int x, int y) {
        return (x + y - 1) / y;
    }
public:
    int numRabbits(vector<int>& answers) {
        int MAXN = *max_element(answers.begin(), answers.end());
        vector<int> freq(MAXN + 1, 0);
        for (int x : answers) {
            freq[x]++;
        }
        int ans = 0;
        for (int i = 0; i <= MAXN; i++) {
            ans += (i + 1) * ceil(freq[i], i + 1);
        }
        return ans;
    }
};
```

## 1399. Count Largest Group

### Solution 1: map for frequency, digit sum, accumulate

```cpp
class Solution {
private:
    int calc(int n) {
        int res = 0;
        while (n > 0) {
            res += n % 10;
            n /= 10;
        }
        return res;
    }
public:
    int countLargestGroup(int n) {
        int maxFreq = 0;
        map<int, int> freq;
        for (int i = 1; i <= n; i++) {
            int digitSum = calc(i);
            freq[digitSum]++;
            maxFreq = max(maxFreq, freq[digitSum]);
        }
        int ans = accumulate(freq.begin(), freq.end(), 0, [&](int accum, const pair<int, int> &elem) {
            return accum + (elem.second == maxFreq);
        });
        return ans;
    }
};
```

## 2845. Count of Interesting Subarrays

### Solution 1: modular arithmetic, prefix sum, map

```cpp
using int64 = long long;
class Solution {
public:
    int64 countInterestingSubarrays(vector<int>& nums, int modulo, int k) {
        int64 ans = 0;
        int N = nums.size(), psum = 0;
        map<int, int> freq;
        freq[0] = 1;
        for (int x : nums) {
            if (x % modulo == k) psum++;
            int val = psum % modulo;
            ans += freq[(val - k + modulo) % modulo];
            freq[val]++;
        }
        return ans;
    }
};
```

## 2071. Maximum Number of Tasks You Can Assign

### Solution 1:  greedy, binary search, multiset, sorting

1. The key is that you try to complete k tasks.
1. Always best to try to complete the k easiest tasks, using the strongest workers.
1. Iterate from largest to smallest task in those k easiest tasks, and if the stronger worker available can do it do that.
1. Otherwise, take the weakest worker that can do it with a pill.
1. If it is possible under that logic you are good and binary search works because if you can do it for k workers, you can do it for k - 1 workers.

```cpp
class Solution {
private:
    vector<int> A;
    int K, S;
    bool possible(int target, multiset<int> pool) {
        int cnt = 0;
        for (int i = target - 1; i >= 0; i--) {
            if (pool.empty()) return false;
            auto it = prev(pool.end());
            if (*it >= A[i]) {
                pool.erase(it);
            } else {
                auto it = pool.lower_bound(A[i] - S);
                if (it == pool.end()) return false;
                pool.erase(it);
                cnt++;

            }
            if (cnt > K) return false;
        }
        return true;
    }
public:
    int maxTaskAssign(vector<int>& tasks, vector<int>& B, int pills, int strength) {
        int N = tasks.size();
        K = pills, S = strength;
        sort(tasks.begin(), tasks.end());
        A = tasks;
        multiset<int> workers(B.begin(), B.end());
        int lo = 0, hi = N;
        while (lo < hi) {
            int mid = lo + (hi - lo + 1) / 2;
            if (possible(mid, workers)) lo = mid;
            else hi = mid - 1;
        }
        return lo;
    }
};
```

## 1128. Number of Equivalent Domino Pairs

### Solution 1: map, counter, combinatorics

```cpp
class Solution {
private:
    int calc(int n) {
        return n * (n - 1) / 2;
    }
public:
    int numEquivDominoPairs(vector<vector<int>>& dominoes) {
        map<int, int> freq;
        int ans = 0;
        for (vector<int> &dom : dominoes) {
            sort(dom.begin(), dom.end());
            int val = dom[0] * 10 + dom[1];
            freq[val]++;
        }
        for (const auto &[k, v] : freq) {
            ans += calc(v);
        }
        return ans;
    }
};
```

## 1931. Painting a Grid With Three Different Colors

### Solution 1: dfs, base encoding, base 3 representation, undirected graph, dynamic programming, counting

1. The key is to represent the grid as a number in base 3, where each digit represents the color of a cell.
1. The number of states is 3 * 2^(m - 1), where m is the number of rows.
1. The adjacency list is built by checking if two states are compatible (i.e., they can be painted without conflicts).
1. The dp array is used to count the number of ways to paint the grid, where dp[i][j] represents the number of ways to paint the first i rows with state j.


```cpp
const int MOD = 1e9 + 7;
class Solution {
private:
    int M;
    vector<int> states;
    void dfs(int idx, int code, int prv) {
        if (idx == M) {
            states.emplace_back(code);
            return;
        }
        for (int i = 0; i < 3; i++) {
            if (i == prv) continue;
            dfs(idx + 1, 3 * code + i, i);
        }
    }
    bool isCompatible(int x, int y) {
        for (int i = 0; i < M; i++) {
            if (x % 3 == y % 3) return false;
            x /= 3;
            y /= 3;
        }
        return true;
    }
public:
    int colorTheGrid(int m, int n) {
        M = m;
        dfs(0, 0, -1);
        int numStates = states.size();
        vector<vector<int>> adj(numStates, vector<int>());
        for (int i = 0; i < numStates; i++) {
            for (int j = 0; j < i; j++) {
                int u = states[i], v = states[j];
                if (!isCompatible(u, v)) continue;
                adj[i].emplace_back(j);
                adj[j].emplace_back(i);
            }
        }
        vector<vector<int>> dp(n, vector<int>(numStates, 0));
        for (int i = 0; i < numStates; i++) {
            dp[0][i] = 1; // start with all of the states, in first column
        }
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < numStates; j++) {
                for (int k : adj[j]) {
                    dp[i][j] = (dp[i][j] + dp[i - 1][k]) % MOD;
                }
            }
        }
        int ans = 0;
        for (int i = 0; i < numStates; i++) {
            ans = (ans + dp[n - 1][i]) % MOD;
        }
        return ans;
    }
};
```

## 1857. Largest Color Value in a Directed Graph

### Solution 1: directed graph, topological sort, dynamic programming, counting

After doing this for all 26 colors, v's table reflects the best possible color-counts on any path that reaches v via node u.

Correctness follows because we’re doing a DP on a DAG in topological order. By the time we “visit” a node, we’ve already computed the optimum counts for every path into it.

Finds the maximum number of identically-colored nodes on any root-to-leaf path in a directed graph, or detects a cycle.

```cpp
class Solution {
private:
    int decode(char ch) {
        return ch - 'a';
    }
public:
    int largestPathValue(string colors, vector<vector<int>>& edges) {
        int N = colors.size(), cnt = 0, ans = 0;
        vector<vector<int>> adj(N, vector<int>()), dp(N, vector<int>(26, 0));
        vector<int> ind(N, 0);
        for (const auto &edge : edges) {
            int u = edge[0], v = edge[1];
            adj[u].emplace_back(v);
            ind[v]++;
        }
        queue<int> q;
        for (int i = 0; i < N; i++) {
            if (!ind[i]) q.emplace(i);
        }
        while (!q.empty()) {
            int u = q.front();
            dp[u][decode(colors[u])]++;
            ans = max(ans, dp[u][decode(colors[u])]);
            cnt++;
            q.pop();
            for (int v : adj[u]) {
                ind[v]--;
                for (int i = 0; i < 26; i++) {
                    dp[v][i] = max(dp[v][i], dp[u][i]);
                }
                if (ind[v]) continue;
                q.emplace(v);
            }
        }
        return cnt == N ? ans : -1;
    }
};
```

## 2929. Distribute Candies Among Children II

### Solution 1: dynamic programming, prefix sum, counting, knapsack related, bounded weak integer composition problem

What is the name of this problem? 
This is known as the  bounded weak integer composition problem, — not to be confused with integer partitioning. While partitions treat order as irrelevant, compositions are ordered sequences of integers that sum to a target number.

Specifically, we are interested in counting the number of bounded weak compositions of a number n into exactly 3 parts, where each part is an integer between 0 and L. 

Composition: An ordered sequence of integers summing to n.
Weak composition: Allows zero as a valid part (i.e., parts are in {0, 1, …, L}).
Bounded: Each part has a maximum value of L.
Fixed number of parts: We're composing n into exactly 3 parts, which ensures the solution space is finite, even though zeros are allowed.

Without the restriction on the number of parts, weak compositions into unbounded-length sequences would be infinite. So fixing the number of parts (in this case, 3) is crucial to make the problem well-defined and tractable.

I'm solving this using a counting dynamic programming (DP) approach, which tracks how many ways I can reach a sum n given constraints on how many parts and how large each part can be.

dp[i][j] = number of ways to compose sum j using i parts

With constraints ensuring each part is in [0, L].

To optimize the computation, I'm using a sliding window prefix sum technique — which reduces the inner loop's complexity by keeping track of rolling sums rather than recalculating the sum of L + 1 terms each time. This helps bring down the runtime to O(3n), or more generally O(kn) for k parts.

```cpp
using int64 = long long;
class Solution {
private:
    int64 rangeSum(const vector<int64> &psum, int l, int r) {
        int64 ans = psum[r];
        if (l > 0) ans -= psum[l - 1];
        return ans;
    }
public:
    int64 distributeCandies(int n, int limit) {
        vector<int64> dp(n + 1, 0), psum(n + 1, 0);
        dp[0] = 1;
        for (int k = 0; k < 3; k++) {
            for (int i = 0; i <= n; i++) {
                psum[i] = dp[i];
                if (i > 0) psum[i] += psum[i - 1];
            }
            for (int i = 0; i <= n; i++) {
                dp[i] = rangeSum(psum, i - limit, i);
            }
        }
        return dp[n];
    }
};
```

## 1298. Maximum Candies You Can Get from Boxes

### Solution 1:  bfs, queue, visited

```cpp
class Solution {
public:
    int maxCandies(vector<int>& status, vector<int>& candies, vector<vector<int>>& keys, vector<vector<int>>& containedBoxes, vector<int>& initialBoxes) {
        int N = status.size(), ans = 0;
        queue<int> q;
        vector<bool> boxes(N, false);
        for (int x : initialBoxes) {
            if (status[x]) q.emplace(x);
            else boxes[x] = true;
        }
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            ans += candies[u];
            for (int v : keys[u]) {
                status[v] = 1; // box can be opend now
                if (boxes[v]) { // box could not be opened earlier
                    boxes[v] = false;
                    q.emplace(v);
                }
            }
            for (int v : containedBoxes[u]) {
                if (status[v]) q.emplace(v); // box can be opened
                else boxes[v] = true; // box cannot be opened yet
            }
        }
        return ans;
    }
};
```

## 3170. Lexicographically Minimum String After Removing Stars

### Solution 1:  stack, greedy, simulation

```cpp
class Solution {
public:
    string clearStars(string s) {
        int N = s.size();
        vector<vector<int>> last(26, vector<int>());
        vector<bool> remains(N, true);
        for (int i = 0; i < N; i++) {
            if (s[i] == '*') {
                remains[i] = false;
                for (int j = 0; j < 26; j++) {
                    if (!last[j].empty()) {
                        remains[last[j].back()] = false;
                        last[j].pop_back();
                        break;
                    }
                }
            } else {
                last[s[i] - 'a'].emplace_back(i);
            }
        }
        string ans;
        for (int i = 0; i < N; i++) {
            if (!remains[i]) continue;
            ans += s[i];
        }
        return ans;
    }
};
```

## 2016. Maximum Difference Between Increasing Elements

### Solution 1:  prefix minimum, greedy

```cpp
const int INF = numeric_limits<int>::max();
class Solution {
public:
    int maximumDifference(vector<int>& nums) {
        int N = nums.size();
        int pmin = INF, ans = -1;
        for (int x : nums) {
            if (x > pmin) ans = max(ans, x - pmin);
            pmin = min(pmin, x);
        }
        return ans;
    }
};
```

## 2014. Longest Subsequence Repeated k Times

### Solution 1: brute force, permutations, two pointer, frequency array, bfs queue

```cpp
class Solution {
private:
    int N, K;
    string S;
    bool repeatsKTimes(string s) {
        int cntMatches = 0, M = s.size();
        for (int i = 0, pos = 0; i < N; ++i) {
            if (s[pos] == S[i]) {
                if (++pos == M) cntMatches++;
                pos %= M;
            }
        }
        return cntMatches >= K;
    }
public:
    string longestSubsequenceRepeatedK(string s, int k) {
        S = s, N = s.size(), K = k;
        vector<char> candidates;
        vector<int> freq(26, 0);
        for (char ch : s) freq[ch - 'a']++;
        for (int i = 25; i >= 0; --i) {
            if (freq[i] < k) continue;
            candidates.emplace_back(i + 'a');
        }
        queue<string> q;
        for (char ch : candidates) {
            q.emplace(string(1, ch));
        }
        string ans = "";
        while (!q.empty()) {
            string seq = q.front();
            q.pop();
            if (seq.size() > ans.size()) ans = seq;
            for (char ch : candidates) {
                string candSeq = seq + ch;
                if (repeatsKTimes(candSeq)) {
                    q.emplace(candSeq);
                }
            }
        }
        return ans;
    }
};
```

## 2040. Kth Smallest Product of Two Sorted Arrays

### Solution 1:  greedy binary search, product, nested binary search

1. Handle the separate cases of when x is positive or negative.

```cpp
using int64 = int64_t;
int64 INF = 1e10 + 5;
class Solution {
private:
    vector<int> A, B, rB;
    int helper(int64 target, int64 x, const vector<int> &arr) {
        int N = arr.size();
        int lo = 0, hi = N;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            int64 prod = x * arr[mid];
            if (prod <= target) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }
    int64 calc(int64 target) {
        int64 ans = 0;
        int N = A.size(), M = B.size();
        for (int x : A) {
            if (x < 0) {
                ans += helper(target, x, rB);
            } else {
                ans += helper(target, x, B);
            }
        }
        return ans;
    }
public:
    int64 kthSmallestProduct(vector<int>& nums1, vector<int>& nums2, int64 k) {
        A = vector<int>(nums1.begin(), nums1.end());
        B = vector<int>(nums2.begin(), nums2.end());
        rB = vector<int>(nums2.rbegin(), nums2.rend());
        int64 lo = -INF, hi = INF;
        while (lo < hi) {
            int64 mid = lo + (hi - lo) / 2;
            if (calc(mid) < k) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }
};
```

## 594. Longest Harmonious Subsequence

### Solution 1: sorting, two pointers

```cpp
class Solution {
public:
    int findLHS(vector<int>& nums) {
        int N = nums.size(), ans = 0;
        sort(nums.begin(), nums.end());
        for (int i = 0, j = 0; i < N; ++i) {
            while (nums[i] - nums[j] > 1) ++j;
            if (nums[i] - nums[j] == 1) ans = max(ans, i - j + 1);
        }
        return ans;
    }
};
```

## Find Lucky Integer in an Array

### Solution 1: frequency, reverse iteration

```cpp
const int MAXN = 505;
class Solution {
private:
    int freq[MAXN];
public:
    int findLucky(vector<int>& arr) {
        memset(freq, 0, sizeof(freq));
        for (int x : arr) freq[x]++;
        for (int i = MAXN - 1; i > 0; --i) {
            if (freq[i] == i) return i;
        }
        return -1;
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

##

### Solution 1: 

```cpp

```