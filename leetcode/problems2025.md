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