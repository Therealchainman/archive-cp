# Codeforces Educational Rounds

# Codeforces Round 170

## C. New Game

### Solution 1:  sliding window, counts, coordinate compression

```cpp
int N, K;
vector<int> arr;
 
void solve() {
    cin >> N >> K;
    arr.resize(N);
    vector<int> val, cnt;
    map<int, int> freq;
    for (int i = 0; i < N; i++) {
        cin >> arr[i];
        freq[arr[i]]++;
    }
    for (auto [key, value] : freq) {
        val.emplace_back(key);
        cnt.emplace_back(value);
    }
    int ans = 0, wcnt = 0;
    for (int l = 0, r = 0; r < val.size(); r++) {
        if (r > 0 && val[r] > val[r - 1] + 1) {
            wcnt = 0;
            l = r;
        }
        wcnt += cnt[r];
        if (r - l + 1 > K) {
            wcnt -= cnt[l];
            l++;
        }
        ans = max(ans, wcnt);
    }
    cout << ans << endl;
}
 
signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    int T;
    cin >> T;
    while (T--) {
        solve();
    }
    return 0;
}
```

## D. Attribute Checks

### Solution 1:  dynamic programming, suffix frequency

```cpp
const int MAXN = 5e3 + 5;
int N, M;
vector<int> arr, dp, ndp;
int str[MAXN], intel[MAXN];
 
void solve() {
    cin >> N >> M;
    arr.resize(N);
    memset(str, 0, sizeof(str));
    memset(intel, 0, sizeof(intel));
    for (int i = 0; i < N; i++) {
        cin >> arr[i];
        if (arr[i] < 0) {
            str[-arr[i]]++;
        } else if (arr[i] > 0) {
            intel[arr[i]]++;
        }
    }
    int ans = 0, cnt = 0;
    dp.assign(M + 1, 0);
    for (int i = 0; i < N; i++) {
        if (arr[i] < 0) {
            str[-arr[i]]--;
        } else if (arr[i] > 0) {
            intel[arr[i]]--;
        } else {
            cnt++;
            ndp.assign(M + 1, 0);
            for (int j = 0; j <= cnt; j++) {
                if (j > 0) {
                    ndp[j] = max(ndp[j], dp[j - 1] + str[j]);
                }
                ndp[j] = max(ndp[j], dp[j] + intel[cnt - j]);
                ans = max(ans, ndp[j]);
            }
            swap(dp, ndp);
        }
    }
 
    cout << ans << endl;
}
 
signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## E. Card Game

### Solution 1:  iterative dynamic programming

```cpp
const int MOD = 998244353;
int N, M;
vector<int> dp1, ndp1, dp, ndp;
 
void solve() {
    cin >> N >> M;
    dp1.assign(M + 1, 0);
    dp1[0] = 1;
    for (int i = 0; i < M; i++) {
        ndp1.resize(M + 1);
        for (int j = 0; j <= M; j++) {
            ndp1[j] = 0;
            if (j > 0) ndp1[j] += dp1[j - 1];
            if (j < M) ndp1[j] += dp1[j + 1];
            ndp1[j] %= MOD;
        }
        swap(dp1, ndp1);
    }
    dp.resize(M + 1);
    for (int i = 0; i <= M; i++) {
        dp[i] = dp1[i];
    }
    for (int i = 1; i < N; i++) {
        ndp.resize(M + 1);
        for (int j = 0; j <= M; j++) {
            ndp[j] = 0;
            for (int k = 0; k <= j; k++) {
                ndp[j - k] += dp[j] * dp1[k];
                ndp[j - k] %= MOD;
            }
        }
        swap(dp, ndp);
    }
    cout << dp[0] << endl;
}
 
signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

### Solution 2:  recursive dynamic programming

```cpp
const int MOD = 998244353;
int N, M;
vector<int> dp1, ndp1;
vector<vector<int>> dp;
 
int dfs(int i, int j) {
    if (i == 0) return dp1[j];
    if (dp[i][j] != -1) return dp[i][j];
    int ans = 0;
    for (int k = 0; j + k <= M; k++) {
        ans += dfs(i - 1, j + k) * dp1[k];
        ans %= MOD;
    }
    return dp[i][j] = ans;
}
 
void solve() {
    cin >> N >> M;
    dp1.assign(M + 1, 0);
    dp1[0] = 1;
    for (int i = 0; i < M; i++) {
        ndp1.resize(M + 1);
        for (int j = 0; j <= M; j++) {
            ndp1[j] = 0;
            if (j > 0) ndp1[j] += dp1[j - 1];
            if (j < M) ndp1[j] += dp1[j + 1];
            ndp1[j] %= MOD;
        }
        swap(dp1, ndp1);
    }
    dp.assign(N, vector<int>(M + 1, -1));
    cout << dfs(N - 1, 0) << endl;
}
 
signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## F. Choose Your Queries

You are given an array of n pairs of integers, where each pair is of the form [(x1, y1), (x2, y2), ... (xn, yn)], with the integers xi and yi in each pair satisfying 1 ≤ xi, yi ≤ N, and N ≤ 300,000. From each pair, you must select exactly one integer. Your goal is to select integers in such a way that minimizes the number of distinct integers that are chosen an odd number of times across all pairs.

You can also think of it as a directed graph, and analysis of indegree parity.

### Solution 1: 

```cpp

```

# Codeforces Round 171

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

# Codeforces Round 172

## D. Recommendations

### Solution 1:  sorting, segment overlapping, binary search, set

```cpp
struct Segment {
    int l, r, i;
    Segment() {}
    Segment(int l, int r, int i) : l(l), r(r), i(i) {}
    bool operator==(const Segment &other) const {
        return l == other.l && r == other.r;
    }
};

bool sortByLAscendingRDescending(const Segment &a, const Segment &b) {
    if (a.l != b.l) return a.l < b.l;  // Ascending order of l
    return a.r > b.r;                 // Descending order of r
}

bool sortByRDescendingLAscending(const Segment &a, const Segment &b) {
    if (a.l != b.l) return a.l > b.l; // Descending order of r
    return a.r < b.r;             // Ascending order of l
}

int N;
vector<int> ans;

void calcForward(vector<Segment> &segments) {
    set<int> endPoints;
    sort(segments.begin(), segments.end(), sortByLAscendingRDescending);
    for (const Segment &seg : segments) {
        auto it = endPoints.lower_bound(seg.r);
        if (it != endPoints.end()) {
            ans[seg.i] += (*it) - seg.r;
        }
        endPoints.insert(seg.r);
    }
}

void calcBackward(vector<Segment> &segments) {
    set<int> endPoints;
    sort(segments.begin(), segments.end(), sortByRDescendingLAscending);
    for (const Segment &seg : segments) {
        auto it = endPoints.upper_bound(seg.r);
        if (it != endPoints.begin()) {
            it--;
            ans[seg.i] += seg.r - (*it);
        }
        endPoints.insert(seg.r);
    }
}

void solve() {
    cin >> N;
    vector<Segment> segments;
    for (int i = 0; i < N; i++) {
        int l, r;
        cin >> l >> r;
        segments.emplace_back(l, r, i);
    }
    ans.assign(N, 0);
    calcForward(segments);
    for (Segment &seg : segments) swap(seg.l, seg.r);
    calcBackward(segments);
    for (int i = 1; i < N; i++) {
        if (segments[i] == segments[i - 1]) {
            ans[segments[i].i] = ans[segments[i - 1].i] = 0;
        }
    }
    for (int x : ans) {
        cout << x << endl;
    }
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    int T;
    cin >> T;
    while (T--) {
        solve();
    }
    return 0;
}

```

## 

### Solution 1: 

```cpp

```

# Codeforces Round 173

## C. Sums on Segments

### Solution 1: 

```cpp

```

## D. Problem about GCD

### Solution 1: 

```cpp

```

## E. Matrix Transformation

### Solution 1:  bit manipulation, directed graph, three color to detect cycle

1. The key observation is that when x = 11, it can always be split into 1, 2, and 8.  So you just care about the bits.
1. So solve the problem for each bit individually.
1. Understand the relationship between when A_ij = 0 and B_ij = 1, and A_ij = 1 and B_ij = 0, for one you must perform operation on ith row and other jth column.
1. Then you can construct a directed graph based on the fact that if B_ij = 0 or 1, that indicates if you have to perform operation to unset it you then need to perform another operation like an edge x -> y to return back to proper value.
1. So for each bit begin search from operation that must happen and it will determine if there is cycle.
1. Does this for each bit separately so you have 30 directed graphs to handle and determine if any has a cycle.

```cpp
enum Color {
    WHITE,
    GREY,
    BLACK
};

const int BITS = 32;
int R, C;
vector<vector<int>> A, B;
vector<vector<int>> adj;
vector<int> color;

bool isSet(int mask, int i) {
    return (mask >> i) & 1;
}

bool hasCycle(int u) {
    if (color[u] == BLACK) return false;
    if (color[u] == GREY) return true;
    color[u] = GREY;
    bool res = false;
    for (int v : adj[u]) {
        res |= hasCycle(v);
    }
    color[u] = BLACK;
    return res;
}

bool isDAG(int i) {
    adj.assign(R + C, vector<int>());
    vector<bool> hasRow(R, false);
    vector<bool> hasCol(C, false);
    color.assign(R + C, WHITE);
    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C; c++) {
            if (!isSet(A[r][c], i) && isSet(B[r][c], i)) {
                hasCol[c] = true;
            } else if (isSet(A[r][c], i) && !isSet(B[r][c], i)) {
                hasRow[r] = true;
            }
            if (isSet(B[r][c], i)) {
                adj[r].emplace_back(c + R);
            } else {
                adj[c + R].emplace_back(r);
            }
        }
    }
    for (int r = 0; r < R; r++) {
        if (hasRow[r]) {
            if (hasCycle(r)) return false; 
        }
    }
    for (int c = 0; c < C; c++) {
        if (hasCol[c]) {
            if (hasCycle(c + R)) return false;
        }
    }
    return true;
}

void solve() {
    cin >> R >> C;
    A.assign(R, vector<int>(C, 0));
    B.assign(R, vector<int>(C, 0));
    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C; c++) {
            cin >> A[r][c];
        }
    }
    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C; c++) {
            cin >> B[r][c];
        }
    }
    for (int i = 0; i < BITS; i++) {
        if (!isDAG(i)) {
            cout << "No" << endl;
            return;
        }
    }
    cout << "Yes" << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    int T;
    cin >> T;
    while (T--) {
        solve();
    }
    return 0;
}
```

# Codeforces Round 178

## 

### Solution 1: 

```cpp

```

# Codeforces Round 179

## Creating a Schedule

### Solution 1: greedy, sorting, two pointers

N / 2 largest and N / 2 smallest and do some alternating pattern to maximize the delta between each classroom for each group.

```cpp
int N, M;
vector<int> A;

void solve() {
    cin >> N >> M;
    A.assign(M, 0);
    vector<vector<int>> ans(N, vector<int>(6, 0));
    for (int i = 0; i < M; ++i) {
        cin >> A[i];
    }
    sort(A.begin(), A.end());
    for (int j = 0; j < 6; j++) {
        for (int i = 0, l = 0, r = M - 1; i < N; i++) {
            if (j % 2 == 0 && i < N / 2) {
                ans[i][j] = A[l++];
            } else if (j % 2 == 0) {
                ans[i][j] = A[r--];
            } else if (i < N / 2) {
                ans[i][j] = A[r--];
            } else {
                ans[i][j] = A[l++];
            }
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < 6; j++) {
            cout << ans[i][j] << " ";
        }
        cout << endl;
    }
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    int T;
    cin >> T;
    while (T--) {
        solve();
    }
    return 0;
}
```

## Changing the String

### Solution 1: greedy, set, two pointers

```cpp
int N, Q;
string S;
vector<vector<set<int>>> adj;

int decode(char ch) {
    return ch - 'a';
}

void solve() {
    cin >> N >> Q >> S;
    adj.assign(3, vector<set<int>>(3));
    for (int i = 0; i < Q; i++) {
        char s, t;
        cin >> s >> t;
        int u = decode(s), v = decode(t);
        adj[u][v].insert(i);
    }
    for (int i = 0; i < N; i++) {
        if (S[i] == 'b') {
            if (!adj[1][0].empty()) {
                adj[1][0].erase(adj[1][0].begin());
                S[i] = 'a';
                continue;
            }
            if (adj[1][2].empty()) continue;
            int u = *adj[1][2].begin();
            auto it = adj[2][0].lower_bound(u);
            if (it == adj[2][0].end()) continue;
            int v = *it;
            adj[1][2].erase(u);
            adj[2][0].erase(v);
            S[i] = 'a';
        } else if (S[i] == 'c') {
            if (!adj[2][0].empty()) {
                adj[2][0].erase(adj[2][0].begin());
                S[i] = 'a';
                continue;
            }
            if (adj[2][1].empty()) continue;
            int u = *adj[2][1].begin();
            auto it = adj[1][0].lower_bound(u);
            if (it == adj[1][0].end()) {
                S[i] = 'b';
                adj[2][1].erase(u);
                continue;
            }
            int v = *it;
            adj[2][1].erase(u);
            adj[1][0].erase(v);
            S[i] = 'a';
        }
    }
    cout << S << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    int T;
    cin >> T;
    while (T--) {
        solve();
    }
    return 0;
}

```