# Starters 150

## Equal Pairs (Hard)

### Solution 1:  counter, hash table, combinatorics

1. Greedy solution, where you want to assign all the 0s to be 1s
2. You also need to track the frequency and update the total sum of pairs based on that. 
3. And for the element with maximum frequency remove it from answer and add it with the count of 0s included

```cpp
int N;

int choose(int n) {
    return n * (n - 1) / 2;
}

void solve() {
    cin >> N;
    map<int, int> freq;
    int mx = 0, ans = 0;
    for (int i = N - 1; i >= 0; i--) {
        int x, v;
        cin >> x >> v;
        ans += freq[v];
        freq[v]++;
        mx = max(mx, freq[v]);
        int res = ans - choose(mx) + choose(i + mx);
        cout << res << " ";
    }
    cout << endl;
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

## Replacing Game

### Solution 1:  array, pointers

1. The idea is to iterate through array from left to right, and place into separate array anytime you have a chunk that is greater than size K
2. Then you iterate from right to left, until it reaches the last K sized chunk, and it puts this back in array A
3. Now you can construct it from this method. Also need to check if there is no K sized chunk, that means it will be impossible

```cpp
int N, K;
string S, T;

void solve() {
    cin >> N >> K >> S >> T;
    if (S == T) {
        cout << 0 << endl;
        return;
    }
    vector<pair<int, char>> A, B;
    int streak = 0;
    bool possible = false;
    int last = 0;
    for (int i = 0; i < N; i++) {
        if (i > 0 && T[i] != T[i - 1]) streak = 0;
        streak++;
        if (streak >= K) {
            possible = true;
            B.emplace_back(i - K + 1, T[i]);
            last = i;
        } else if (i + K <= N) {
            A.emplace_back(i, T[i]);
        }
    }
    if (!possible) {
        cout << -1 << endl;
        return;
    }
    streak = 0;
    for (int i = N - 1; i > last; i--) {
        if (i + 1 < N && T[i] != T[i + 1]) streak = 0;
        streak++;
        A.emplace_back(i - K + 1, T[i]);
    }
    A.insert(A.end(), B.begin(), B.end());
    int sz = A.size();
    cout << sz << endl;
    for (auto &[i, c] : A) {
        cout << i + 1 << " " << c << endl;
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

# Starters 151

## Ball Game

### Solution 1:  monotonic stack, sort

```cpp
int N;
vector<int> A, B; // A[i] = position of ith ball, B[i] = speed of ith ball
vector<pair<int, int>> ball; // (position, speed)

// i < j, will ball(j) collide with ball(i)?
bool collide(int i, int j) {
    if (ball[j].second <= ball[i].second) return false;
    long double t = (long double)(ball[j].first - ball[i].first) / (ball[j].second - ball[i].second);
    long double pos = ball[i].first - (long double)ball[i].second * t;
    return pos > 0;
}

void solve() {
    cin >> N;
    A.resize(N);
    B.resize(N);
    ball.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    for (int i = 0; i < N; i++) {
        cin >> B[i];
    }
    for (int i = 0; i < N; i++) {
        ball[i] = {A[i], B[i]};
    }
    sort(ball.begin(), ball.end());
    stack<int> stk;
    for (int i = 0; i < N; i++) {
        while (!stk.empty() && collide(stk.top(), i)) {
            stk.pop();
        }
        stk.push(i);
    }
    cout << stk.size() << endl;
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

## Sequence Search

### Solution 1:  binary search, count odd and even, find kth smallest

```cpp
const int INF = 1e18;
int A, B, K;

int floor(int x, int y) {
    return x / y;
}
// count odd terms
// count even terms
// binary search the answer, based ont he count being less than or equal to K

void solve() {
    cin >> A >> B >> K;
    int lo = 0, hi = INF;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        int cnt = floor(mid, B) + 1;
        if (mid >= A) cnt += floor(mid - A, B) + 1;
        if (cnt < K) lo = mid + 1;
        else hi = mid;
    }
    cout << lo << endl;
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

## Shooting (Hard)

### Solution 1:  sort, manhattan trick, binary search, prefix sum

```cpp
int R, C;
vector<vector<int>> grid;
vector<int> psumx[2], psumy[2], arrx[2], arry[2];

int calc(const vector<int>& arr, const vector<int>& psum, int v) {
    int N = arr.size();
    int lo = 0, hi = N;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (arr[mid] <= v) lo = mid + 1;
        else hi = mid;
    }
    lo--;
    int lsum = 0;
    if (lo >= 0) lsum += v * (lo + 1) - psum[lo];
    int rsum = -v * (N - 1 - lo);
    if (N - 1 >= 0) rsum += psum[N - 1];
    if (lo >= 0) rsum -= psum[lo];
    int res = lsum + rsum;
    return res;
}

void solve() {
    cin >> R >> C;
    grid.assign(R, vector<int>(C, 0));
    for (int i = 0; i < 2; i++) {
        arrx[i].clear();
        arry[i].clear();
        psumx[i].clear();
        psumy[i].clear();
    }
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            cin >> grid[i][j];
            if (grid[i][j] == 1) {
                arrx[0].push_back(i + j);
                arry[0].push_back(i - j);
            } else if (grid[i][j] == 2) {
                arrx[1].push_back(i + j);
                arry[1].push_back(i - j);
            }
        }
    }
    for (int i = 0; i < 2; i++) {
        sort(arrx[i].begin(), arrx[i].end());
        sort(arry[i].begin(), arry[i].end());
        psumx[i].assign(arrx[i].size(), 0);
        psumy[i].assign(arry[i].size(), 0);
        for (int j = 0; j < arrx[i].size(); j++) {
            psumx[i][j] = arrx[i][j];
            if (j > 0) {
                psumx[i][j] += psumx[i][j - 1];
            }
            psumy[i][j] = arry[i][j];
            if (j > 0) {
                psumy[i][j] += psumy[i][j - 1];
            }
        }
    }
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            int s1 = calc(arrx[0], psumx[0], i + j) + calc(arry[0], psumy[0], i - j);
            int s2 = calc(arrx[1], psumx[1], i + j) + calc(arry[1], psumy[1], i - j);
            int ans = abs(s1 - s2) / 2;
            cout << ans << " ";
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

## RGB Grid (Easy)

### Solution 1:  dynamic programming, bitmask dp, binary exponentiation, push dp, base conversion

1. get digit that is encoded with a base, find the ith term
2. count valid, by counting invalid and taking total - invalid possibilities
3. Use recursion dp, so it returns 1 when it reaches a final state

```cpp
int N, M, P;
vector<vector<vector<int>>> dp;
vector<int> masks;
int POWS[8];

// p = b^i
// mask is the integer representation
int get_digit(int mask, int p, int b = 3) {
    return (mask / p) % b;
}

int exponentiation(int b, int p, int m) {
    int res = 1;
    while (p > 0) {
        if (p & 1) res = (res * b) % m;
        b = (b * b) % m;
        p >>= 1;
    }
    return res;
}

bool is_valid(int m1, int m2, int m3) {
    for (int i = 0; i < N; i++) {
        int d1 = get_digit(m1, POWS[i]);
        int d2 = get_digit(m2, POWS[i]);
        int d3 = get_digit(m3, POWS[i]);
        if (d1 == 0 && d2 == 1 && d3 == 2) return false;
        if (d1 == 2 && d2 == 1 && d3 == 0) return false;
    }
    return true;
}

// solve by valid - invalid

// use push dp since can't figure out how to do with pull dp

// from (m1, m2) to (m2, m3)
int push(int i, int m1, int m2) {
    if (i == M) return 1;
    if (dp[i][m1][m2] != -1) return dp[i][m1][m2];
    int ans = 0;
    for (int m3 : masks) {
        if (is_valid(m1, m2, m3)) {
            ans = (ans + push(i + 1, m2, m3)) % P;
        }
    }
    return dp[i][m1][m2] = ans;
}

void solve() {
    cin >> N >> M >> P;
    POWS[0] = 1;
    for (int i = 1; i < 7; i++) POWS[i] = POWS[i - 1] * 3;
    for (int mask = 0; mask < POWS[N]; mask++) {
        bool valid = true;
        for (int i = 0; i < N - 2; i++) {
            if (get_digit(mask, POWS[i]) == 0 && get_digit(mask, POWS[i + 1]) == 1 && get_digit(mask, POWS[i + 2]) == 2) {
                valid = false;
                break;
            }
            if (get_digit(mask, POWS[i]) == 2 && get_digit(mask, POWS[i + 1]) == 1 && get_digit(mask, POWS[i + 2]) == 0) {
                valid = false;
                break;
            }
        }
        if (valid) masks.push_back(mask);
    }
    int total = exponentiation(3, N * M, P);
    if (M == 1) {
        total = (total - masks.size() + P) % P;
        cout << total << endl;
        return;
    }
    dp.assign(M, vector<vector<int>>(POWS[N], vector<int>(POWS[N], -1)));
    for (int m1 : masks) {
        for (int m2 : masks) {
            total = (total - push(2, m1, m2) + P) % P; // at index 2, and taking m1, m2 from index 0, 1, and picking m3 from index 2
        }
    }
    cout << total << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

# Starters 152

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

# Starters 153

## Another Game

### Solution 1:  max, array, sorting, permutations

```cpp
int N;
vector<int> arr;

void solve() {
    cin >> N;
    arr.resize(N);
    int mx = -1;
    for (int i = 0; i < N; i++) {
        cin >> arr[i];
        if (arr[i] != i + 1) mx = max(mx, arr[i]);
    }
    int ans = mx + 1;
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

## Colorful Tree (Easy Version)

### Solution 1:  undirected graph, tree, degree

1. if the degree = 1 it contributes 2 to the answer, otherwise it contributes 3

```cpp
int N;
vector<int> deg;

void solve() {
    cin >> N;
    deg.assign(N, 0);
    for (int i = 0; i < N - 1; i++) {
        int u, v;
        cin >> u >> v;
        u--; v--;
        deg[u]++;
        deg[v]++;
    }
    int ans = 0;
    for (int i = 0; i < N; i++) {
        ans += 2;
        if (deg[i] == 1) ans++;
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

## Xometry (Easy Version)

### Solution 1:  frequency array, math

1. The idea is to iterate through the array and find the xor of the two elements
1. Then you can find the frequency of the xor and add it to the answer
1. The answer is the frequency of the xor * 8
1. This only works because all the elements are distinct

```cpp
const int MAXN = 5'005, MAX_XOR = 1 << 20;
int N;
int arr[MAXN];
int freq[MAX_XOR];

void solve() {
    cin >> N;
    for (int i = 0; i < N; i++) {
        cin >> arr[i];
    }
    memset(freq, 0, sizeof(freq));
    int ans = 0;
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            int v = arr[i] ^ arr[j];
            ans += freq[v];
            freq[v]++;
        }
    }
    cout << ans * 8LL << endl;
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

## Xometry (Hard Version)

### Solution 1:  prefix sums, combinatorics, frequency arrays, xor, math

```cpp
const int MAXN = 1e6 + 5, MAX_XOR = 1 << 20;
int N;
vector<int> arr;
int freq[MAXN], freq_xor[MAX_XOR];

int choose(int n) {
    return n * (n - 1) / 2;
}

void solve() {
    cin >> N;
    arr.resize(N);
    memset(freq, 0, sizeof(freq));
    set<int> vals;
    for (int i = 0; i < N; i++) {
        cin >> arr[i];
        freq[arr[i]]++;
        vals.insert(arr[i]);
    }
    memset(freq_xor, 0, sizeof(freq_xor));
    vector<int> vals_vec(vals.begin(), vals.end());
    int ans = 0, M = vals_vec.size();
    int sum = 0, pways = 0;
    vector<int> xor_vals;
    for (int i = 0; i < M; i++) {
        sum += pways * choose(freq[vals_vec[i]]);
        pways += choose(freq[vals_vec[i]]);
        for (int j = i + 1; j < M; j++) {
            int val = vals_vec[i] ^ vals_vec[j];
            if (!freq_xor[val]) xor_vals.push_back(val);
            ans -= choose(freq[vals_vec[i]] * freq[vals_vec[j]]);
            freq_xor[val] += freq[vals_vec[i]] * freq[vals_vec[j]];
        }
    }
    for (int v : xor_vals) {
        ans += choose(freq_xor[v]);
    }
    ans += sum;
    cout << ans * 8LL << endl;
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

# Starters 154

## Add 1 or 2 Game

### Solution 1:  observation

```cpp
int N;

void solve() {
    cin >> N;
    if (N == 1) {
        cout << "ALICE" << endl;
    } else {
        cout << "BOB" << endl;
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

## GCD and XOR

### Solution 1:  greedy, gcd, xor

```cpp
int N, K;
vector<int> arr;

bool check() {
    vector<int> vals;
    vector<int> indices;
    for (int i = 0; i < N; i++) {
        if (arr[i] != K) {
            vals.push_back(arr[i]);
            indices.push_back(i);
        }
    }
    for (int i = 1; i < vals.size(); i++) {
        if (vals[i] != vals[i - 1]) {
            return false;
        }
    }
    for (int i = 1; i < indices.size(); i++) {
        if (indices[i] != indices[i - 1] + 1) {
            return false;
        }
    }
    return true;
}

void solve() {
    cin >> N >> K;
    arr.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> arr[i];
    }
    if (all_of(arr.begin(), arr.end(), [&](int x) { return x == K; })) {
        cout << 0 << endl;
    } else if (check()) {
        cout << 1 << endl;
    } else if (all_of(arr.begin(), arr.end(), [&](int x) { return gcd(x, K) == K; })) {
        cout << 1 << endl;
    } else {
        cout << 2 << endl;
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

## Triangle Count (Easy)

### Solution 1:  math, triangle inequality, sort, intervals

```cpp
const int INF = 1e18;
int N;
vector<int> arr;

void solve() {
    cin >> N;
    arr.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> arr[i];
    }
    sort(arr.begin(), arr.end());
    int ans = 0;
    vector<pair<int, int>> intervals;
    for (int i = 1; i < N; i++) {
        int l = arr[i] - arr[i - 1] + 1;
        int r = arr[i - 1] + arr[i] - 1;
        intervals.emplace_back(l, r);
    }
    sort(intervals.begin(), intervals.end());
    int upper = -INF;
    for (auto [l, r] : intervals) {
        int val = max(0LL, r - max(upper, l) + 1);
        ans += val;
        upper = max(upper, r + 1) ;
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

## Count Winning Subarrays

### Solution 1:  pointers, greedy

```cpp
int N;
vector<int> arr;

void solve() {
    cin >> N;
    arr.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> arr[i];
    }
    int last1 = -1, last2 = -1;
    int ans = 0;
    for (int i = 1; i < N; i++) {
        if (arr[i] == arr[i - 1] && arr[i] == 1) {
            last1 = i - 1;
        } 
        if (i > 1 && arr[i] == 1 && arr[i - 1] == 0 && arr[i - 2] == 1) {
            last2 = i - 2;
        }
        int diff = max(last1, last2) + 1;
        ans += diff;
    }
    ans += count(arr.begin(), arr.end(), 1);
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

## Tree Cut Xor

### Solution 1:  tree, xor, stack, dfs

```cpp
int N;
vector<vector<int>> adj;
stack<pair<int, int>> edges;

void dfs(int u, int p = -1) {
    for (int v : adj[u]) {
        if (v == p) continue;
        edges.emplace(u, v);
        dfs(v, u);
    }
}

void solve() {
    cin >> N;
    adj.assign(N, vector<int>());
    for (int i = 0; i < N - 1; i++) {
        int u, v;
        cin >> u >> v;
        u--, v--;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    dfs(0);
    int ans = N > 2 ? 0 : 1;
    cout << ans << endl;
    while (!edges.empty()) {
        auto [u, v] = edges.top(); // u - v (v is leaf)
        if (N & 1 || edges.size() > 3) {
            cout << u + 1 << " " << v + 1 << " " << v + 1 << endl;
        } else {
            cout << u + 1 << " " << v + 1 << " " << u + 1 << endl;
        }
        edges.pop();
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

# Starters 155

## GCD to 1 (Hard)

### Solution 1: 

```cpp
int R, C;
vector<vector<int>> mat;

void solve() {
    cin >> R >> C;
    mat.assign(R, vector<int>(C, 2));
    for (int i = 0; i < max(R , C); i++) {
        mat[i % R][i % C] = 3;
    }
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            cout << mat[i][j] << " ";
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

## Prefix Suffix Min Max

### Solution 1: 

```cpp
int N;
vector<int> A, B;

void solve() {
    cin >> N;
    B.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> B[i];
    }
    A.assign(N, 0);
    for (int i = 1; i < N; i++) {
        A[0] = max(A[0], B[i] - B[i - 1]);
    }
    for (int i = 1; i < N; i++) {
        A[i] = B[i] - B[i - 1];
    }
    for (int i = 0; i < N; i++) {
        cout << A[i] << " ";
    }
    cout << endl;
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

## Array Concatanation

### Solution 1:  combinatorics, counting, parity

```cpp
const int MOD = 1e9 + 7, MAXN = 2e6 + 5;
int a, b;

int ceil(int x, int y) {
    return (x + y - 1) / y;
}

int floor(int x, int y) {
    return x / y;
}

int inv(int i) {
  return i <= 1 ? i : MOD - (int)(MOD/i) * inv(MOD % i) % MOD;
}

vector<int> fact, inv_fact;

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

int choose(int n, int r) {
    if (n < r) return 0;
    return (fact[n] * inv_fact[r] % MOD) * inv_fact[n - r] % MOD;
}

void solve() {
    cin >> a >> b;
    int cnt1 = ceil(a + b, 4), cnt2 = floor(a + b, 4);
    int ans = 0;
    for (int i = 0; i <= floor(b, 2); i++) {
        int c1 = choose(cnt1, i) * choose(cnt2, i) % MOD;
        int c2 = c1 * choose(a + b - cnt1 - cnt2, b - 2 * i) % MOD;
        ans = (ans + c2) % MOD;
    }
    cout << ans << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    factorials(MAXN);
    int T;
    cin >> T;
    while (T--) {
        solve();
    }
    return 0;
}
```

# Starters 156

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

# Starters 157

## Unmedian

### Solution 1:  min, max, index

```cpp
int N;
vector<int> A;

void solve() {
    cin >> N;
    A.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    int smallest = *min_element(A.begin(), A.end());
    int largest = *max_element(A.begin(), A.end());
    int si = 0, li = 0;
    for (int i = 0; i < N; i++) {
        if (A[i] == smallest) si = i;
        if (A[i] == largest) li = i;
    }
    if (li < si) {
        cout << -1 << endl;
        return;
    }
    cout << N - 2 << endl;
    for (int i = 0; i < N - 2; i++) {
        cout << 1 << " " << 3 << endl;
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

## Normal is Good

### Solution 1:  stack, frequency, map, prefix sum

1. The key to solving this is to map the values to A[i] - 2, then 1 -> -1, 2 -> 0, 3 -> 1,  then if the subarray sum equals to 0 that indicates it is the case where median is 2.  That is you have same number of 1s and 3s, count(1) == count(3).  Which is necessary for median = 2, but also for there to be at least one occurrence of 2 in the subarray. 

```cpp
int N;
vector<int> A;

int calc(int n) {
    return n * (n + 1) / 2;
}

void solve() {
    cin >> N;
    A.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    int ans = 0, psum = 0, prv = -1, cur = 0;
    stack<int> stk;
    stk.push(0);
    map<int, int> freq;
    for (int i = 0; i < N; i++) {
        int x = A[i] - 2;
        psum += x;
        if (x == 0) {
            while (!stk.empty()) {
                int top = stk.top();
                stk.pop();
                freq[top]++;
            }
        }
        if (x != prv) {
            cur = 0;
            prv = x;
        }
        cur++;
        if (x != 0) ans += cur;
        ans += freq[psum];
        stk.push(psum);
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

# Starters 158

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