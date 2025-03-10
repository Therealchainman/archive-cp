# Calgary Collegiate Programming Contest

# Calgary Collegiate Programming Contest 2025 Open Division

## Fair Prizes

### Solution 1: prefix sums, dynamic programming

```cpp
const int INF = 1e9;
int T, L, C, W;
vector<int> psum;
vector<vector<int>> dp;

int sumRange(int l, int r) {
    return psum[r] - psum[l];
}

void solve() {
    cin >> T >> L >> C >> W;
    psum.assign(L + 1, 0);
    for (int i = 0, x; i < T; i++) {
        cin >> x;
        psum[x] = 1;
    }
    for (int i = 1; i <= L; i++) {
        psum[i] += psum[i - 1];
    }
    dp.assign(L + 1, vector<int>(C + 1, -INF));
    dp[0][0] = 0;
    for (int i = 1; i <= L; i++) {
        dp[i][0] = 0;
        for (int j = 1; j <= C; j++) {
            dp[i][j] = dp[i - 1][j];
            if (i >= W) {
                if (dp[i - W][j - 1] != -INF) {
                    dp[i][j] = max(dp[i][j], dp[i - W][j - 1] + sumRange(i - W, i));
                }
            }
        }
    }
    int ans = *max_element(dp[L].begin(), dp[L].end());
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

## Lazy River Ride

### Solution 1: min heap, Dijkstra's algorithm, grid

```cpp
struct Cell {
    int r, c, cost;
    Cell() {}
    Cell(int r, int c, int cost) : r(r), c(c), cost(cost) {}
    bool operator<(const Cell& other) const {
        return cost > other.cost;
    }
};

int R, C, x, y;
const vector<pair<int, int>> dirs = {{-1, -1}, {-1, 0}, {-1, 1}, {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}};
vector<vector<int>> grid;

bool inBounds(int r, int c) {
    return r >= 0 && r < R && c >= 0 && c < C;
}

void solve() {
    cin >> R >> C >> x >> y;
    grid.assign(R, vector<int>(C));
    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C; c++) {
            char ch;
            cin >> ch;
            if (ch == '#') grid[r][c] = -1;
            else grid[r][c] = ch - '0' - 1;
        }
    }
    x--; y--;
    priority_queue<Cell> minheap;
    vector<vector<bool>> vis(R, vector<bool>(C, false));
    minheap.emplace(x, y, 0);
    bool isReady = false;
    int dr, dc;
    while (!minheap.empty()) {
        auto [r, c, cost] = minheap.top();
        minheap.pop();
        if (r == x && c == y) {
            if (isReady) {
                cout << cost << endl;
                return;
            }
            isReady = true;
        }
        if (vis[r][c]) continue;
        vis[r][c] = true;
        tie(dr, dc) = dirs[(grid[r][c] - 1 + 8) % 8];
        if (inBounds(r + dr, c + dc) && grid[r + dr][c + dc] != -1) {
            minheap.emplace(r + dr, c + dc, cost + 1);
        }
        tie(dr, dc) = dirs[grid[r][c]];
        if (inBounds(r + dr, c + dc) && grid[r + dr][c + dc] != -1) {
            minheap.emplace(r + dr, c + dc, cost);
        }
        tie(dr, dc) = dirs[(grid[r][c] + 1) % 8];
        if (inBounds(r + dr, c + dc) && grid[r + dr][c + dc] != -1) {
            minheap.emplace(r + dr, c + dc, cost + 1);
        }
    }
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Magic Bucket

### Solution 1: fraction comparison, brute force

```cpp
int64 S, R, M;

void solve() {
    cin >> S >> R >> M;
    int64 num = 0, den = 1, cur = 0, t = 0;
    while (cur < M) {
        t++;
        cur += R;
        cur *= S;
        cur = min(cur, M);
        if (cur * den > num * t) {
            num = cur;
            den = t;
        }
    }
    cout << den << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Pool Filling

### Solution 1: greedy, sorting, reverse

```cpp
int N;
vector<long double> A;

void solve() {
    cin >> N;
    A.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    sort(A.rbegin(), A.rend());
    long double tank = 1.0;
    for (int i = 0; i < N; i++) {
        long double mixed = (tank + A[i]) / 2;
        if (mixed > A[i]) {
            A[i] = mixed;
            tank = mixed;
        }
    }
    long double ans = accumulate(A.begin(), A.end(), 0.0);
    cout << fixed << setprecision(10) << ans << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Same Slides

### Solution 1: brute force

```cpp
int N, Y, M, D;

bool isLeapYear(int y) {
    if (y % 100 == 0 && y % 400 != 0) return false;
    return y % 4 == 0;
}

void solve() {
    cin >> N >> Y >> M >> D;
    if (D > 29 && M == 2) {
        cout << "IMPOSSIBLE" << endl;
        return;
    }
    if (D == 29 && M == 2 && !isLeapYear(Y)) {
        cout << "IMPOSSIBLE" << endl;
        return;
    }
    int64 ans = 0;
    if (D == 29 && M == 2 && isLeapYear(Y)) {
        while (true) {
            ans += 369LL;
            Y++;
            if (isLeapYear(Y)) ans++;
            if (isLeapYear(Y) && ans % N == 0) {
                cout << ans << endl;
                return;
            }
        }
    }
    while (true) {
        ans += 369LL;
        if ((M == 1 || (M == 2 && D < 29)) && isLeapYear(Y)) {
            ans++;
        } else if (M > 2 && isLeapYear(Y + 1)) {
            ans++;
        }
        Y++;
        if (ans % N == 0) {
            cout << ans << endl;
            return;
        }
    }
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Squirt Gun Trajectory

### Solution 1: physics, kinematics

```cpp
const long double PI = 3.14159265358979323846, g = 9.81;
long double angle, V, Dmin, Dmax;

long double degToRad(long double degrees) {
    return degrees * (PI / 180.0);
}

void solve() {
    cin >> angle >> V >> Dmin >> Dmax;
    long double A = 2 * V * V * sin(degToRad(angle)) * cos(degToRad(angle));
    if (g * Dmin <= A && g * Dmax >= A) {
        cout << "POSSIBLE" << endl;
    } else {
        cout << "IMPOSSIBLE" << endl;
    }
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Pool Pollution

### Solution 1: dfs, undirected graph, connected components

```cpp
int N, M, K;
vector<vector<int>> adj;
vector<int> cap;

void dfs(int u) {
    if (!cap[u]) return;
    cap[u] = 0;
    for (int v : adj[u]) {
        dfs(v);
    }
}

void solve() {
    cin >> N >> M >> K;
    cap.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> cap[i];
    }
    adj.assign(N, vector<int>());
    for (int i = 0; i < M; i++) {
        int u, v;
        cin >> u >> v;
        u--, v--;
        adj[u].emplace_back(v);
        adj[v].emplace_back(u);
    }
    while (K--) {
        int u;
        cin >> u;
        u--;
        dfs(u);
    }
    int ans = accumulate(cap.begin(), cap.end(), 0);
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