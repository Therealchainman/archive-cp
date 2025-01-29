# USACO

# USACO 2024 December Contest, Silver

## Cake Game

### Solution 1:  prefix sum, suffix sum

```cpp
int N;
vector<int> A;

void solve() {
    cin >> N;
    A.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    int psum = 0, ssum = 0;
    for (int i = 0; i < N / 2 - 1; i++) {
        psum += A[i];
    }
    int ans = psum;
    for (int i = N - 1, j = N / 2 - 2; i > N / 2; i--, j--) {
        ssum += A[i];
        psum -= A[j];
        ans = max(ans, psum + ssum);
    }
    int sum = accumulate(A.begin(), A.end(), 0LL);
    cout << sum - ans << " " << ans << endl;
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

## Deforestation

### Solution 1:  min heap, line sweep, sorting, binary search

```cpp
struct Event {
    int x, tp, r, t;
    Event() {}
    Event(int x, int tp, int r = 0, int t = 0) : x(x), tp(tp), r(r), t(t) {}
    bool operator<(const Event &other) const {
        if (x != other.x) {
            return x < other.x;
        }
        return tp < other.tp;
    }
};

int N, K;
vector<int> A;
vector<Event> events;

void solve() {
    cin >> N >> K;
    A.assign(N, 0);
    events.clear();
    for (int i = 0; i < N; i++) {
        cin >> A[i];
        events.emplace_back(A[i], 1);
    }
    sort(A.begin(), A.end());
    for (int i = 0; i < K; i++) {
        int l, r, t;
        cin >> l >> r >> t;
        int cnt = upper_bound(A.begin(), A.end(), r) - lower_bound(A.begin(), A.end(), l);
        events.emplace_back(l, 0, r, cnt - t);
    }
    sort(events.begin(), events.end());
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> minheap;
    int ans = 0;
    for (auto &[x, tp, r, t] : events) {
        while (!minheap.empty() && minheap.top().second < x) {
            minheap.pop();
        }
        if (tp == 0) {
            minheap.emplace(ans + t, r);
        } else {
            if (minheap.empty() || minheap.top().first > ans) {
                ans++;
            }
        }
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

## 2D Conveyor Belt

### Solution 1: 

```cpp

```

# USACO 2025 January Contest, Silver

## Cow Checkups

### Solution 1:  prefix sums, aggregation, sorting, binary search, greedy

```cpp
int N;
vector<int> A, B;
vector<vector<int>> groups;
vector<vector<int64>> psums;

int64 chooseTwo(int64 n) {
    return n * (n + 1) / 2;
}

void solve() {
    cin >> N;
    A.resize(N);
    B.resize(N);
    groups.assign(N + 1, vector<int>());
    psums.assign(N + 1, vector<int64>());
    for (int i = 0; i < N; i++) {
        cin >> A[i];
        groups[A[i]].emplace_back(min(i + 1, N - i));
    }
    for (int i = 1; i <= N; i++) {
        sort(groups[i].begin(), groups[i].end());
        psums[i].emplace_back(0);
        for (int j = 0; j < groups[i].size(); j++) {
            psums[i].emplace_back(psums[i].back() + groups[i][j]);
        }
    }
    int64 ans = 0;
    for (int i = 0; i < N; i++) {
        cin >> B[i];
        int idx = min(i + 1, N - i);
        int j = upper_bound(groups[B[i]].begin(), groups[B[i]].end(), idx) - groups[B[i]].begin();
        ans += psums[B[i]][j];
        int64 thresholdSum = static_cast<int64>(groups[B[i]].size() - j) * idx;
        ans += thresholdSum;
        if (A[i] == B[i]) {
            ans += chooseTwo(i) + chooseTwo(N - i - 1);
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

## Farmer John's Favorite Operation

### Solution 1: 

help this is wrong somehow

```cpp
int64 N, M;
map<int64, vector<bool>> events;

int ceil(int64 a, int64 b) {
    return (a + b - 1) / b;
}

void solve() {
    cin >> N >> M;
    events.clear();
    int64 cur = 0, pos = 0, neg = 0, pt = 0;
    int64 H = M / 2;
    for (int i = 0; i < N; i++) {
        int64 x;
        cin >> x;
        x %= M;
        // event 0
        events[x].emplace_back(false);
        events[x + M].emplace_back(false);
        debug("x: " , x, "event 0: ", x, x + M, "\n");
        if (x >= ceil(M, 2)) x -= M;
        cur += abs(x);
        // event 1
        int64 ev1 = M + x - H, ev2 = 2LL * M + x - H;
        if (M & 1) {
            if (x == H) {
                ev1 = 0, ev2 = M;
            }
        }
        events[ev1].emplace_back(true);
        events[ev2].emplace_back(true);
        debug("x: ", x, "event 1: " , ev1, ev2, "\n");
        if (x > 0) ++pos;
        else ++neg;
    }
    int64 ans = cur;
    for (const auto& [t, eventTypes] : events) {
        assert(eventTypes.size() > 0);
        assert(pos + neg == N);
        int64 cnt0 = 0, cnt1 = 0;
        for (bool eventType : eventTypes) {
            if (eventType) {
                cnt1++;
            } else {
                cnt0++;
            }
        }
        int64 deltaT = t - pt;
        int64 delta = neg - pos;
        cur += deltaT * delta;
        if (M & 1 && deltaT > 0) cur -= cnt1;
        debug("t: ", t, "delta: ", delta, "cnt0: ", cnt0, "cnt1: ", cnt1, "cur: ", cur, "deltaT: ", deltaT, "\n");
        pt = t;
        pos += cnt1;
        neg -= cnt1;
        neg += cnt0;
        pos -= cnt0;
        ans = min(ans, cur);
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

## Table Recovery

### Solution 1: 

```cpp

```

# USACO 2025 February Contest, Silver

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

# USACO 2025 Open???

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