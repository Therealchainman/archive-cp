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