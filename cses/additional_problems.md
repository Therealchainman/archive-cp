# Additional Problems

## 

### Solution 1:  

```py

```

## Multiplication Table

### Solution 1:  math, binary search, greedy

```cpp
const int64 INF = numeric_limits<int64>::max();
int64 N;

bool possible(int64 target) {
    int64 cnt = 0;
    for (int r = 1; r <= N; ++r) {
        int64 c = min(target / r, N);
        cnt += c;
    }
    return cnt <= N * N / 2;
}

void solve() {
    cin >> N;
    int64 lo = 1, hi = INF;
    while (lo < hi) {
        int64 mid = lo + (hi - lo) / 2;
        if (possible(mid)) lo = mid + 1;
        else hi = mid;
    }
    cout << lo << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Intersection Points

### Solution 1:  PURQ, point updates, range queries, sum, BIT, fenwick, line sweep, 2D grid to 1D range query, coordinate compression

```cpp
int N;

int neutral = 0;
struct FenwickTree {
    vector<int> nodes;
    
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
        return query(right) - query(left - 1);
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

struct Vertical {
    int x, y1, y2;
};

struct Horizontal {
    int y, x1, x2;
};

signed main() {
	int x1, y1, x2, y2;
    cin >> N;
    vector<Vertical> verticals;
    vector<Horizontal> horizontals;
    vector<int> values;
    for (int i = 0; i < N; i++) {
        cin >> x1 >> y1 >> x2 >> y2;
        values.push_back(x1);
        values.push_back(x2);
        values.push_back(y1);
        values.push_back(y2);
        if (x1 == x2) {
            verticals.push_back({x1, y1, y2});
        } else {
            horizontals.push_back({y1, x1, x2});
        }
    }
    FenwickTree fenwick;
    fenwick.init(values.size());
    // COORDINATE COMPRESSION
    sort(values.begin(), values.end());
    auto it = unique(values.begin(), values.end()); 
    values.resize(distance(values.begin(), it));
    map<int, int> compress;
    for (int &v : values) {
        compress[v] = compress.size() + 1;
    }
    // 1D LINE SWEEP ALGORITHM
    vector<tuple<int, int, int>> events;
    for (int i = 0; i < (int)horizontals.size(); i++) {
        events.emplace_back(compress[horizontals[i].x1], 1, i);
        events.emplace_back(compress[horizontals[i].x2], -1, i);
    }
    for (int i = 0; i < (int)verticals.size(); i++) {
        events.emplace_back(compress[verticals[i].x], 0, i);
    }
    sort(events.begin(), events.end());
    int ans = 0, l, r;
    for (auto &[_, t, idx] : events) {
        int y = compress[horizontals[idx].y];
        if (t == -1) {
            fenwick.update(y, -1);
        } else if (t == 0) {
            l = compress[verticals[idx].y1], r = compress[verticals[idx].y2];
            ans += fenwick.query(l, r);
        } else {
            fenwick.update(y, 1);
        }
    }
    cout << ans << endl;
}
```

## 

### Solution 1:  

```py

```

## 

### Solution 1:  

```py

```

## 

### Solution 1:  

```py

```

## Area of Rectangles

### Solution 1: line sweep, segment tree for union of rectangles, area covered by squares on 2d plane, segment tree with coordinate compression

```cpp
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

int N;

void solve() {
    cin >> N;
    vector<Event> events;
    vector<int64> xs;
    for (int i = 0; i < N; i++) {
        int x1, y1, x2, y2;
        cin >> x1 >> y1 >> x2 >> y2;
        events.emplace_back(y1, 1, x1, x2);
        events.emplace_back(y2, -1, x1, x2);
        xs.push_back(x1);
        xs.push_back(x2);
    }
    sort(events.begin(), events.end());
    SegmentTree seg(xs);
    int64 ans = 0, prevY = 0;
    for (const auto& [y, t, l, r] : events) {
        int64 dy = y - prevY;
        int64 dx = seg.query();
        ans += dy * dx;
        seg.update(l, r, t);
        prevY = y;
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