# USP Tryouts 2024

## 

### Solution 1: 

```cpp

```

## C. Road Cycling

### Solution 1: 

```cpp

```

## D. A is for Apple

### Solution 1: 

```cpp

```

## Acaraje

### Solution 1: 

```cpp
void solve() {
    int N;
    cin >> N;
    vector<int> consumers(N);
    for (int i = 0; i < N; i++) {
        cin >> consumers[i];
    }
    sort(consumers.begin(), consumers.end());
    int price = 0, rev = 0;
    for (int i = 0; i < N; i++) {
        int cand = consumers[i] * (N - i);
        if (cand > rev) {
            rev = cand;
            price = consumers[i];
        }
    }
    cout << price << " " << rev << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## From Baikonur to Mars

### Solution 1: 

```cpp

```

## G. Teleporting through Kazakhstan

### Solution 1: 

```cpp
const int INF = 1e16;
int N;
vector<int> dp, ndp, arr;

void solve() {
    cin >> N;
    dp.assign(N + 1, INF);
    dp[0] = 0;
    arr.resize(N + 1);
    arr[0] = 0;
    for (int i = 1; i <= N; i++) {
        cin >> arr[i];
        ndp.assign(N + 1, INF);
        for (int j = 0; j < i; j++) {
            ndp[j] = min(ndp[j], dp[j] + abs(arr[i] - arr[i - 1]));
            if (j < i - 1) ndp[i - 1] = min(ndp[i - 1], dp[j] + abs(arr[i] - arr[j]));
        }
        swap(dp, ndp);
    }
    int ans = *min_element(dp.begin(), dp.end());
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

## K. Grabbing plush

### Solution 1: 

```cpp
const int INF = 1e12;
int neutral = -INF, noop = 0;

struct LazySegmentTree {
    vector<int> values;
    vector<int> operations;
    int size;

    void init(int n) {
        size = 1;
        while (size < n) size *= 2;
        values.assign(2 * size, noop);
        operations.assign(2 * size, noop);
    }

    int modify_op(int x, int y, int length = 1) {
        return x + y * length;
    }

    int calc_op(int x, int y) {
        return max(x, y);
    }

    bool is_leaf(int segment_right_bound, int segment_left_bound) {
        return segment_right_bound - segment_left_bound == 1;
    }

    void propagate(int segment_idx, int segment_left_bound, int segment_right_bound) {
        if (is_leaf(segment_right_bound, segment_left_bound) || operations[segment_idx] == noop) return;
        int left_segment_idx = 2 * segment_idx + 1, right_segment_idx = 2 * segment_idx + 2;
        int children_segment_len = (segment_right_bound - segment_left_bound) >> 1;
        operations[left_segment_idx] = modify_op(operations[left_segment_idx], operations[segment_idx]);
        operations[right_segment_idx] = modify_op(operations[right_segment_idx], operations[segment_idx]);
        values[left_segment_idx] = modify_op(values[left_segment_idx], operations[segment_idx], children_segment_len);
        values[right_segment_idx] = modify_op(values[right_segment_idx], operations[segment_idx], children_segment_len);
        operations[segment_idx] = noop;
    }

    void ascend(int segment_idx) {
        while (segment_idx > 0) {
            segment_idx--;
            segment_idx >>= 1;
            int left_segment_idx = 2 * segment_idx + 1, right_segment_idx = 2 * segment_idx + 2;
            values[segment_idx] = calc_op(values[left_segment_idx], values[right_segment_idx]);
        }
    }

    void update(int left, int right, int val) {
        stack<tuple<int, int, int>> stk;
        stk.emplace(0, size, 0);
        vector<int> segments;
        int segment_left_bound, segment_right_bound, segment_idx;
        while (!stk.empty()) {
            tie(segment_left_bound, segment_right_bound, segment_idx) = stk.top();
            stk.pop();
            // NO OVERLAP
            if (segment_left_bound >= right || segment_right_bound <= left) continue;
            // COMPLETE OVERLAP
            if (segment_left_bound >= left && segment_right_bound <= right) {
                operations[segment_idx] = modify_op(operations[segment_idx], val);
                int segment_len = segment_right_bound - segment_left_bound;
                values[segment_idx] = modify_op(values[segment_idx], val, segment_len);
                segments.push_back(segment_idx);
                continue;
            }
            // PARTIAL OVERLAP
            int mid_point = (segment_left_bound + segment_right_bound) >> 1;
            int left_segment_idx = 2 * segment_idx + 1, right_segment_idx = 2 * segment_idx + 2;
            propagate(segment_idx, segment_left_bound, segment_right_bound);
            stk.emplace(mid_point, segment_right_bound, right_segment_idx);
            stk.emplace(segment_left_bound, mid_point, left_segment_idx);
        }
        for (int segment_idx : segments) ascend(segment_idx);
    }

    pair<int, int> query(int left, int right) {
        stack<tuple<int, int, int>> stk;
        stk.emplace(0, size, 0);
        int result = neutral, idx = -1;
        int segment_left_bound, segment_right_bound, segment_idx;
        while (!stk.empty()) {
            tie(segment_left_bound, segment_right_bound, segment_idx) = stk.top();
            stk.pop();
            // NO OVERLAP
            if (segment_left_bound >= right || segment_right_bound <= left) continue;
            propagate(segment_idx, segment_left_bound, segment_right_bound);
            // COMPLETE OVERLAP
            if (segment_left_bound >= left && segment_right_bound <= right) {
                if (is_leaf(segment_right_bound, segment_left_bound)) {
                    // cout << segment_left_bound << " " << segment_right_bound << endl;
                    if (values[segment_idx] >= result) idx = segment_left_bound;
                } else {
                    int left_segment_idx = 2 * segment_idx + 1, right_segment_idx = 2 * segment_idx + 2;
                    int mid_point = (segment_left_bound + segment_right_bound) >> 1;
                    // cout << segment_left_bound << " " << mid_point << " " << segment_right_bound << endl;
                    // cout << "left: " << values[left_segment_idx] << " right: " << values[right_segment_idx] << endl;
                    if (values[left_segment_idx] > values[right_segment_idx]) {
                        stk.emplace(segment_left_bound, mid_point, left_segment_idx);
                    } else {
                        stk.emplace(mid_point, segment_right_bound, right_segment_idx);
                    }
                }
                result = calc_op(result, values[segment_idx]);
                continue;
            }
            // cout << "partial overlap" << endl;
            // PARTIAL OVERLAP
            int mid_point = (segment_left_bound + segment_right_bound) >> 1;
            int left_segment_idx = 2 * segment_idx + 1, right_segment_idx = 2 * segment_idx + 2;
            stk.emplace(mid_point, segment_right_bound, right_segment_idx);
            stk.emplace(segment_left_bound, mid_point, left_segment_idx);
        }
        return {result, idx};
    }
};

LazySegmentTree seg;
int N, M, W;
vector<int> arr, ssum;
vector<bool> vis;

void solve() {
    cin >> N >> M >> W;
    arr.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> arr[i];
    }
    ssum.assign(N, 0);
    for (int i = N - 1; i >= 0; i--) {
        ssum[i] = arr[i];
        if (i + 1 < N) ssum[i] += ssum[i + 1];
        if (i + W < N) ssum[i] -= arr[i + W];
    }
    vis.assign(N, false);
    for (int i = N - 1; i > N - W; i--) {
        ssum[i] = -INF;
    }
    // for (int x : ssum) cout << x << " ";
    // cout << endl;
    seg.init(N);
    for (int i = 0; i < N; i++) {
        seg.update(i, i + 1, ssum[i]);
    }
    // need to find index as well as value. 
    // auto [val, idx] = seg.query(0, N);
    // cout << val << " " << idx << endl;
    int ans = 0;
    while (true) {
        auto [val, idx] = seg.query(0, N);
        // cout << val << " " << idx << endl;
        if (val <= 0) break;
        ans += val;
        for (int i = idx; i < min(N, idx + W); i++) {
            if (vis[i]) continue;
            vis[i] = true;
            seg.update(max(0LL, i - W + 1), i + 1, -arr[i]);
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

## 

### Solution 1: 

```cpp

```