# Codeforces Round 955 Div 2

## E. Number of k-good subarrays

### Solution 1: 

```cpp

```

## F. Sorting Problem Again

### Solution 1:  min/max segment tree, point updates, range queries, sorted set, binary searching sorted prefix and suffix

```cpp
const int INF = 1e9 + 5;
int N, Q;
vector<int> arr;
set<int> crits;

struct SegmentTree {
    int size;
    vector<int> nodes_max, nodes_min;

    void init(int num_nodes) {
        size = 1;
        while (size < num_nodes) size *= 2;
        nodes_min.assign(size * 2, INF);
        nodes_max.assign(size * 2, -INF);
    }

    void ascend(int segment_idx) {
        while (segment_idx > 0) {
            int left_segment_idx = 2 * segment_idx, right_segment_idx = 2 * segment_idx + 1;
            nodes_min[segment_idx] = min(nodes_min[left_segment_idx], nodes_min[right_segment_idx]);
            nodes_max[segment_idx] = max(nodes_max[left_segment_idx], nodes_max[right_segment_idx]);
            segment_idx >>= 1;
        }
    }

    void update(int segment_idx, int val) {
        segment_idx += size;
        nodes_min[segment_idx] = val;
        nodes_max[segment_idx] = val;
        segment_idx >>= 1;
        ascend(segment_idx);
    }

    pair<int, int> query(int left, int right) {
        left += size, right += size;
        int mx = -INF, mn = INF;
        while (left <= right) {
            if (left & 1) {
                mn = min(mn, nodes_min[left]);
                mx = max(mx, nodes_max[left]);
                left++;
            }
            if (~right & 1) {
                mn = min(mn, nodes_min[right]);
                mx = max(mx, nodes_max[right]);
                right--;
            }
            left >>= 1, right >>= 1;
        }
        return {mn, mx};
    }
};

SegmentTree seg;


pair<int, int> query() {
    if (crits.size() > 0) {
        int i_l = *crits.begin(), i_r = *--crits.end();
        auto [mn, mx] = seg.query(i_l - 1, i_r);
        int pi = upper_bound(arr.begin(), arr.begin() + i_l, mn) - arr.begin();
        int si = lower_bound(arr.begin() + i_r, arr.end(), mx) - arr.begin() - 1;
        return {++pi, ++si};
    } else {
        return {-1, -1};
    }
}

void solve() {
    cin >> N;
    arr.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> arr[i];
    }
    seg.init(N);
    crits.clear();
    for (int i = 0; i < N; i++) {
        seg.update(i, arr[i]);
        if (i > 0 && arr[i] < arr[i - 1]) crits.insert(i);
    }
    pair<int, int> res = query();
    cout << res.first << " " << res.second << endl;
    cin >> Q;
    while (Q--) {
        int pos, x;
        cin >> pos >> x;
        pos--;
        // update
        arr[pos] = x;
        if (pos > 0 && arr[pos - 1] <= arr[pos]) {
            crits.erase(pos);
        }
        if (pos > 0 && arr[pos - 1] > arr[pos]) {
            crits.insert(pos);
        }
        if (pos + 1 < N && arr[pos] <= arr[pos + 1]) {
            crits.erase(pos + 1);
        } 
        if (pos + 1 < N && arr[pos] > arr[pos + 1]) {
            crits.insert(pos + 1);
        }
        seg.update(pos, x);
        pair<int, int> res = query();
        cout << res.first << " " << res.second << endl;
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