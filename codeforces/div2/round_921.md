# Codeforces Round 921 Div 2

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
def main():
    n, k, m = map(int, input().split())
    s = input()
    unicode = lambda ch: ord(ch) - ord("a")
    char = lambda i: chr(i + ord("a"))
    ans = []
    seen = [0] * k
    sum_ = 0
    for i in range(m):
        v = unicode(s[i])
        if not seen[v]:
            seen[v] = 1
            sum_ += 1
        if sum_ == k:
            ans.append(s[i])
            seen = [0] * k
            sum_ = 0
        if len(ans) == n: return print("YES")
    for i in range(k):
        if not seen[i]:
            ans.append(char(i))
            break
    while len(ans) < n:
        ans.append(ans[-1])
    print("NO")
    print("".join(ans))

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## 

### Solution 1: 

```py

```

## E. Space Harbour

### Solution 1:  Lazy Segment Tree, assignment of value, ranges, arithmetic progression

```cpp
const int MAXN = 3 * 100'000 + 5, neutral = 0, noop = 0;
int N, M, Q;
int pos[MAXN];

struct LazySegmentTree {
    vector<int> values;
    vector<int> operations;
    vector<pair<int, int>> range;
    int size;

    void init(int n, vector<int> &init_arr) {
        size = 1;
        while (size < n) size *= 2;
        values.assign(2 * size, neutral);
        operations.assign(2 * size, noop);
        range.assign(2 * size, {0, 0});
        build(init_arr);
    }

    void build(vector<int> &init_arr) {
        for (int i = 0; i < init_arr.size(); i++) {
            int segment_idx = i + size -1;
            int val = init_arr[i];
            values[segment_idx] = val;
            ascend(segment_idx);
        }
    }

    int arithmetic_progression(int lo, int hi) {
        return (hi - lo + 1) * (lo + hi) / 2;
    }

    int modify_op(int segment_idx, int left_bound, int right_bound) {
        int left = max(left_bound, range[segment_idx].start);
        int right = min(right_bound, range[segment_idx].end);
        if (right - left < 1) return 0;
        // 5 4 3 2 1 and I need 4 3, how to do that. 
        // 3 4 5 6 7
        // end = 8, let's say right = 8, then it gives 0, if right = 3, you take 8 - 3 = 5
        int lo = range[segment_idx].end - right + 1, hi = range[segment_idx].end - left;
        return operations[segment_idx] * arithmetic_progression(lo, hi);
    }

    int calc_op(int x, int y) {
        return x + y;
    }

    bool is_leaf(int segment_right_bound, int segment_left_bound) {
        return segment_right_bound - segment_left_bound == 1;
    }

    void propagate(int segment_idx, int segment_left_bound, int segment_right_bound) {
        if (is_leaf(segment_right_bound, segment_left_bound) || operations[segment_idx] == noop) return;
        int left_segment_idx = 2 * segment_idx + 1, right_segment_idx = 2 * segment_idx + 2;
        int segment_mid = (segment_left_bound + segment_right_bound) >> 1;
        operations[left_segment_idx] = operations[segment_idx];
        operations[right_segment_idx] = operations[segment_idx];
        range[left_segment_idx] = range[segment_idx];
        range[right_segment_idx] = range[segment_idx];
        values[left_segment_idx] = modify_op(segment_idx, segment_left_bound, segment_mid);
        values[right_segment_idx] = modify_op(segment_idx, segment_mid, segment_right_bound);
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
                operations[segment_idx] = val;
                range[segment_idx] = make_pair(left, right);
                values[segment_idx] = modify_op(segment_idx, segment_left_bound, segment_right_bound);
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

    int query(int left, int right) {
        stack<tuple<int, int, int>> stk;
        stk.emplace(0, size, 0);
        int result = neutral;
        int segment_left_bound, segment_right_bound, segment_idx;
        while (!stk.empty()) {
            tie(segment_left_bound, segment_right_bound, segment_idx) = stk.top();
            stk.pop();
            // NO OVERLAP
            if (segment_left_bound >= right || segment_right_bound <= left) continue;
            // COMPLETE OVERLAP
            if (segment_left_bound >= left && segment_right_bound <= right) {
                result = calc_op(result, values[segment_idx]);
                continue;
            }
            // PARTIAL OVERLAP
            int mid_point = (segment_left_bound + segment_right_bound) >> 1;
            int left_segment_idx = 2 * segment_idx + 1, right_segment_idx = 2 * segment_idx + 2;
            propagate(segment_idx, segment_left_bound, segment_right_bound);
            stk.emplace(mid_point, segment_right_bound, right_segment_idx);
            stk.emplace(segment_left_bound, mid_point, left_segment_idx);
        }
        return result;
    }

};

void solve() {
    cin >> N >> M >> Q;
    map<int, int> harbors;
    int val;
    for (int i = 0; i < M; i++) {
        cin >> pos[i];
    }
    for (int i = 0; i < M; i++) {
        cin >> val;
        harbors[--pos[i]] = val;
    }
    LazySegmentTree segtree;
    vector<int> arr(N, 0);
    val = harbors[0];
    for (int i = 1; i < N; i++) {
        int k, v;
        tie(k, v) = *harbors.lower_bound(i);
        int dist = k - i;
        arr[i] = dist * val;
        if (k == i) val = v;
    }
    segtree.init(N, arr);
    int t, x, v, l, r;
    while (Q--) {
        cin >> t;
        if (t == 1) {
            cin >> x >> v;
            x--;
            auto prev_it = harbors.upper_bound(x);
            auto next_it = prev_it--;
            harbors[x] = v;
            int left, right, val;
            // update to segment to left of new harbor
            left = (*prev_it).first + 1, right = x, val = (*prev_it).second;
            segtree.update(left, right, val);
            // set placement of new harbor
            segtree.update(x, x + 1, 0);
            // update to segment to right of new harbor
            left = x + 1, right = (*next_it).first, val = v;
            segtree.update(left, right, val);
        } else {
            cin >> l >> r;
            l--; r--;
            cout << segtree.query(l, r + 1) << endl;
        }
    }
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    while (T--) {
        solve();
    }
    return 0;
}
```

## F. Fractal Origami

### Solution 1: 

```py

```

