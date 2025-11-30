# Codeforces 2025

## The Child and Sequence

### Solution 1: segment tree beats, amortized analysis, range updates with modulo, range sum queries, point assignment updates

1. Modulo either does not affect a number, or decreases it by at least half of what it was.
1. The key idea: the bad cases where you recurse deeply cannot happen too many times, because each such deep recursion strictly shrinks values.

```cpp
const int64 INF = numeric_limits<int64>::max();
int N, M;

template<class Node, class Update>
struct SegmentTreeBeat {
    struct Configuration {
        const Node neutral;                           // identity for merge
        function<Node(const Node&, const Node&)> merge;           // combine two nodes
        // high level beats hook:
        // returns true if this node is fully handled by the update
        // returns false if the tree should recurse to children
        function<bool(Node&, int, int, const Update&)> apply;
    } config;

    int size = 0;
    vector<Node> nodes;

    SegmentTreeBeat(int n, Configuration config) : config(config) { init(n); }

    void init(int num_nodes) {
        size = 1;
        while (size < num_nodes) size *= 2;
        nodes.assign(size * 2, config.neutral);
    }

    void build(const vector<Node>& arr) {
        int n = arr.size();
        for (int i = 0; i < n; ++i) {
            nodes[size + i] = arr[i];
        }
        for (int i = size - 1; i >= 1; --i) {
            pull(i);
        }
    }

    void update_point(int segment_idx, const Node& val) {
        segment_idx += size;
        nodes[segment_idx] = val;
        for (segment_idx >>= 1; segment_idx >= 1; segment_idx >>= 1) pull(segment_idx);
    }

    void update_range(int left, int right, const Update& val) {
        update_range(1, 0, size - 1, left, right, val);
    }

    void update_range(int segment_idx, int segment_left_bound, int segment_right_bound, int left, int right, const Update& val) {
        // NO OVERLAP
        if (right < segment_left_bound || segment_right_bound < left) return;
        if (config.apply(nodes[segment_idx], segment_left_bound, segment_right_bound, val)) return;
        // RECURSE
        int mid_point = (segment_left_bound + segment_right_bound) >> 1;
        int left_segment_idx = segment_idx << 1, right_segment_idx = segment_idx << 1 | 1;
        update_range(left_segment_idx, segment_left_bound, mid_point, left, right, val);
        update_range(right_segment_idx, mid_point + 1, segment_right_bound, left, right, val);
        pull(segment_idx);
    }

    Node query(int left, int right) {
        left += size, right += size;
        Node left_acc = config.neutral;
        Node right_acc = config.neutral;
        while (left <= right) {
           if (left & 1) {
                // res on left
                left_acc = config.merge(left_acc, nodes[left++]);
            }
            if (~right & 1) {
                // res on right
                right_acc = config.merge(nodes[right--], right_acc);
            }
            left >>= 1, right >>= 1;
        }
        return config.merge(left_acc, right_acc);
    }
    private:
        inline void pull(int segment_idx) { nodes[segment_idx] = config.merge(nodes[segment_idx << 1], nodes[segment_idx << 1 | 1]); }
};

struct Node {
    int64 maxVal, sumVal;
    Node() {}
    Node(int64 maxVal, int64 sumVal) : maxVal(maxVal), sumVal(sumVal) {}
};

SegmentTreeBeat<Node, int64>::Configuration cfg{
    Node(-INF, 0),
    [](const Node& x, const Node& y) {
        if (x.maxVal == -INF) return y;
        if (y.maxVal == -INF) return x;
        return Node(max(x.maxVal, y.maxVal), x.sumVal + y.sumVal);
    },
    [](Node& x, int l, int r,  const int64& val) {
        if (x.maxVal < val) return true;
        if (l == r) {
            x.maxVal = x.sumVal = x.maxVal % val;
            return true;
        }
        return false;
    }
};

void solve() {
    cin >> N >> M;
    SegmentTreeBeat<Node, int64> seg(N, cfg);
    vector<Node> A(N);
    for (int i = 0; i < N; ++i) {
        int x;
        cin >> x;
        A[i] = Node(x, x);
    }
    seg.build(A);
    while (M--) {
        int t, l, r, k, x;
        cin >> t;
        if (t == 1) {
            cin >> l >> r;
            l--, r--;
            Node ans = seg.query(l, r);
            cout << ans.sumVal << endl;
        } else if (t == 2) {
            cin >> l >> r >> x;
            l--, r--;
            seg.update_range(l, r, x);
        } else {
            cin >> k >> x;
            k--;
            seg.update_point(k, Node(x, x));
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