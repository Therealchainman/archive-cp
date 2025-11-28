# Segment Tree Beats

This is a variant of segment tree that can handle range updates and range queries for certain problems that normal segment trees and lazy segment trees cannot handle. Works under particular conditions. 

segment tree beats is a technique that allows a non-polylogarithmic range update complexity that amortizes to $\mathcal{O}(n \log n)$ or $\mathcal{O}(n \log^2 n)$

Segment tree beats uses the fact that certain updates strictly improve some "measure" of the data and cannot keep doing heavy work forever.


## Implementation in C++

This is for inclusive range updates and queries, [l, r].

Update is the type that represents one range update operation.

You need this because the update is different from the node, it is what will apply to a node and not necessarily a leaf. 

```cpp
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
```

## Example Usage

```cpp
const int64 INF = numeric_limits<int64>::max();
int N, M;

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
/*
vector<Node> A(N);
SegmentTreeBeat<Node, int64> seg(N, cfg);
seg.build(A);
*/
```