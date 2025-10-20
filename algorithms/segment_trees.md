# SEGMENT TREES

![max_range_query](images/max_range_query.PNG)

## Fast Segment tree in C++ Point updates and Range Queries PURQ

This is a good implementation, been using it for many problems.

point update, range query

Implement the function in here, such as max for func, but it can be other functions.  and update each index value. 

0-indexed

Inclusive queries [left, right].  

this is the best one right now for C++

you can just get the value from seg.nodes[1] if you are querying the full range of the array.

```cpp

struct SegmentTree {
    int size;
    int neutral = 0;
    vector<int64> nodes;

    void init(int num_nodes) {
        size = 1;
        while (size < num_nodes) size *= 2;
        nodes.assign(size * 2, neutral);
    }

    int64 func(int64 x, int64 y) {
        return x + y;
    }

    void ascend(int segment_idx) {
        while (segment_idx > 0) {
            int left_segment_idx = 2 * segment_idx, right_segment_idx = 2 * segment_idx + 1;
            nodes[segment_idx] = func(nodes[left_segment_idx], nodes[right_segment_idx]);
            segment_idx >>= 1;
        }
    }
    // this is for assign, for addition change to += val
    void update(int segment_idx, int64 val) {
        segment_idx += size;
        nodes[segment_idx] = val; // += val if want addition, to track frequency
        segment_idx >>= 1;
        ascend(segment_idx);
    }

    int64 query(int left, int right) {
        left += size, right += size;
        int64 res = neutral;
        while (left <= right) {
           if (left & 1) {
                // res on left
                res = func(res, nodes[left]);
                left++;
            }
            if (~right & 1) {
                // res on right
                res = func(nodes[right], res);
                right--;
            }
            left >>= 1, right >>= 1;
        }
        return res;
    }
};
```

## Segment Tree for finding rightmost index with given prefix sum

1. prefix sum means you start from [0, r], and want the largest r such that it equals some target value.

```cpp
const int MAXN = 1e5 + 5, INF = 1e9;
struct SegmentTree {
    int n;
    int size;
    int neutral = 0;
    vector<int64> nodes, mn, mx;

    void init(int num_nodes) {
        n = num_nodes;
        size = 1;
        while (size < num_nodes) size *= 2;
        nodes.assign(size * 2, neutral);
        mn.assign(size * 2, INF);
        mx.assign(size * 2, -INF);
        for (int i = 0; i < n; ++i) {
            int segment_idx = size + i;
            mn[segment_idx] = 0;
            mx[segment_idx] = 0;
        }
        for (int segment_idx = size - 1; segment_idx >= 0; --segment_idx) {
            int left_segment_idx = 2 * segment_idx, right_segment_idx = 2 * segment_idx + 1;
            mn[segment_idx] = min(mn[left_segment_idx], mn[right_segment_idx]);
            mx[segment_idx] = max(mx[left_segment_idx], mx[right_segment_idx]);
        }
    }

    int64 func(int64 x, int64 y) {
        return x + y;
    }

    void ascend(int segment_idx) {
        while (segment_idx > 0) {
            int left_segment_idx = 2 * segment_idx, right_segment_idx = 2 * segment_idx + 1;
            nodes[segment_idx] = func(nodes[left_segment_idx], nodes[right_segment_idx]);
            mn[segment_idx] = min(mn[left_segment_idx], nodes[left_segment_idx] + mn[right_segment_idx]);
            mx[segment_idx] = max(mx[left_segment_idx], nodes[left_segment_idx] + mx[right_segment_idx]);
            segment_idx >>= 1;
        }
    }
    // this is for assign, for addition change to += val
    void update(int segment_idx, int64 val) {
        segment_idx += size;
        nodes[segment_idx] = val; // += val if want addition, to track frequency
        mn[segment_idx] = val;
        mx[segment_idx] = val;
        segment_idx >>= 1;
        ascend(segment_idx);
    }

    int64 query(int left, int right) {
        left += size, right += size;
        int64 res = neutral;
        while (left <= right) {
           if (left & 1) {
                // res on left
                res = func(res, nodes[left]);
                left++;
            }
            if (~right & 1) {
                // res on right
                res = func(nodes[right], res);
                right--;
            }
            left >>= 1, right >>= 1;
        }
        return res;
    }
    // Find the rightmost index r in [0..n-1] such that prefixSum(0..r) == target.
    // Returns -1 if no such index exists.
    int rightmostEqualPrefix(int64 target) const {
        if (target < mn[1] || target > mx[1]) return -1;
        int segment_idx = 1;
        int l = 0, r = size - 1;
        while (l != r) {
            int left_segment_idx = segment_idx << 1, right_segment_idx = left_segment_idx | 1;
            int mid = (l + r) >> 1;
            int64 tRight = target - nodes[left_segment_idx];

            // try the right child first
            if (mn[right_segment_idx] <= tRight && tRight <= mx[right_segment_idx]) {
                segment_idx = right_segment_idx;
                l = mid + 1;
                target = tRight;
            } else {
                segment_idx = left_segment_idx;
                r = mid;
            }
        }
        return (l < n ? l : -1);
    }
};
```

## Segment Tree for Many Values Per Index

Segment tree that supports add and remove operation or insert/erase.  
The difference from classic segment tree is that you will perform insert, erase of elements at an index. 
Leaf node will maintain an order bag of elements (multiset) and the min/max.  

0-indexed

Inclusive queries [left, right].  

Adjust this for something other than min range queries

```cpp
struct SegmentTree {
    int size;
    int64 neutral = INF;
    vector<int64> nodes;
    vector<multiset<int64>> leaves;

    void init(int num_nodes) {
        size = 1;
        while (size < num_nodes) size *= 2;
        nodes.assign(size * 2, neutral);
        leaves.assign(size * 2, multiset<int64>());
    }

    int64 func(int64 x, int64 y) {
        return min(x, y);
    }

    void ascend(int segment_idx) {
        while (segment_idx > 0) {
            int left_segment_idx = 2 * segment_idx, right_segment_idx = 2 * segment_idx + 1;
            nodes[segment_idx] = func(nodes[left_segment_idx], nodes[right_segment_idx]);
            segment_idx >>= 1;
        }
    }

    void add(int segment_idx, int64 val) {
        segment_idx += size;
        leaves[segment_idx].insert(val);
        nodes[segment_idx] = *leaves[segment_idx].begin(); // grab the min
        segment_idx >>= 1;
        ascend(segment_idx);
    }

    void remove(int segment_idx, int64 val) {
        segment_idx += size;
        leaves[segment_idx].erase(leaves[segment_idx].find(val));
        nodes[segment_idx] = leaves[segment_idx].empty() ? neutral : *leaves[segment_idx].begin();
        segment_idx >>= 1;
        ascend(segment_idx);
    }

    int64 query(int left, int right) {
        left += size, right += size;
        int64 res = neutral;
        while (left <= right) {
           if (left & 1) {
                // res on left
                res = func(res, nodes[left]);
                left++;
            }
            if (~right & 1) {
                // res on right
                res = func(nodes[right], res);
                right--;
            }
            left >>= 1, right >>= 1;
        }
        return res;
    }
};
```

## Segment Tree for Modular Product Frequency Queries

This algorithm implements a Segment Tree designed to efficiently handle queries and updates on an array of integers, where the focus is on modular arithmetic. Specifically, it supports:

Building a segment tree based on the modular remainder (mod K) of array elements.

Updating an element at a specific index.

Querying how many contiguous subarray products (within a range) result in a specific value modulo K.

Data Structures
struct Node
Each node in the segment tree stores:

vector<int> freq: Frequency count of all possible values modulo K. freq[i] is the number of subarrays in the node's range whose product modulo K equals i.

int prod: The total product of the range modulo K.

clear(): Resets the node (used during merging).

struct SegmentTree
Core segment tree with the following components:

int size: Size of the segment tree (next power of 2 ≥ array size).

int K: The modulus value used throughout.

vector<Node> tree: The segment tree array storing the Node objects.

```cpp
struct Node {
    vector<int> freq;
    int prod;
    Node(int k) : freq(k, 0), prod(1) {}
    void clear() {
        fill(freq.begin(), freq.end(), 0);
        prod = 1;
    }
};

struct SegmentTree {
    int size;
    int K;
    vector<Node> tree;
    void init(int num_nodes, int k) {
        K = k;
        size = 1;
        while (size < num_nodes) size *= 2;
        tree.assign(2 * size, Node(K));
    }
    void build(const vector<int> &nums) {
        int n = nums.size();
        for (int i = 0; i < size; i++) {
            int p = size + i;
            if (i < n) {
                int r = nums[i] % K;
                tree[p].prod = r;
                tree[p].freq[r] = 1;
            }
        }
        for (int i = size - 1; i >= 1; i--) {
            int l = 2 * i, r = 2 * i + 1;
            merge_(tree[i], tree[l], tree[r]);
        }
    }
    void merge_(Node &res, const Node& left, const Node& right) {
        for (int i = 0; i < K; i++) {
            res.freq[i] = left.freq[i];
        }
        for (int i = 0; i < K; i++) {
            res.freq[(i * left.prod) % K] += right.freq[i];
        }
        res.prod = left.prod * right.prod % K;
    }
    void ascend(int segment_idx) {
        while (segment_idx > 0) {
            int left_segment_idx = 2 * segment_idx, right_segment_idx = 2 * segment_idx + 1;
            merge_(tree[segment_idx], tree[left_segment_idx], tree[right_segment_idx]);
            segment_idx >>= 1;
        }
    }
    void update(int segment_idx, int val) {
        segment_idx += size;
        tree[segment_idx].prod = val % K;
        fill(tree[segment_idx].freq.begin(), tree[segment_idx].freq.end(), 0);
        tree[segment_idx].freq[val % K] = 1;
        segment_idx >>= 1;
        ascend(segment_idx);
    }
    int query(int left, int right, int k) {
        left += size, right += size;
        Node leftRes(K), rightRes(K), result(K);
        while (left <= right) {
            if (left & 1) {
                // res on left
                result.clear();
                merge_(result, leftRes, tree[left]);
                swap(result, leftRes);
                left++;
            }
            if (~right & 1) {
                // res on right
                result.clear();
                merge_(result, tree[right], rightRes);
                swap(result, rightRes);
                right--;
            }
            left >>= 1, right >>= 1;
        }
        result.clear();
        merge_(result, leftRes, rightRes);
        return result.freq[k];
    }
};
```

## Segment tree for mex

It can deal with updates to values in an array and find the mex for each query. 

```py
class SegmentTree:
    def __init__(self, n: int, neutral: int, func):
        self.func = func
        self.neutral = neutral
        self.size = 1
        self.n = n
        while self.size<n:
            self.size*=2
        self.nodes = [neutral for _ in range(self.size*2)]
    def ascend(self, segment_idx: int) -> None:
        while segment_idx > 0:
            segment_idx -= 1
            segment_idx >>= 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.nodes[segment_idx] = self.func(self.nodes[left_segment_idx], self.nodes[right_segment_idx]) 
    def update(self, segment_idx: int, val: int) -> None:
        segment_idx += self.size - 1
        self.nodes[segment_idx] += val
        self.ascend(segment_idx)
    def query_mex(self) -> int:
        segment_left_bound, segment_right_bound, segment_idx = 0, self.size, 0
        while segment_left_bound + 1 < segment_right_bound:
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2 * segment_idx + 1, 2 * segment_idx + 2
            child_segment_len = (segment_right_bound - segment_left_bound) >> 1
            if self.nodes[left_segment_idx] < child_segment_len:
                segment_idx = left_segment_idx
                segment_right_bound = mid_point
            else:
                segment_idx = right_segment_idx
                segment_left_bound = mid_point
        return segment_left_bound
    def __repr__(self) -> str:
        return f"nodes array: {self.nodes}, next array: {self.nodes}"

freq = Counter()
summation = lambda x, y: x + y
seg = SegmentTree(MAXN + 1, 0, summation)  
for num in arr:
    freq[num] += 1
    if freq[num] == 1 and num <= MAXN:
        seg.update(num, 1)
```

## Segment tree for Point updates and full range queries

The basic difference is that when you perform full range query, you can just get the value from seg.nodes[1]

This one shows one with an interesting merge methodology.

This is for a problem where you have a dynamic programming solution, but you are updating values, which you can then update the rest of the tree and propagate to the root or full segment by the fact you can merge two segments with some arithmetic.

Just a tip you can implement this with a Node, or you can use arrays to represent these, I think it is easier to actually use separate arrays to represent the node. 

```cpp
struct Node {
    int cumsum_prefix_products, cumsum_suffix_products, prefix_product, exp;
};

struct SegmentTree {
    int size;
    vector<Node> nodes;
    SegmentTree(int n) {
        size = 1;
        while (size < n) size *= 2;
        nodes.assign(size * 2, {-INF, INF, -INF, INF, 0});
    }
    Node func(Node x, Node y) {
        Node res;
        res.cumsum_prefix_products = (x.cumsum_prefix_products + (y.cumsum_prefix_products * x.prefix_product) % MOD) % MOD;
        res.prefix_product = (x.prefix_product * y.prefix_product) % MOD;
        res.cumsum_suffix_products = ((x.cumsum_suffix_products * y.prefix_product) % MOD + y.cumsum_suffix_products) % MOD;
        res.exp = (x.exp + y.exp + (x.cumsum_suffix_products * y.cumsum_prefix_products) % MOD) % MOD;
        return res;
    }
    void ascend(int segment_idx) {
        while (segment_idx > 0) {
            int left_segment_idx = 2 * segment_idx, right_segment_idx = 2 * segment_idx + 1;
            nodes[segment_idx] = func(nodes[left_segment_idx], nodes[right_segment_idx]);
            segment_idx >>= 1;
        }
    }
    void update(int segment_idx, int val) {
        segment_idx += size;
        nodes[segment_idx] = {val, val, val, val};
        segment_idx >>= 1;
        ascend(segment_idx);
    }
};
```

### Alternative approach, when storing multiple variables in segment tree

Still for inclusive range queries [L, R], returns minimum and the smallest index at which the min occurs.


```cpp
struct MinSegmentTree {
    int size;
    vector<int> nodes, index;
    int neutral = INF;

    void init(int num_nodes) {
        size = 1;
        while (size < num_nodes) size *= 2;
        nodes.assign(size * 2, neutral);
        index.assign(size * 2, neutral);
    }

    int func(int x, int y) {
        return min(x, y);
    }

    void ascend(int segment_idx) {
        while (segment_idx > 0) {
            int left_segment_idx = 2 * segment_idx, right_segment_idx = 2 * segment_idx + 1;
            int left_min = nodes[left_segment_idx], right_min = nodes[right_segment_idx];
            if (left_min < right_min) index[segment_idx] = index[left_segment_idx];
            else index[segment_idx] = index[right_segment_idx];
            nodes[segment_idx] = func(nodes[left_segment_idx], nodes[right_segment_idx]);
            segment_idx >>= 1;
        }
    }

    void update(int segment_idx, int idx, int val) {
        segment_idx += size;
        nodes[segment_idx] = val;
        index[segment_idx] = idx;
        segment_idx >>= 1;
        ascend(segment_idx);
    }

    pair<int, int> query(int left, int right) {
        left += size, right += size;
        int res = neutral, idx = INF;
        while (left <= right) {
            if (left & 1) {
                if (nodes[left] == res) idx = min(idx, index[left]);
                if (nodes[left] < res) {
                    idx = index[left];
                    res = nodes[left];
                }
                left++;
            }
            if (~right & 1) {
                if (nodes[right] == res) idx = min(idx, index[right]);
                if (nodes[right] < res) {
                    res = nodes[right];
                    idx = index[right];
                }
                right--;
            }
            left >>= 1, right >>= 1;
        }
        return make_pair(res, idx);
    }
};
```

## segment tree

Base segment tree that takes in a function and a neutral element and some initial array

Just note for range queries it is querying the range [L, R), so that means it does not include R.  So you are actually searching the range [L, R - 1]

```py
class SegmentTree:
    def __init__(self, n, neutral, func, initial_arr):
        self.func = func
        self.neutral = neutral
        self.size = 1
        self.n = n
        while self.size<n:
            self.size*=2
        self.nodes = [neutral for _ in range(self.size*2)] 
        self.build(initial_arr)

    def build(self, initial_arr):
        for i, segment_idx in enumerate(range(self.n)):
            segment_idx += self.size - 1
            val = initial_arr[i]
            self.nodes[segment_idx] = val
            self.ascend(segment_idx)

    def ascend(self, segment_idx):
        while segment_idx > 0:
            segment_idx -= 1
            segment_idx >>= 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.nodes[segment_idx] = self.func(self.nodes[left_segment_idx], self.nodes[right_segment_idx])
        
    def update(self, segment_idx, val):
        segment_idx += self.size - 1
        self.nodes[segment_idx] = val
        self.ascend(segment_idx)
            
    def query(self, left, right):
        stack = [(0, self.size, 0)]
        result = self.neutral
        while stack:
            # BOUNDS FOR CURRENT INTERVAL and idx for tree
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # COMPLETE OVERLAP
            if segment_left_bound >= left and segment_right_bound <= right:
                result = self.func(result, self.nodes[segment_idx])
                continue
            # PARTIAL OVERLAP
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
        return result
    
    def __repr__(self):
        return f"nodes array: {self.nodes}"
```

segment tree without initial array

```py
class SegmentTree:
    def __init__(self, n: int, neutral: int, func):
        self.func = func
        self.neutral = neutral
        self.size = 1
        self.n = n
        while self.size<n:
            self.size*=2
        self.nodes = [neutral for _ in range(self.size*2)]

    def ascend(self, segment_idx: int) -> None:
        while segment_idx > 0:
            segment_idx -= 1
            segment_idx >>= 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.nodes[segment_idx] = self.func(self.nodes[left_segment_idx], self.nodes[right_segment_idx])
        
    def update(self, segment_idx: int, val: int) -> None:
        segment_idx += self.size - 1
        self.nodes[segment_idx] = val
        self.ascend(segment_idx)
            
    def query(self, left: int, right: int) -> int:
        stack = [(0, self.size, 0)]
        result = self.neutral
        while stack:
            # BOUNDS FOR CURRENT INTERVAL and idx for tree
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # COMPLETE OVERLAP
            if segment_left_bound >= left and segment_right_bound <= right:
                result = self.func(result, self.nodes[segment_idx])
                continue
            # PARTIAL OVERLAP
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
        return result
    
    def __repr__(self) -> str:
        return f"nodes array: {self.nodes}"
```

This variation was needed for a problem where you want to always keep the minimum or maximum value of an integer at an index in a segment tree, so that means the update function is updating based on the self.func(cur, val)

```py
class SegmentTree:
    def __init__(self, n: int, neutral: int, func):
        self.func = func
        self.neutral = neutral
        self.size = 1
        self.n = n
        while self.size<n:
            self.size*=2
        self.nodes = [neutral for _ in range(self.size*2)]

    def ascend(self, segment_idx: int) -> None:
        while segment_idx > 0:
            segment_idx -= 1
            segment_idx >>= 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.nodes[segment_idx] = self.func(self.nodes[left_segment_idx], self.nodes[right_segment_idx])
        
    def update(self, segment_idx: int, val: int) -> None:
        segment_idx += self.size - 1
        self.nodes[segment_idx] = self.func(self.nodes[segment_idx], val)
        self.ascend(segment_idx)
            
    def query(self, left: int, right: int) -> int:
        stack = [(0, self.size, 0)]
        result = self.neutral
        while stack:
            # BOUNDS FOR CURRENT INTERVAL and idx for tree
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # COMPLETE OVERLAP
            if segment_left_bound >= left and segment_right_bound <= right:
                result = self.func(result, self.nodes[segment_idx])
                continue
            # PARTIAL OVERLAP
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
        return result
    
    def __repr__(self) -> str:
        return f"nodes array: {self.nodes}, next array: {self.nodes}"
```

## Segment tree with line sweep for area of union of rectangles

This is a segment tree that is used to track which segments are currently covered and efficiently compute the union of covered x-intervals.

line sweeps over the y-coordinate events. able to track the width of each region covered by squares with the segment tree. And you know the current height based on previous y and current y value. 

Example problem with how it is used to calculate the area of the union of rectangles on a 2d plane. 

The union of rectangles refers to the total area covered by multiple rectangles, where some rectangles may overlap with each other. The goal is to compute the unique area occupied by at least one rectangle, without counting overlapping regions multiple times.

```cpp
using int64 = long long;
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

class Solution {
public:
    double separateSquares(vector<vector<int>>& squares) {
        int N = squares.size();
        vector<int64> xs;
        for (const vector<int>& square : squares) {
            int x = square[0], y = square[1], l = square[2];
            xs.emplace_back(x);
            xs.emplace_back(x + l);
        }
        SegmentTree st(xs);
        vector<Event> events;
        for (const vector<int>& square : squares) {
            int x = square[0], y = square[1], l = square[2];
            events.emplace_back(y, 1, x, x + l);
            events.emplace_back(y + l, -1, x, x + l);
        }
        sort(events.begin(), events.end());
        int64 prevY = -INF;
        long double totalArea = 0;
        for (const Event& event : events) {
            long double dh = event.v - prevY;
            long double dw = st.query();
            totalArea += dh * dw;
            st.update(event.l, event.r, event.t);
            prevY = event.v;
        }
        prevY = -INF;
        long double curSumArea = 0;
        for (const Event& event : events) {
            long double dh = event.v - prevY;
            long double dw = st.query();
            long double area = dh * dw;
            if (2 * (area + curSumArea) >= totalArea) {
                return (totalArea / 2.0 - curSumArea) / dw + prevY;
            }
            curSumArea += area;
            st.update(event.l, event.r, event.t);
            prevY = event.v;
        }
        return curSumArea;
    }
};

```
