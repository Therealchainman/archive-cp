# Lazy Segment Tree

## Lazy Segment Tree range queries and range updates

- range updates
- range queries

This particular implementation is for range addition updates, but can easily modify it for other types of updates and queries

- range addition updates
- range sum queries

range updates are [L, R) (exclusive for right end point)

```cpp
struct LazySegmentTree {
    vector<int> values;
    vector<int> operations;
    int size;
    int neutral = 0, noop = 0;

    void init(int n) {
        size = 1;
        while (size < n) size *= 2;
        values.assign(2 * size, neutral);
        operations.assign(2 * size, noop);
    }

    void build(const vector<int>& arr) {
        copy(arr.begin(), arr.end(), values.begin() + size);
        for (int i = size - 1; i > 0; i--) {
            values[i] = values[2 * i] + values[2 * i + 1];
        }
    }

    int modify_op(int x, int y, int length = 1) {
        return x + y * length;
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
```

## Lazy Segment Tree point queries and range updates

- range updates
- point queries

This particular implementation is for range addition updates

- range addition updates
- point queries

range updates are [L, R) (exclusive for right end point)

- Initialization array can speed up it significantly, recommended to use that. 

```cpp
struct LazySegmentTree {
    vector<int> values;
    int size, noop = 0;

    void init(int n) {
        size = 1;
        while (size < n) size *= 2;
        values.assign(2 * size, noop);
    }

    void build(const vector<int>& arr) {
        copy(arr.begin(), arr.end(), values.begin() + size);
    }

    bool is_leaf(int segment_right_bound, int segment_left_bound) {
        return segment_right_bound - segment_left_bound == 1;
    }

    void propagate(int segment_idx, int segment_left_bound, int segment_right_bound) {
        if (is_leaf(segment_right_bound, segment_left_bound) || values[segment_idx] == noop) return;
        int left_segment_idx = 2 * segment_idx + 1, right_segment_idx = 2 * segment_idx + 2;
        values[left_segment_idx] += values[segment_idx];
        values[right_segment_idx] += values[segment_idx];
        values[segment_idx] = noop;
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
                values[segment_idx] += val;
                continue;
            }
            // PARTIAL OVERLAP
            int mid_point = (segment_left_bound + segment_right_bound) >> 1;
            int left_segment_idx = 2 * segment_idx + 1, right_segment_idx = 2 * segment_idx + 2;
            propagate(segment_idx, segment_left_bound, segment_right_bound);
            stk.emplace(mid_point, segment_right_bound, right_segment_idx);
            stk.emplace(segment_left_bound, mid_point, left_segment_idx);
        }
    }
    int query(int i) {
        stack<tuple<int, int, int>> stk;
        stk.emplace(0, size, 0);
        int segment_left_bound, segment_right_bound, segment_idx;
        while (!stk.empty()) {
            tie(segment_left_bound, segment_right_bound, segment_idx) = stk.top();
            stk.pop();
            // NO OVERLAP
            if (i < segment_left_bound || i >= segment_right_bound) continue;
            // COMPLETE OVERLAP
            if (is_leaf(segment_right_bound, segment_left_bound)) return values[segment_idx];
            // PARTIAL OVERLAP
            int mid_point = (segment_left_bound + segment_right_bound) >> 1;
            int left_segment_idx = 2 * segment_idx + 1, right_segment_idx = 2 * segment_idx + 2;
            propagate(segment_idx, segment_left_bound, segment_right_bound);
            stk.emplace(mid_point, segment_right_bound, right_segment_idx);
            stk.emplace(segment_left_bound, mid_point, left_segment_idx);
        }
        return -1;
    }
};
```

## Lazy Segment tree assign value on segment and calculate minimum on segments

update by assigning values in a range
query by the minimum in a range

```py
class LazySegmentTree:
    def __init__(self, n: int, neutral: int, noop: int, initial_val: int = 0):
        self.neutral = neutral
        self.size = 1
        self.noop = noop
        self.n = n 
        while self.size<n:
            self.size*=2
        self.operations = [noop for _ in range(self.size*2)]
        self.values = [initial_val for _ in range(self.size*2)]

    def modify_op(self, v: int, segment_len: int = 1) -> int:
        return v*segment_len

    def calc_op(self, x: int, y: int) -> int:
        return min(x, y)

    def is_leaf_node(self, segment_right_bound, segment_left_bound) -> bool:
        return segment_right_bound - segment_left_bound == 1

    def propagate(self, segment_idx: int, segment_left_bound: int, segment_right_bound: int) -> None:
        if self.is_leaf_node(segment_right_bound, segment_left_bound) or self.operations[segment_idx] == self.noop: return
        left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
        self.operations[left_segment_idx] = self.modify_op(self.operations[segment_idx], 1)
        self.operations[right_segment_idx] = self.modify_op(self.operations[segment_idx], 1)
        self.values[left_segment_idx] = self.modify_op(self.operations[segment_idx], 1)
        self.values[right_segment_idx] = self.modify_op(self.operations[segment_idx], 1)
        self.operations[segment_idx] = self.noop

    def ascend(self, segment_idx: int) -> None:
        while segment_idx > 0:
            segment_idx -= 1
            segment_idx >>= 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.values[segment_idx] = self.calc_op(self.values[left_segment_idx], self.values[right_segment_idx])

    def update(self, left: int, right: int, val: int) -> None:
        stack = [(0, self.size, 0)]
        segments = []
        while stack:
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # COMPLETE OVERLAP
            if segment_left_bound >= left and segment_right_bound <= right:
                self.operations[segment_idx] = self.modify_op(val, 1)
                self.values[segment_idx] = self.modify_op(val, 1)
                segments.append(segment_idx)
                continue
            # PARTIAL OVERLAP
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.propagate(segment_idx, segment_left_bound, segment_right_bound)
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
        for segment_idx in segments:
            self.ascend(segment_idx)

    def query(self, left: int, right: int) -> int:
        stack = [(0, self.size, 0)]
        result = self.neutral
        while stack:
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # COMPLETE OVERLAP
            if segment_left_bound >= left and segment_right_bound <= right:
                result = self.calc_op(result, self.values[segment_idx])
                continue
            # PARTIAL OVERLAP
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2    
            self.propagate(segment_idx, segment_left_bound, segment_right_bound)      
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
        return result
    
    def __repr__(self) -> str:
        return f"values: {self.values}, operations: {self.operations}"
```

## Lazy Segment Tree with Nodes

This particular one calculates the maximum subarray in queries with range assignment updates.

```py
# node represents a segment
class Node:
    def __init__(self):
        self.pref = 0
        self.suf = 0
        self.msum = 0
        self.sum = 0

    def __repr__(self):
        return f"pref: {self.pref}, suf: {self.suf}, msum: {self.msum}, sum: {self.sum}"

class LazySegmentTree:
    def __init__(self, n: int, noop: int):
        self.size = 1
        self.noop = noop
        self.n = n 
        while self.size<n:
            self.size*=2
        self.operations = [noop for _ in range(self.size*2)]
        self.values = [Node() for _ in range(self.size * 2)]

    def modify_op(self, node, v, segment_len):
        if v > 0:
            node.pref = node.suf = node.msum = node.sum = v * segment_len
        else:
            node.pref = node.suf = node.msum = 0
            node.sum = v * segment_len

    def merge(self, node, lnode, rnode):
        node.pref = max(lnode.pref, lnode.sum + rnode.pref)
        node.suf = max(rnode.suf, lnode.suf + rnode.sum)
        node.sum = lnode.sum + rnode.sum
        node.msum = max(lnode.msum, rnode.msum, lnode.suf + rnode.pref)

    def is_leaf_node(self, segment_right_bound, segment_left_bound) -> bool:
        return segment_right_bound - segment_left_bound == 1

    def propagate(self, segment_idx: int, segment_left_bound: int, segment_right_bound: int) -> None:
        if self.is_leaf_node(segment_right_bound, segment_left_bound) or self.operations[segment_idx] == self.noop: return
        left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
        children_segment_len = (segment_right_bound - segment_left_bound) >> 1
        self.operations[left_segment_idx] = self.operations[segment_idx]
        self.operations[right_segment_idx] = self.operations[segment_idx]
        self.modify_op(self.values[left_segment_idx], self.operations[segment_idx], children_segment_len)
        self.modify_op(self.values[right_segment_idx], self.operations[segment_idx], children_segment_len)
        self.operations[segment_idx] = self.noop

    def ascend(self, segment_idx: int) -> None:
        while segment_idx > 0:
            segment_idx -= 1
            segment_idx >>= 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.merge(self.values[segment_idx], self.values[left_segment_idx], self.values[right_segment_idx])

    def update(self, left: int, right: int, val: int) -> None:
        stack = [(0, self.size, 0)]
        segments = []
        while stack:
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # COMPLETE OVERLAP
            if segment_left_bound >= left and segment_right_bound <= right:
                self.operations[segment_idx] = val
                segment_len = segment_right_bound - segment_left_bound
                self.modify_op(self.values[segment_idx], val, segment_len)
                segments.append(segment_idx)
                continue
            # PARTIAL OVERLAP
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.propagate(segment_idx, segment_left_bound, segment_right_bound)
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
        for segment_idx in segments:
            self.ascend(segment_idx)

    def query(self, left: int, right: int) -> int:
        stack = [(0, self.size, 0)]
        result = Node()
        while stack:
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # LEAF NODE
            if segment_left_bound >= left and segment_right_bound <= right:
                cur = Node()
                self.merge(cur, result, self.values[segment_idx])
                result = cur
                continue
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2    
            self.propagate(segment_idx, segment_left_bound, segment_right_bound)      
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
        return result.msum
    
    def __repr__(self) -> str:
        return f"values: {self.values}, operations: {self.operations}"
```


## Example of lazy segment tree

1. range queries
1. range updates

The counts is update based on the size of the segment and is used to store the number of 1 bits.
number of slips just need to be a 1 or 0 to indicate if a flip is necessary or not on a segment

This lazy segment tree is used to allow to flip a range which is the range update. and then to query on a range to 
get the number of 1 bits in that range. 


```py
class LazySegmentTree:
    def __init__(self, n: int, neutral: int, noop: int, arr: List[int]):
        self.neutral = neutral
        self.size = 1
        self.noop = noop
        self.n = n 
        self.arr = arr
        while self.size<n:
            self.size*=2
        self.num_flips = [noop for _ in range(self.size*2)] # number of flips in a segment
        self.counts = [neutral for _ in range(self.size*2)] # count of ones in a segment
        self.build()
        
    def build(self):
        for segment_idx in range(self.n):
            v = self.arr[segment_idx]
            segment_idx += self.size - 1
            self.counts[segment_idx] = v
            self.ascend(segment_idx)

    def is_leaf_node(self, segment_right_bound, segment_left_bound) -> bool:
        return segment_right_bound - segment_left_bound == 1
    
    def calc_op(self, x: int, y: int) -> int:
        return x + y

    def modify_op(self, x: int, y: int) -> int:
        return x ^ y
    
    """
    Gives the count of a bit in a segment, which is a range. and the length of that range is represented by the segment_len.
    And it flips all the bits such as 0000110 -> 1111001, the number of 1s are now segment_len - cnt, where cnt is the current number of 1s
    So it goes from 2 -> 7 - 2 = 5
    """
    def flip_op(self, segment_len: int, cnt: int) -> int:
        return segment_len - cnt

    def propagate(self, segment_idx: int, segment_left_bound: int, segment_right_bound: int) -> None:
        # do not want to propagate if it is a leaf node or if it is no operation (means there are no updates stored there).
        if self.is_leaf_node(segment_right_bound, segment_left_bound) or self.num_flips[segment_idx] == self.noop: return
        left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
        children_segment_len = (segment_right_bound - segment_left_bound) >> 1
        self.num_flips[left_segment_idx] = self.modify_op(self.num_flips[left_segment_idx], self.num_flips[segment_idx])
        self.num_flips[right_segment_idx] = self.modify_op(self.num_flips[right_segment_idx], self.num_flips[segment_idx])
        self.counts[left_segment_idx] = self.flip_op(children_segment_len, self.counts[left_segment_idx])
        self.counts[right_segment_idx] = self.flip_op(children_segment_len, self.counts[right_segment_idx])
        self.num_flips[segment_idx] = self.noop
    
    def ascend(self, segment_idx: int) -> None:
        while segment_idx > 0:
            segment_idx -= 1
            segment_idx >>= 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.counts[segment_idx] = self.calc_op(self.counts[left_segment_idx], self.counts[right_segment_idx])
    
    def update(self, left: int, right: int, val: int) -> None:
        stack = [(0, self.size, 0)]
        segments = []
        while stack:
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # COMPLETE OVERLAP
            if segment_left_bound >= left and segment_right_bound <= right:
                self.num_flips[segment_idx] = self.modify_op(self.num_flips[segment_idx], val)
                segment_len = segment_right_bound - segment_left_bound
                self.counts[segment_idx] = self.flip_op(segment_len, self.counts[segment_idx])
                segments.append(segment_idx)
                continue
            # PARTIAL OVERLAP
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.propagate(segment_idx, segment_left_bound, segment_right_bound)
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
        for segment_idx in segments:
            self.ascend(segment_idx)

    def query(self, left: int, right: int) -> int:
        stack = [(0, self.size, 0)]
        result = self.neutral
        while stack:
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # LEAF NODE
            if segment_left_bound >= left and segment_right_bound <= right:
                result = self.calc_op(result, self.counts[segment_idx])
                continue
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2    
            self.propagate(segment_idx, segment_left_bound, segment_right_bound)      
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
        return result
    
    def __repr__(self) -> str:
        return f"counts: {self.counts}, num_flips: {self.num_flips}"
```

## Lazy Segment Tree applied to finding the kth one

The kquery function can probably be used for solving many other problems involving finding the kth element?

```py
class LazySegmentTree:
    def __init__(self, n: int, neutral: int, noop: int, initial_val: int = 0):
        self.neutral = neutral
        self.size = 1
        self.noop = noop
        self.n = n 
        while self.size<n:
            self.size*=2
        self.operations = [noop for _ in range(self.size*2)]
        self.values = [initial_val for _ in range(self.size*2)]

    def modify_op(self, x: int, y: int) -> int:
        return x ^ y

    def calc_op(self, x: int, y: int) -> int:
        return x + y
    
    """
    Gives the count of a bit in a segment, which is a range. and the length of that range is represented by the segment_len.
    And it flips all the bits such as 0000110 -> 1111001, the number of 1s are now segment_len - cnt, where cnt is the current number of 1s
    So it goes from 2 -> 7 - 2 = 5
    """
    def flip_op(self, segment_len: int, cnt: int) -> int:
        return segment_len - cnt

    def is_leaf_node(self, segment_right_bound, segment_left_bound) -> bool:
        return segment_right_bound - segment_left_bound == 1

    def propagate(self, segment_idx: int, segment_left_bound: int, segment_right_bound: int) -> None:
        if self.is_leaf_node(segment_right_bound, segment_left_bound) or self.operations[segment_idx] == self.noop: return
        left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
        children_segment_len = (segment_right_bound - segment_left_bound) >> 1
        self.operations[left_segment_idx] = self.modify_op(self.operations[left_segment_idx], self.operations[segment_idx])
        self.operations[right_segment_idx] = self.modify_op(self.operations[right_segment_idx], self.operations[segment_idx])
        self.values[left_segment_idx] = self.flip_op(children_segment_len, self.values[left_segment_idx])
        self.values[right_segment_idx] = self.flip_op(children_segment_len, self.values[right_segment_idx])
        self.operations[segment_idx] = self.noop

    def ascend(self, segment_idx: int) -> None:
        while segment_idx > 0:
            segment_idx -= 1
            segment_idx >>= 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.values[segment_idx] = self.calc_op(self.values[left_segment_idx], self.values[right_segment_idx])

    def update(self, left: int, right: int, val: int) -> None:
        stack = [(0, self.size, 0)]
        segments = []
        while stack:
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # COMPLETE OVERLAP
            if segment_left_bound >= left and segment_right_bound <= right:
                self.operations[segment_idx] = self.modify_op(self.operations[segment_idx], val)
                segment_len = segment_right_bound - segment_left_bound
                # print("segment_len", segment_len)
                self.values[segment_idx] = self.flip_op(segment_len, self.values[segment_idx])
                segments.append(segment_idx)
                continue
            # PARTIAL OVERLAP
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.propagate(segment_idx, segment_left_bound, segment_right_bound)
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
        for segment_idx in segments:
            self.ascend(segment_idx)
    
    def kquery(self, k: int) -> int:
        segment_left_bound, segment_right_bound, segment_idx = 0, self.size, 0
        while segment_left_bound + 1 < segment_right_bound:
            self.propagate(segment_idx, segment_left_bound, segment_right_bound)
            left_segment_idx, right_segment_idx = 2 * segment_idx + 1, 2 * segment_idx + 2
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            segment_left_count = self.values[left_segment_idx]
            if segment_left_count >= k:
                segment_right_bound = mid_point
                segment_idx = left_segment_idx
            else:
                k -= segment_left_count
                segment_left_bound = mid_point
                segment_idx = right_segment_idx
        return segment_left_bound
    
    def __repr__(self) -> str:
        return f"values: {self.values}, operations: {self.operations}"
```

## Lazy Segment tree with range addition, and range assignment and range queries for sum. 

```py
class LazySegmentTree:
    def __init__(self, n: int, neutral: int, initial_val: int = 0):
        self.neutral = neutral
        self.size = 1
        self.add_noop = 0
        self.assign_noop = -1
        self.n = n 
        while self.size<n:
            self.size*=2
        self.assign_operations = [self.assign_noop for _ in range(self.size * 2)]
        self.add_operations = [self.add_noop for _ in range(self.size * 2)]
        self.values = [initial_val for _ in range(self.size * 2)]
    
    def assign_op(self, v: int, segment_len: int) -> int:
        return v * segment_len
    
    def add_op(self, x: int, v: int, segment_len: int) -> int:
        return x + v * segment_len

    def calc_op(self, x: int, y: int) -> int:
        return x + y

    def is_leaf_node(self, segment_right_bound, segment_left_bound) -> bool:
        return segment_right_bound - segment_left_bound == 1

    def propagate(self, segment_idx: int, segment_left_bound: int, segment_right_bound: int) -> None:
        if self.is_leaf_node(segment_right_bound, segment_left_bound):  return
        left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
        children_segment_len = (segment_right_bound - segment_left_bound) >> 1
        if self.assign_operations[segment_idx] != self.assign_noop:
            self.assign_operations[left_segment_idx] = self.assign_op(self.assign_operations[segment_idx], 1)
            self.assign_operations[right_segment_idx] = self.assign_op(self.assign_operations[segment_idx], 1)
            self.values[left_segment_idx] = self.assign_op(self.assign_operations[segment_idx], children_segment_len)
            self.values[right_segment_idx] = self.assign_op(self.assign_operations[segment_idx], children_segment_len)
            self.add_operations[left_segment_idx] = self.add_operations[right_segment_idx] = self.add_noop
        if self.add_operations[segment_idx] != self.add_noop:
            self.add_operations[left_segment_idx] = self.add_op(self.add_operations[left_segment_idx], self.add_operations[segment_idx], 1)
            self.add_operations[right_segment_idx] = self.add_op(self.add_operations[right_segment_idx], self.add_operations[segment_idx], 1)
            self.values[left_segment_idx] = self.add_op(self.values[left_segment_idx], self.add_operations[segment_idx], children_segment_len)
            self.values[right_segment_idx] = self.add_op(self.values[right_segment_idx], self.add_operations[segment_idx], children_segment_len)
        self.add_operations[segment_idx] = self.add_noop
        self.assign_operations[segment_idx] = self.assign_noop

    def ascend(self, segment_idx: int) -> None:
        while segment_idx > 0:
            segment_idx -= 1
            segment_idx >>= 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.values[segment_idx] = self.calc_op(self.values[left_segment_idx], self.values[right_segment_idx])

    def add(self, left: int, right: int, val: int) -> None:
        stack = [(0, self.size, 0)]
        segments = []
        while stack:
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # COMPLETE OVERLAP
            if segment_left_bound >= left and segment_right_bound <= right:
                self.add_operations[segment_idx] = self.add_op(self.add_operations[segment_idx], val, 1)
                segment_len = segment_right_bound - segment_left_bound
                self.values[segment_idx] = self.add_op(self.values[segment_idx], val, segment_len)
                segments.append(segment_idx)
                continue
            # PARTIAL OVERLAP
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.propagate(segment_idx, segment_left_bound, segment_right_bound)
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
        for segment_idx in segments:
            self.ascend(segment_idx)

    def assign(self, left: int, right: int, val: int) -> None:
        stack = [(0, self.size, 0)]
        segments = []
        while stack:
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # COMPLETE OVERLAP
            if segment_left_bound >= left and segment_right_bound <= right:
                self.add_operations[segment_idx] = self.add_noop
                self.assign_operations[segment_idx] = self.assign_op(val, 1)
                segment_len = segment_right_bound - segment_left_bound
                self.values[segment_idx] = self.assign_op(val, segment_len)
                segments.append(segment_idx)
                continue
            # PARTIAL OVERLAP
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.propagate(segment_idx, segment_left_bound, segment_right_bound)
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
        for segment_idx in segments:
            self.ascend(segment_idx)

    def query(self, left: int, right: int) -> int:
        stack = [(0, self.size, 0)]
        result = self.neutral
        while stack:
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # COMPLETE OVERLAP
            if segment_left_bound >= left and segment_right_bound <= right:
                result = self.calc_op(result, self.values[segment_idx])
                continue
            # PARTIAL OVERLAP
            # [L, M), [M, R)
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            # need to early propagate down the tree
            # pushes it down into left and right children for visiting those segments next
            self.propagate(segment_idx, segment_left_bound, segment_right_bound)      
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
        return result
    
    def __repr__(self) -> str:
        return f"values: {self.values}, add operations: {self.add_operations}, assign operations: {self.assign_operations}"
```

## Lazy Segment tree for finding the first element above value x in a range from L to end of array

```py
class LazySegmentTree:
    def __init__(self, n: int, neutral: int, noop: int, initial_val: int = 0):
        self.neutral = neutral
        self.size = 1
        self.noop = noop
        self.n = n
        self.initial_val = initial_val
        while self.size<n:
            self.size*=2
        self.operations = [noop for _ in range(self.size*2)]
        self.values = [initial_val for _ in range(self.size*2)]

    def calc_op(self, x: int, y: int) -> int:
        return max(x, y)

    def is_leaf(self, segment_left_bound, segment_right_bound) -> bool:
        return segment_right_bound - segment_left_bound == 1

    def ascend(self, segment_idx: int) -> None:
        while segment_idx > 0:
            segment_idx -= 1
            segment_idx >>= 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.values[segment_idx] = self.calc_op(self.values[left_segment_idx], self.values[right_segment_idx])
            self.values[segment_idx] += self.operations[segment_idx]

    def propagate(self, segment_idx, segment_left_bound, segment_right_bound):
        if self.is_leaf(segment_left_bound, segment_right_bound) or self.operations[segment_idx] == self.noop: return
        left_segment_idx = 2 * segment_idx + 1
        right_segment_idx = 2 * segment_idx + 2
        self.operations[left_segment_idx] += self.operations[segment_idx];
        self.operations[right_segment_idx] += self.operations[segment_idx];
        self.values[left_segment_idx] += self.operations[segment_idx];
        self.values[right_segment_idx] += self.operations[segment_idx];
        self.operations[segment_idx] = self.noop;

    def update(self, left: int, right: int, val: int) -> None:
        stack = [(0, self.size, 0)]
        segments = []
        while stack:
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # COMPLETE OVERLAP
            if segment_left_bound >= left and segment_right_bound <= right:
                self.operations[segment_idx] += val
                self.values[segment_idx] += val
                segments.append(segment_idx)
                continue
            # PARTIAL OVERLAP
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.propagate(segment_idx, segment_left_bound, segment_right_bound)
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
        for segment_idx in segments:
            self.ascend(segment_idx)

    def query(self, left: int, x: int) -> int:
        stack = [(0, self.size, 0)]
        result = -1
        while stack and result == -1:
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_right_bound <= left or self.values[segment_idx] < x: continue
            # LEAF NODE
            if self.is_leaf(segment_left_bound, segment_right_bound):
                result = segment_left_bound
                break
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.propagate(segment_idx, segment_left_bound, segment_right_bound)
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
        return result
    
    def __repr__(self) -> str:
        return f"values: {self.values}, operations: {self.operations}"
```

## Lazy segment tree for C++ for range update and range queries

This is a very generalize implementation, you can really put anything, you probably won't need the range though,  For this one I wanted to do arithmetic progression, so I was storing ranges to get the start and end value.

Anyway you can take this one and modify it for whatever purposes.

```cpp
const int neutral = 0, noop = 0;

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
```

## Lazy Segment Tree point queries and range updates

- range assignment updates
- point queries

range updates are [L, R) (exclusive for right end point)

Assignment of non-negative integers, if there are different, you may need to change noop to some integer value that is never assigned. 


```cpp
struct LazySegmentTree {
    vector<int> values;
    int size, noop = -1;
    // assignment of non-negative integers

    void init(int n) {
        size = 1;
        while (size < n) size *= 2;
        values.assign(2 * size, noop);
    }

    bool is_leaf(int segment_right_bound, int segment_left_bound) {
        return segment_right_bound - segment_left_bound == 1;
    }

    void propagate(int segment_idx, int segment_left_bound, int segment_right_bound) {
        if (is_leaf(segment_right_bound, segment_left_bound) || values[segment_idx] == noop) return;
        int left_segment_idx = 2 * segment_idx + 1, right_segment_idx = 2 * segment_idx + 2;
        values[left_segment_idx] = values[segment_idx];
        values[right_segment_idx] = values[segment_idx];
        values[segment_idx] = noop;
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
                values[segment_idx] = val;
                continue;
            }
            // PARTIAL OVERLAP
            int mid_point = (segment_left_bound + segment_right_bound) >> 1;
            int left_segment_idx = 2 * segment_idx + 1, right_segment_idx = 2 * segment_idx + 2;
            propagate(segment_idx, segment_left_bound, segment_right_bound);
            stk.emplace(mid_point, segment_right_bound, right_segment_idx);
            stk.emplace(segment_left_bound, mid_point, left_segment_idx);
        }
    }
    int query(int i) {
        stack<tuple<int, int, int>> stk;
        stk.emplace(0, size, 0);
        int segment_left_bound, segment_right_bound, segment_idx;
        while (!stk.empty()) {
            tie(segment_left_bound, segment_right_bound, segment_idx) = stk.top();
            stk.pop();
            // NO OVERLAP
            if (i < segment_left_bound || i >= segment_right_bound) continue;
            // COMPLETE OVERLAP
            if (is_leaf(segment_right_bound, segment_left_bound)) return values[segment_idx];
            // PARTIAL OVERLAP
            int mid_point = (segment_left_bound + segment_right_bound) >> 1;
            int left_segment_idx = 2 * segment_idx + 1, right_segment_idx = 2 * segment_idx + 2;
            propagate(segment_idx, segment_left_bound, segment_right_bound);
            stk.emplace(mid_point, segment_right_bound, right_segment_idx);
            stk.emplace(segment_left_bound, mid_point, left_segment_idx);
        }
        return -1;
    }
};
```
