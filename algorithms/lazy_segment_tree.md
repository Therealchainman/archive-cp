# Lazy Segment Tree

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

## Fast Lazy Segment tree for Range Additions Queries

1. range additions
1. range max queries

This is a lazy segment tree implementation for range additions and any query function.

This is a fast implementation, I need to rewrite it sometime, right now taken from another website. 

It is 0 indexed based and range queries are of the form [L, R) I believe, not 100% sure, need to apply to more problems. 
It allows and range query just change out segfunc, But it only allows addition range updates.  

So you can update a range by adding value x to range [L, R).

It uses delayed propagation of the range updates, to make it faster.

init_val is actually an array of initial values, and ide_ele I believe is the identity element possibly?  It is like the neutral value or the value by default, so for a query it will start with that ide_ele value, and update it according to th segfunc.

```py
# https://qiita.com/ether2420/items/7b67b2b35ad5f441d686
def segfunc(x,y):
    return max(x, y)
class LazySegTree_RAQ:
    def __init__(self,init_val,segfunc,ide_ele):
        n = len(init_val)
        self.segfunc = segfunc
        self.ide_ele = ide_ele
        self.num = 1<<(n-1).bit_length()
        self.tree = [ide_ele]*2*self.num
        self.lazy = [0]*2*self.num
        for i in range(n):
            self.tree[self.num+i] = init_val[i]
        for i in range(self.num-1,0,-1):
            self.tree[i] = self.segfunc(self.tree[2*i], self.tree[2*i+1])
    def gindex(self,l,r):
        l += self.num
        r += self.num
        lm = l>>(l&-l).bit_length()
        rm = r>>(r&-r).bit_length()
        while r>l:
            if l<=lm:
                yield l
            if r<=rm:
                yield r
            r >>= 1
            l >>= 1
        while l:
            yield l
            l >>= 1
    def propagates(self,*ids):
        for i in reversed(ids):
            v = self.lazy[i]
            if v==0:
                continue
            self.lazy[i] = 0
            self.lazy[2*i] += v
            self.lazy[2*i+1] += v
            self.tree[2*i] += v
            self.tree[2*i+1] += v
    def add(self,l,r,x):
        ids = self.gindex(l,r)
        l += self.num
        r += self.num
        while l<r:
            if l&1:
                self.lazy[l] += x
                self.tree[l] += x
                l += 1
            if r&1:
                self.lazy[r-1] += x
                self.tree[r-1] += x
            r >>= 1
            l >>= 1
        for i in ids:
            self.tree[i] = self.segfunc(self.tree[2*i], self.tree[2*i+1]) + self.lazy[i]
    def query(self,l,r):
        self.propagates(*self.gindex(l,r))
        res = self.ide_ele
        l += self.num
        r += self.num
        while l<r:
            if l&1:
                res = self.segfunc(res,self.tree[l])
                l += 1
            if r&1:
                res = self.segfunc(res,self.tree[r-1])
            l >>= 1
            r >>= 1
        return res

# initialization
seg = LazySegTree_RAQ([0] * M, segfunc, 0)
```

## Fast Lazy Segment tree for Range Assignment Queries

This one is untested, but it is suppose to basically assign a value to a range.

```py
def segfunc(x,y):
    return min(x,y)
class LazySegTree_RUQ:
    def __init__(self,init_val,segfunc,ide_ele):
        n = len(init_val)
        self.segfunc = segfunc
        self.ide_ele = ide_ele
        self.num = 1<<(n-1).bit_length()
        self.tree = [ide_ele]*2*self.num
        self.lazy = [None]*2*self.num
        for i in range(n):
            self.tree[self.num+i] = init_val[i]
        for i in range(self.num-1,0,-1):
            self.tree[i] = self.segfunc(self.tree[2*i],self.tree[2*i+1])
    def gindex(self,l,r):
        l += self.num
        r += self.num
        lm = l>>(l&-l).bit_length()
        rm = r>>(r&-r).bit_length()
        while r>l:
            if l<=lm:
                yield l
            if r<=rm:
                yield r
            r >>= 1
            l >>= 1
        while l:
            yield l
            l >>= 1
    def propagates(self,*ids):
        for i in reversed(ids):
            v = self.lazy[i]
            if v is None:
                continue
            self.lazy[i] = None
            self.lazy[2*i] = v
            self.lazy[2*i+1] = v
            self.tree[2*i] = v
            self.tree[2*i+1] = v
    def update(self,l,r,x):
        ids = self.gindex(l,r)
        self.propagates(*self.gindex(l,r))
        l += self.num
        r += self.num
        while l<r:
            if l&1:
                self.lazy[l] = x
                self.tree[l] = x
                l += 1
            if r&1:
                self.lazy[r-1] = x
                self.tree[r-1] = x
            r >>= 1
            l >>= 1
        for i in ids:
            self.tree[i] = self.segfunc(self.tree[2*i], self.tree[2*i+1])
    def query(self,l,r):
        ids = self.gindex(l,r)
        self.propagates(*self.gindex(l,r))
        res = self.ide_ele
        l += self.num
        r += self.num
        while l<r:
            if l&1:
                res = self.segfunc(res,self.tree[l])
                l += 1
            if r&1:
                res = self.segfunc(res,self.tree[r-1])
            l >>= 1
            r >>= 1
        return res
```

## Generalizable Lazy Segment Tree from ACL library

I'm a bit confused about this lazy segment tree, but there exists some documentation on it, and it consists of S, F and mappings and compositions and binary operation

```py
class lazy_segtree():
    def update(self,k):self.d[k]=self.op(self.d[2*k],self.d[2*k+1])
    def all_apply(self,k,f):
        self.d[k]=self.mapping(f,self.d[k])
        if (k<self.size):self.lz[k]=self.composition(f,self.lz[k])
    def push(self,k):
        self.all_apply(2*k,self.lz[k])
        self.all_apply(2*k+1,self.lz[k])
        self.lz[k]=self.identity
    def __init__(self,V,OP,E,MAPPING,COMPOSITION,ID):
        self.n=len(V)
        self.log=(self.n-1).bit_length()
        self.size=1<<self.log
        self.d=[E for i in range(2*self.size)]
        self.lz=[ID for i in range(self.size)]
        self.e=E
        self.op=OP
        self.mapping=MAPPING
        self.composition=COMPOSITION
        self.identity=ID
        for i in range(self.n):self.d[self.size+i]=V[i]
        for i in range(self.size-1,0,-1):self.update(i)
    def set(self,p,x):
        assert 0<=p and p<self.n
        p+=self.size
        for i in range(self.log,0,-1):self.push(p>>i)
        self.d[p]=x
        for i in range(1,self.log+1):self.update(p>>i)
    def get(self,p):
        assert 0<=p and p<self.n
        p+=self.size
        for i in range(self.log,0,-1):self.push(p>>i)
        return self.d[p]
    def prod(self,l,r):
        assert 0<=l and l<=r and r<=self.n
        if l==r:return self.e
        l+=self.size
        r+=self.size
        for i in range(self.log,0,-1):
            if (((l>>i)<<i)!=l):self.push(l>>i)
            if (((r>>i)<<i)!=r):self.push(r>>i)
        sml,smr=self.e,self.e
        while(l<r):
            if l&1:
                sml=self.op(sml,self.d[l])
                l+=1
            if r&1:
                r-=1
                smr=self.op(self.d[r],smr)
            l>>=1
            r>>=1
        return self.op(sml,smr)
    def all_prod(self):return self.d[1]
    def apply_point(self,p,f):
        assert 0<=p and p<self.n
        p+=self.size
        for i in range(self.log,0,-1):self.push(p>>i)
        self.d[p]=self.mapping(f,self.d[p])
        for i in range(1,self.log+1):self.update(p>>i)
    def apply(self,l,r,f):
        assert 0<=l and l<=r and r<=self.n
        if l==r:return
        l+=self.size
        r+=self.size
        for i in range(self.log,0,-1):
            if (((l>>i)<<i)!=l):self.push(l>>i)
            if (((r>>i)<<i)!=r):self.push((r-1)>>i)
        l2,r2=l,r
        while(l<r):
            if (l&1):
                self.all_apply(l,f)
                l+=1
            if (r&1):
                r-=1
                self.all_apply(r,f)
            l>>=1
            r>>=1
        l,r=l2,r2
        for i in range(1,self.log+1):
            if (((l>>i)<<i)!=l):self.update(l>>i)
            if (((r>>i)<<i)!=r):self.update((r-1)>>i)
    def max_right(self,l,g):
        assert 0<=l and l<=self.n
        assert g(self.e)
        if l==self.n:return self.n
        l+=self.size
        for i in range(self.log,0,-1):self.push(l>>i)
        sm=self.e
        while(1):
            while(l%2==0):l>>=1
            if not(g(self.op(sm,self.d[l]))):
                while(l<self.size):
                    self.push(l)
                    l=(2*l)
                    if (g(self.op(sm,self.d[l]))):
                        sm=self.op(sm,self.d[l])
                        l+=1
                return l-self.size
            sm=self.op(sm,self.d[l])
            l+=1
            if (l&-l)==l:break
        return self.n
    def min_left(self,r,g):
        assert (0<=r and r<=self.n)
        assert g(self.e)
        if r==0:return 0
        r+=self.size
        for i in range(self.log,0,-1):self.push((r-1)>>i)
        sm=self.e
        while(1):
            r-=1
            while(r>1 and (r%2)):r>>=1
            if not(g(self.op(self.d[r],sm))):
                while(r<self.size):
                    self.push(r)
                    r=(2*r+1)
                    if g(self.op(self.d[r],sm)):
                        sm=self.op(self.d[r],sm)
                        r-=1
                return r+1-self.size
            sm=self.op(self.d[r],sm)
            if (r&-r)==r:break
        return 0

mod = 998244353

# I don't get this? 
def op(a, b):
    return ((a[0] + b[0]) % mod, (a[1] + b[1]) % mod)
# maps F, S to S
def mapping(f, x):
    return ((f[0] * x[0] + x[1] * f[1]) % mod, x[1])
# composition of F, F to F
def composition(f, g):
    return ((f[0] * g[0]) % mod, (g[1] * f[0] + f[1]) % mod)
# identity element for op (0, 0)
# identity element for mapping (1, 0)
```

## Lazy segment tree for C++

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