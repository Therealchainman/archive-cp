from math import inf
from typing import Callable, Type, List
from copy import deepcopy


# Implement an algorithm to find the first element greater than a given amount
# design a query algorithm that will find the first element greater than a given amount
# max segment tree

# Segment tree datastructure that stores the maximum value over ranges

"""
Note to self the query are still untested, need further testing
"""
class MaxSegmentTree:
    def __init__(self,arr):
        self.arr = arr
        n = len(arr)
        self.neutral = -inf
        self.size = 1
        while self.size<n:
            self.size*=2
        self.tree = [0 for _ in range(self.size*2)]
    def update_tree(self, idx, val):
        self.update(idx,val,0,0,self.size)
    def build_tree(self):
        for i, val in enumerate(self.arr):
            self.update_tree(i,val)
    def update(self,idx,val,x,lx,rx):
        if rx-lx==1:
            self.tree[x] = val
            return
        mid = rx+lx>>1
        if idx<mid:
            self.update(idx,val,2*x+1,lx,mid)
        else:
            self.update(idx,val,2*x+2,mid,rx)
        self.tree[x] = max(self.tree[2*x+1],self.tree[2*x+2])
    def query(self, l, r, x, lx, rx):
        if lx>=r or l>=rx:
            return self.neutral
        if lx>=l and rx<=r:
            return self.tree[x]
        m = lx+rx>>1
        sl = self.query(l,r,2*x+1,lx,m)
        sr = self.query(l,r,2*x+2,m,rx)
        return max(sl,sr)
    def query_tree(self, l, r):
        return self.query(l,r,0,0,self.size)
    def get_first_tree(self,l, r,val):
        return self.get_first(l,r,0,0,self.size,val)
    def get_first(self,l,r,x,lx,rx,val):
        if lx>=r or rx<=l: return -1
        if l<=lx and rx<=r:
            if self.tree[x] <= val: return -1
            while rx != lx+1:
                mid = lx + rx >> 1
                if self.tree[2*x+1]>val:
                    x = 2*x+1
                    rx = mid
                else:
                    x = 2*x+2
                    lx = mid

            return lx
        mid = lx+rx>>1
        left_segment = self.get_first(l,r,2*x+1,lx,mid,val)
        if left_segment != -1: return left_segment
        return self.get_first(l,r,2*x+2,mid,rx,val)
    def get_last_tree(self,l,r,val):
        return self.get_last(l,r,0,0,self.size,val)
    def get_last(self,l,r,x,lx,rx,val):
        if lx>=r or rx<=l: return -1
        if l<=lx and rx<=r:
            if self.tree[x] <= val: return -1
            while rx != lx+1:
                mid = lx+rx>>1
                if self.tree[2*x+2]>val:
                    x=2*x+2
                    lx=mid
                else:
                    x=2*x+1
                    rx=mid
            return lx
        mid = lx+rx>>1
        right_segment = self.get_last(l,r,2*x+2,mid,rx,val)
        if right_segment != -1: return right_segment
        return self.get_last(l,r,2*x+1,lx,mid,val)
    def get_count(self,i):
        left_index = self.get_last_tree(0,i+1,self.arr[i])
        right_index = self.get_first_tree(i+1,n,arr[i])
        right_index = right_index if right_index!=-1 else len(self.arr)
        return right_index-left_index-1


"""
This max segment tree is tested and works for a problem

This works by building a segment tree that contains the maximum value in every range. 

This is 0-indexed based max segment tree so to query for the range (0,2) you want to query (0,3) in the tree because it is exclusive for the right_bound. 

This function includes the update and query methods

The method that gets the first tree, will return the first tree that contains value that is less than or equal to val.  

This is how it differs with the function above, because it will only return value that is less than val.  
"""


class MaxSegmentTree:
    def __init__(self,arr):
        self.arr = arr
        n = len(arr)
        self.neutral = -inf
        self.size = 1
        while self.size<n:
            self.size*=2
        self.tree = [0 for _ in range(self.size*2)]
    def update_tree(self, idx, val):
        self.update(idx,val,0,0,self.size)
    def build_tree(self):
        for i, val in enumerate(self.arr):
            self.update_tree(i,val)
    def update(self,idx,val,x,lx,rx):
        if rx-lx==1:
            self.tree[x] = val
            return
        mid = rx+lx>>1
        if idx<mid:
            self.update(idx,val,2*x+1,lx,mid)
        else:
            self.update(idx,val,2*x+2,mid,rx)
        self.tree[x] = max(self.tree[2*x+1],self.tree[2*x+2])
    def query(self, l, r, x, lx, rx):
        if lx>=r or l>=rx:
            return self.neutral
        if lx>=l and rx<=r:
            return self.tree[x]
        m = lx+rx>>1
        sl = self.query(l,r,2*x+1,lx,m)
        sr = self.query(l,r,2*x+2,m,rx)
        return max(sl,sr)
    def query_tree(self, l, r):
        return self.query(l,r,0,0,self.size)
    def get_first_tree(self,l, r,val):
        return self.get_first(l,r,0,0,self.size,val)
    def get_first(self,l,r,x,lx,rx,val):
        if lx>=r or rx<=l: return -1
        if l<=lx and rx<=r:
            if self.tree[x] < val: return -1
            while rx != lx+1:
                mid = lx + rx >> 1
                if self.tree[2*x+1]>=val:
                    x = 2*x+1
                    rx = mid
                else:
                    x = 2*x+2
                    lx = mid

            return lx
        mid = lx+rx>>1
        left_segment = self.get_first(l,r,2*x+1,lx,mid,val)
        if left_segment != -1: return left_segment
        return self.get_first(l,r,2*x+2,mid,rx,val)
    def __repr__(self):
        return f"array: {self.tree}"

"""
This is a segment tree that is for range queries for finding the count of elements in range
this means for the update tree part it performs self.tree[x] += val, so that it increases by the update
amount, so increase by the number of count that this value increases by
"""
class CountSegmentTree:
    def __init__(self, arr):
        self.arr = arr
        n = len(arr)
        self.neutral = 0
        self.size = 1
        while self.size <n:
            self.size *= 2
        self.tree = [0 for _ in range(self.size*2)]
    def update_tree(self, idx, val):
        self.update(idx,val,0,0,self.size)
    def build_tree(self):
        for i, val in enumerate(self.arr):
            self.update_tree(i,val)
    def update(self,idx,val,x,lx,rx):
        if rx-lx==1:
            self.tree[x] += val
            return
        mid = rx+lx>>1
        if idx<mid:
            self.update(idx,val,2*x+1,lx,mid)
        else:
            self.update(idx,val,2*x+2,mid,rx)
        self.tree[x] = self.tree[2*x+1] + self.tree[2*x+2]
    def query(self, l, r, x, lx, rx):
        if lx>=r or l>=rx:
            return self.neutral
        if lx>=l and rx<=r:
            return self.tree[x]
        m = lx+rx>>1
        sl = self.query(l,r,2*x+1,lx,m)
        sr = self.query(l,r,2*x+2,m,rx)
        return sl + sr
    def query_tree(self, l, r):
        return self.query(l,r,0,0,self.size)

"""
segment tree for sum 

to query range [1,4], query_tree(1,5), note that it is exclusive for the right endpoint
so that means it queries [left, right)
"""
class PreSumSegmentTree:
    def __init__(self,arr):
        self.arr = arr
        n = len(arr)
        self.neutral = 0
        self.size = 1
        while self.size<n:
            self.size*=2
        self.tree = [0 for _ in range(self.size*2)]
        self.build_tree()
    def update_tree(self, idx, val):
        self.update(idx,val,0,0,self.size)
    def build_tree(self):
        for i, val in enumerate(self.arr):
            self.update_tree(i,val)
    def update(self,idx,val,x,lx,rx):
        if rx-lx==1:
            self.tree[x] = val
            return
        mid = rx+lx>>1
        if idx<mid:
            self.update(idx,val,2*x+1,lx,mid)
        else:
            self.update(idx,val,2*x+2,mid,rx)
        self.tree[x] = self.tree[2*x+1] + self.tree[2*x+2]
    def query(self, l, r, x, lx, rx):
        if lx>=r or l>=rx:
            return self.neutral
        if lx>=l and rx<=r:
            return self.tree[x]
        m = lx+rx>>1
        sl = self.query(l,r,2*x+1,lx,m)
        sr = self.query(l,r,2*x+2,m,rx)
        return sl + sr
    def query_tree(self, l, r):
        return self.query(l,r,0,0,self.size)
    def __repr__(self):
        return f"array: {self.tree}"

"""
General Segment Tree

Iterative implementation because it has a better runtime compared to recursive implementation

Accepts n which is the size of what you need for the leaf nodes in the segment tree, so if need segment tree for an array
of size 4, then n =4
Accepts a neutral value, which is what the value should be set as if no value has exists yet, usually 0 for sum and -inf for max
and inf for min function
Accepts an update function that takes two arguments
Also accepts a count variable, for if you want the tree to keep count
cause then it neesd to set values at root to be incremented

For the range queries it is [l,r), that is l is inclusive and r is exclusive bound. 
So if you query [2,5), it be computing values 2,3,4 in that range
"""

class SegmentTree:
    def __init__(self, n: int, neutral: int, func: Callable[[int, int], int], is_count: bool = False):
        self.neutral = neutral
        self.size = 1
        self.is_count = is_count
        self.func = func
        self.n = n
        while self.size<n:
            self.size*=2
        self.tree = [0 for _ in range(self.size*2)]
        
    def update(self, idx: int, val: int) -> None:
        idx += self.size - 1
        self.tree[idx] = self.tree[idx] + val if self.is_count else val
        while idx > 0:
            idx -= 1
            idx >>= 1
            self.tree[idx] = self.func(self.tree[2*idx+1], self.tree[2*idx+2])
            
    def query(self, l: int, r: int) -> int:
        stack = [(0, self.size, 0)]
        result = 0
        while stack:
            # BOUNDS FOR CURRENT INTERVAL and idx for tree
            left_bound, right_bound, idx = stack.pop()
            # OUT OF BOUNDS
            if left_bound >= r or right_bound <= l: continue
            # CHECK IF CURRENT BOUNDS ARE WITHIN THE l and r
            if left_bound >= l and right_bound <= r:
                result = self.func(result, self.tree[idx])
                continue
            mid = (left_bound + right_bound)>>1
            stack.extend([(left_bound, mid, 2*idx+1), (mid, right_bound, 2*idx+2)])
        return result

    # computes the kth one from right to left, so finds index where there are k ones to the right.  
    def k_query(self, k: int) -> int:
        left_bound, right_bound, idx = 0, self.size, 0
        while right_bound - left_bound != 1:
            left_index, right_index = 2*idx+1, 2*idx+2
            mid = (left_bound+right_bound)>>1
            if k > self.tree[right_index]: # continue in the left branch
                idx, right_bound = left_index, mid
                k -= self.tree[right_index]
            else: # continue in the right branch
                idx, left_bound = right_index, mid
        return left_bound
    
    def __repr__(self) -> str:
        return f"array: {self.tree}"
"""
An even more general segment tree that solves a wide range of functions, because it allows unique update and merge function 

It also allows unique nodes to be provided, this has an example with inversion count
"""
class Node:
    def __init__(self, inversion_count: int, freq: List[int]):
        self.inversion_count = inversion_count
        self.freq = freq
class SegmentTree:
    def __init__(self, n: int, neutral: int, func: Callable[[int, int], int]):
        self.neutral = neutral
        self.size = 1
        self.merge = func
        self.n = n
        while self.size<n:
            self.size*=2
        self.tree = [deepcopy(neutral) for _ in range(self.size*2)]
        
    def update(self, idx: int, val: int, update_func: Callable[[List[List[int]],int,int], None]) -> None:
        idx += self.size - 1
        update_func(self.tree, idx, val)
        while idx > 0:
            idx -= 1
            idx >>= 1
            self.tree[idx] = self.merge(self.tree[2*idx+1], self.tree[2*idx+2])
            
    def query(self, left: int, right: int) -> int:
        stack = [(0, self.size, 0)]
        result = self.neutral
        while stack:
            # BOUNDS FOR CURRENT INTERVAL and idx for tree
            left_segment_bound, right_segment_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if left_segment_bound >= right or right_segment_bound <= left: continue
            # COMPLETE OVERLAP
            if left_segment_bound >= left and right_segment_bound <= right:
                result += self.tree[segment_idx]
                continue
            # PARTIAL OVERLAP
            mid_point = (left_segment_bound + right_segment_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            stack.extend([(left_segment_bound, mid_point, left_segment_idx), (mid_point, right_segment_bound, right_segment_idx)])
        return result

    # computes the kth one from right to left, so finds index where there are k ones to the right.  
    def k_query(self, k: int) -> int:
        left_bound, right_bound, idx = 0, self.size, 0
        while right_bound - left_bound != 1:
            left_index, right_index = 2*idx+1, 2*idx+2
            mid = (left_bound+right_bound)>>1
            if k > self.tree[right_index]: # continue in the left branch
                idx, right_bound = left_index, mid
                k -= self.tree[right_index]
            else: # continue in the right branch
                idx, left_bound = right_index, mid
        return left_bound
    
    def __repr__(self) -> str:
        return f"array: {self.tree}"
        

class Node:
    def __init__(self, val: int):
        self.val = val

    def __repr__(self) -> str:
        return f"val: {self.val}"
"""
Lazy segment tree data structure that works with 
- range updates
- point queries

This particular implementation works for assignment operation, but can always switch out the operation

A segment tree datastructure that can solve range updates that are non-commutative, that is order matters

This uses lazy propagation to optimally perform these range updates.  This means that it updates the value
when it needs to.  

Nodes represent segments in segment tree datastructure, in other words nodes represent ranges or intervals.

No overlap: The query range does not overlap with the segment range
complete overlap: The segment range is completely contained within the query range
partial overlap: The segment range is not completely contained within the query range, but it does have a non-zero intersection with it.
"""
class LazySegmentTree:
    def __init__(self, n: int, neutral_node: Type[Node], noop: int):
        self.neutral = neutral_node
        self.size = 1
        self.noop = noop
        self.n = n 
        while self.size<n:
            self.size*=2
        self.tree = [deepcopy(neutral_node) for _ in range(self.size*2)]

    def is_leaf_node(self, segment_right_bound, segment_left_bound) -> bool:
        return segment_right_bound - segment_left_bound == 1

    def operation(self, val: int) -> Node:
        return Node(val)

    def propagate(self, segment_idx: int, segment_left_bound: int, segment_right_bound: int) -> None:
        # do not want to propagate if it is a leaf node or if it is no operation (means there are no updates stored there).
        if self.is_leaf_node(segment_right_bound, segment_left_bound) or self.tree[segment_idx].val == self.noop: return
        left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
        self.tree[left_segment_idx] = self.operation(self.tree[segment_idx].val)
        self.tree[right_segment_idx] = self.operation(self.tree[segment_idx].val)
        self.tree[segment_idx].val = self.noop
    
    def update(self, left: int, right: int, val: int) -> None:
        stack = [(0, self.size, 0)]
        while stack:
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # COMPLETE OVERLAP
            if segment_left_bound >= left and segment_right_bound <= right:
                self.tree[segment_idx] = self.operation(val)
                continue
            # PARTIAL OVERLAP
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.propagate(segment_idx, segment_left_bound, segment_right_bound)
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])

    def query(self, i: int) -> int:
        stack = [(0, self.size, 0)]
        while stack:
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if i < segment_left_bound or i >= segment_right_bound: continue
            # LEAF NODE
            if self.is_leaf_node(segment_right_bound, segment_left_bound): 
                return self.tree[segment_idx].val
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.propagate(segment_idx, segment_left_bound, segment_right_bound)            
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
    
    def __repr__(self) -> str:
        return f"array: {self.tree}"


"""
Segment tree datastructure for 
- range updates
- range queries
when the update and query function are distributive such as
min(a,b)+v = min(a+v,b+v)
(a+b)*v = a*v+b*v
"""
class SegmentTree:
    def __init__(self, n: int, neutral: int, initial: int):
        self.mod = int(1e9) + 7
        self.neutral = neutral
        self.size = 1
        self.n = n
        self.initial_val = initial
        while self.size<n:
            self.size*=2
        self.operations = [initial for _ in range(self.size*2)]
        self.values = [neutral for _ in range(self.size*2)]
        self.build()

    def build(self):
        for segment_idx in range(self.n):
            segment_idx += self.size - 1
            self.values[segment_idx]  = self.initial_val
            self.ascend(segment_idx)

    def modify_op(self, x: int, y: int) -> int:
        return x + y
    
    def calc_op(self, x: int, y: int) -> int:
        return min(x, y)

    def ascend(self, segment_idx: int) -> None:
        while segment_idx > 0:
            segment_idx -= 1
            segment_idx >>= 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.values[segment_idx] = self.calc_op(self.values[left_segment_idx], self.values[right_segment_idx])
            self.values[segment_idx] = self.modify_op(self.values[segment_idx], self.operations[segment_idx])
        
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
                self.values[segment_idx] = self.modify_op(self.values[segment_idx], val)
                segments.append(segment_idx)
                continue
            # PARTIAL OVERLAP
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
        for segment_idx in segments:
            self.ascend(segment_idx)
            
    def query(self, left: int, right: int) -> int:
        stack = [(0, self.size, 0, self.initial_val)]
        result = self.neutral
        while stack:
            # BOUNDS FOR CURRENT INTERVAL and idx for tree
            segment_left_bound, segment_right_bound, segment_idx, operation_val = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # COMPLETE OVERLAP
            if segment_left_bound >= left and segment_right_bound <= right:
                modified_val = self.modify_op(self.values[segment_idx], operation_val)
                result = self.calc_op(result, modified_val)
                continue
            # PARTIAL OVERLAP
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            operation_val = self.modify_op(operation_val, self.operations[segment_idx])
            stack.extend([(mid_point, segment_right_bound, right_segment_idx, operation_val), (segment_left_bound, mid_point, left_segment_idx, operation_val)])
        return result
    
    def __repr__(self) -> str:
        return f"operations array: {self.operations}, values array: {self.values}"

"""
Kth Segment Tree 

- point updates
- point query
- kth element query

Sets the value at segment tree equal to val

Can be used to get the count of elements in a range, by setting val=1
"""
class SegmentTree:
    def __init__(self, n: int, neutral: int):
        self.neutral = neutral
        self.size = 1
        self.n = n
        while self.size<n:
            self.size*=2
        self.tree = [0 for _ in range(self.size*2)]

    def ascend(self, segment_idx: int) -> None:
        while segment_idx > 0:
            segment_idx -= 1
            segment_idx >>= 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.tree[segment_idx] = self.tree[left_segment_idx] + self.tree[right_segment_idx]
        
    def update(self, segment_idx: int, val: int) -> None:
        segment_idx += self.size - 1
        self.tree[segment_idx] = val
        self.ascend(segment_idx)

    # computes the kth one from right to left, so finds index where there are k ones to the right.  
    def k_query(self, k: int) -> int:
        left_segment_bound, right_segment_bound, segment_idx = 0, self.size, 0
        while right_segment_bound - left_segment_bound != 1:
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            mid_point = (left_segment_bound + right_segment_bound) >> 1
            if k <= self.tree[left_segment_idx]: # continue in the left branch
                segment_idx, right_segment_bound = left_segment_idx, mid_point
            else: # continue in right branch and decrease the number of 1s needed in the right branch
                segment_idx, left_segment_bound = right_segment_idx, mid_point
                k -= self.tree[left_segment_idx]
        return left_segment_bound
    
    def __repr__(self) -> str:
        return f"array: {self.tree}"

"""
Example of lazy segment tree with 
multiple lazy operations
multiple range update operations
range queries
"""

class LazySegmentTree:
    def __init__(self, n: int, neutral: int, noop: int, initial_arr: List[int]):
        self.neutral = neutral
        self.size = 1
        self.noop = noop
        self.n = n 
        while self.size<n:
            self.size*=2
        self.add_operations = [noop for _ in range(self.size*2)]
        self.assign_operations = [noop for _ in range(self.size*2)]
        self.values = [neutral for _ in range(self.size*2)]
        self.arr = initial_arr
        self.build()

    def build(self):
        for segment_idx in range(self.n):
            v = self.arr[segment_idx]
            segment_idx += self.size - 1
            self.values[segment_idx]  = v
            self.ascend(segment_idx)

    def assign_op(self, v: int, segment_len: int = 1) -> int:
        return v*segment_len

    def add_op(self, x: int, y: int, segment_len: int = 1) -> int:
        return x + y*segment_len

    def calc_op(self, x: int, y: int) -> int:
        return x + y

    def is_leaf_node(self, segment_right_bound, segment_left_bound) -> bool:
        return segment_right_bound - segment_left_bound == 1

    def propagate(self, segment_idx: int, segment_left_bound: int, segment_right_bound: int) -> None:
        if self.is_leaf_node(segment_right_bound, segment_left_bound): return
        left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
        children_segment_len = (segment_right_bound - segment_left_bound) >> 1
        if self.assign_operations[segment_idx] != self.noop:
            self.assign_operations[left_segment_idx] = self.assign_operations[segment_idx]
            self.assign_operations[right_segment_idx] = self.assign_operations[segment_idx]
            self.values[left_segment_idx] = self.assign_op(self.assign_operations[segment_idx], children_segment_len)
            self.values[right_segment_idx] = self.assign_op(self.assign_operations[segment_idx], children_segment_len)
            self.assign_operations[segment_idx] = self.noop
            self.add_operations[left_segment_idx] = self.noop
            self.add_operations[right_segment_idx] = self.noop
        if self.add_operations[segment_idx] != self.noop:
            self.add_operations[left_segment_idx] = self.add_op(self.add_operations[left_segment_idx], self.add_operations[segment_idx], 1)
            self.add_operations[right_segment_idx] = self.add_op(self.add_operations[right_segment_idx], self.add_operations[segment_idx], 1)
            self.values[left_segment_idx] = self.add_op(self.values[left_segment_idx], self.add_operations[segment_idx], children_segment_len)
            self.values[right_segment_idx] = self.add_op(self.values[right_segment_idx], self.add_operations[segment_idx], children_segment_len)
            self.add_operations[segment_idx] = self.noop
            if self.assign_operations[left_segment_idx] != self.noop:
                self.assign_operations[left_segment_idx] = self.add_op(self.assign_operations[left_segment_idx], self.add_operations[left_segment_idx], 1)
                self.add_operations[left_segment_idx] = self.noop
            if self.assign_operations[right_segment_idx] != self.noop:
                self.assign_operations[right_segment_idx] = self.add_op(self.assign_operations[right_segment_idx], self.add_operations[right_segment_idx], 1)
                self.add_operations[right_segment_idx] = self.noop

    def ascend(self, segment_idx: int) -> None:
        while segment_idx > 0:
            segment_idx -= 1
            segment_idx >>= 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.values[segment_idx] = self.calc_op(self.values[left_segment_idx], self.values[right_segment_idx])

    def update(self, left: int, right: int, val: int, operation: str) -> None:
        if operation == "add":
            self.add_update(left, right, val)
        elif operation == "assign":
            self.assign_update(left, right, val)
        else:
            raise ValueError("operation must be either add or assign")

    def assign_update(self, left: int, right: int, val: int) -> None:
        stack = [(0, self.size, 0)]
        segments = []
        while stack:
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # COMPLETE OVERLAP
            if segment_left_bound >= left and segment_right_bound <= right:
                self.assign_operations[segment_idx] = val
                self.add_operations[segment_idx] = self.noop
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

    def add_update(self, left: int, right: int, val: int) -> None:
        stack = [(0, self.size, 0)]
        segments = []
        while stack:
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # COMPLETE OVERLAP
            if segment_left_bound >= left and segment_right_bound <= right:
                if self.assign_operations[segment_idx] != self.noop:
                    self.assign_operations[segment_idx] += val
                else:
                    self.add_operations[segment_idx] += val
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

    def query(self, left: int, right: int) -> int:
        stack = [(0, self.size, 0)]
        result = self.neutral
        while stack:
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # LEAF NODE
            if segment_left_bound >= left and segment_right_bound <= right:
                result = self.calc_op(result, self.values[segment_idx])
                continue
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2    
            self.propagate(segment_idx, segment_left_bound, segment_right_bound)      
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
        return result
    
    def __repr__(self) -> str:
        return f"values: {self.values}, add_operations: {self.add_operations}, assign_operations: {self.assign_operations}"


"""
Lazy segment tree data structure
- range updates
- range queries


"""

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
            # LEAF NODE
            if segment_left_bound >= left and segment_right_bound <= right:
                result = self.calc_op(result, self.values[segment_idx])
                continue
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2    
            self.propagate(segment_idx, segment_left_bound, segment_right_bound)      
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
        return result
    
    def __repr__(self) -> str:
        return f"values: {self.values}, operations: {self.operations}"