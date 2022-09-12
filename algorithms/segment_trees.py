from math import inf
from typing import Callable

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
    
    def __repr__(self) -> str:
        return f"array: {self.tree}"