# Atcoder Beginner Contest 327

## D - Good Tuple Problem 

### Solution 1: 2 coloring, bipartite graph, iterative dfs with stack, undirected graph, even length cycles

if a graph has odd length cycle it cannot be bipartite

```py
def main():
    N, M = map(int, input().split())
    adj = [[] for _ in range(N)]
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    for u, v in zip(A, B):
        u -= 1
        v -= 1
        adj[u].append(v)
        adj[v].append(u)
    color = [-1] * N
    for i in range(N):
        if color[i] != -1: continue
        stack = [i]
        color[i] = 0
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if color[v] == -1:
                    color[v] = color[u] ^ 1
                    stack.append(v)
                elif color[v] == color[u]:
                    return print("No")
    print("Yes")

if __name__ == '__main__':
    main()
```

## E - Maximize Rating

### Solution 1: dynamic programming, math

```py

```

## F - Apples

### Solution 1: segment tree, lazy segment tree, line sweep

lattice points in 2D space.  Use segment tree because it is asking for range addition updates and range maximum queries or can just use lazy segment tree.

```py
import math 

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
        return max(x, y)
 
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

def main():
    M = 200_001
    N, D, W = map(int, input().split())
    queries = [[] for _ in range(M)]
    # line sweep construction
    for _ in range(N):
        t, x = map(int, input().split())
        queries[max(0, t - D)].append((x, 1))
        queries[t].append((x, -1))
    seg = SegmentTree(M, -math.inf, 0)
    res = 0
    for t in range(M):
        for x, delta in queries[t]:
            seg.update(max(0, x - W), x, delta) # range addition update
        # range max query
        res = max(res, seg.query(0, M))
    print(res)

if __name__ == '__main__':
    main()
```

```py
class LazySegmentTree:
    def __init__(self, n: int, neutral: int, noop: int):
        self.mod = int(1e9) + 7
        self.neutral = neutral
        self.size = 1
        self.noop = noop
        self.n = n 
        while self.size<n:
            self.size*=2
        self.operations = [noop for _ in range(self.size*2)]
        self.values = [neutral for _ in range(self.size*2)]

    def is_leaf_node(self, segment_right_bound, segment_left_bound) -> bool:
        return segment_right_bound - segment_left_bound == 1

    def propagate(self, segment_idx: int, segment_left_bound: int, segment_right_bound: int) -> None:
        if self.is_leaf_node(segment_right_bound, segment_left_bound) or self.operations[segment_idx] == self.noop: return
        left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
        self.operations[left_segment_idx] = self.modify_op(self.operations[left_segment_idx], self.operations[segment_idx])
        self.operations[right_segment_idx] = self.modify_op(self.operations[right_segment_idx], self.operations[segment_idx])
        self.values[left_segment_idx] = self.modify_op(self.values[left_segment_idx], self.operations[segment_idx])
        self.values[right_segment_idx] = self.modify_op(self.values[right_segment_idx], self.operations[segment_idx])
        self.operations[segment_idx] = self.noop
 
    def modify_op(self, x: int, y: int) -> int:
        return x + y
    
    def calc_op(self, x: int, y: int) -> int:
        return max(x, y)
 
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
            self.propagate(segment_idx, segment_left_bound, segment_right_bound)
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
        for segment_idx in segments:
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
                result = self.calc_op(result, self.values[segment_idx])
                continue
            # PARTIAL OVERLAP
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.propagate(segment_idx, segment_left_bound, segment_right_bound)
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
        return result
    
    def __repr__(self) -> str:
        return f"operations array: {self.operations}, values array: {self.values}"

def main():
    M = 200_001
    N, D, W = map(int, input().split())
    queries = [[] for _ in range(M)]
    # line sweep construction
    for _ in range(N):
        t, x = map(int, input().split())
        queries[max(0, t - D)].append((x, 1))
        queries[t].append((x, -1))
    seg = LazySegmentTree(M, 0, 0)
    res = 0
    for t in range(M):
        for x, delta in queries[t]:
            seg.update(max(0, x - W), x, delta) # range addition update
        # range max query
        res = max(res, seg.query(0, M))
    print(res)

if __name__ == '__main__':
    main()
```

This is the solution that passes, the other segment trees are apparently too slow. 

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

def main():
    M = 200_001
    N, D, W = map(int, input().split())
    queries = [[] for _ in range(M)]
    # line sweep construction
    for _ in range(N):
        t, x = map(int, input().split())
        queries[max(0, t - D)].append((x, 1))
        queries[t].append((x, -1))
    seg = LazySegTree_RAQ([0] * M, segfunc, 0)
    res = 0
    for t in range(M):
        for x, delta in queries[t]:
            seg.add(max(0, x - W), x, delta) # range addition update
        # range max query
        res = max(res, seg.query(0, M))
    print(res)

if __name__ == '__main__':
    main()
```

## G - Many Good Tuple Problems

### Solution 1: counting

```py

```
