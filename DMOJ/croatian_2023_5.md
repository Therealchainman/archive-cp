


```py
from bisect import bisect_right
class LazySegmentTree:
    def __init__(self, n: int, neutral: int, noop: int):
        self.neutral = neutral
        self.size = 1
        self.noop = noop
        self.n = n 
        while self.size<n:
            self.size*=2
        self.tree = [neutral for _ in range(self.size*2)]

    def is_leaf_node(self, segment_right_bound, segment_left_bound) -> bool:
        return segment_right_bound - segment_left_bound == 1

    def operation(self, x: int, y: int) -> int:
        return x + y

    def propagate(self, segment_idx: int, segment_left_bound: int, segment_right_bound: int) -> None:
        # do not want to propagate if it is a leaf node
        if self.is_leaf_node(segment_right_bound, segment_left_bound): return
        left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
        self.tree[left_segment_idx] = self.operation(self.tree[left_segment_idx], self.tree[segment_idx])
        self.tree[right_segment_idx] = self.operation(self.tree[right_segment_idx], self.tree[segment_idx])
        self.tree[segment_idx] = self.noop
    
    def update(self, left: int, right: int, val: int) -> None:
        stack = [(0, self.size, 0)]
        while stack:
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # COMPLETE OVERLAP
            if segment_left_bound >= left and segment_right_bound <= right:
                self.tree[segment_idx] = self.operation(self.tree[segment_idx], val)
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
                return self.tree[segment_idx]
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.propagate(segment_idx, segment_left_bound, segment_right_bound)            
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
    
    def __repr__(self) -> str:
        return f"array: {self.tree}"
class LazySegmentTree2:
    def __init__(self, n: int, neutral: int, noop: int, arr):
        self.neutral = neutral
        self.size = 1
        self.noop = noop
        self.n = n
        self.arr = arr
        while self.size<n:
            self.size*=2
        self.operations = [noop for _ in range(self.size*2)]
        self.values = [neutral for _ in range(self.size*2)]
        self.build()
        
    def build(self):
        for segment_idx in range(self.n):
            v = self.arr[segment_idx]
            segment_idx += self.size - 1
            self.values[segment_idx] = v
            self.ascend(segment_idx)

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
        result = self.n
        while stack and result == self.n:
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_right_bound <= left or self.values[segment_idx] <= x: continue
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
import math
def main():
    n, t, s, k = map(int, input().split())
    blinds = sorted(map(int, input().split()))
    Q = int(input())
    queries = list(map(int, input().split()))
    queries = sorted([(q, i) for i, q in enumerate(queries)], reverse = True)
    ans = [0] * Q
    segtree = LazySegmentTree(n, 0, 0)
    for i in range(n):
        segtree.update(i, i + 1, blinds[i])
    segtree2 = LazySegmentTree2(n, -math.inf, 0, blinds)
    H = blinds[-1] # max height
    r = bisect_right(blinds, 0) # done once doesn't matter
    total = 0
    for q, i in queries:
        idx = segtree2.query(0, q)
        m = n - idx
        for _ in range(q, H): # know the largest blind still
            while idx < n and segtree.query(idx) == q: # would this be a query , I guess so 
                idx += 1
                m -= 1
            if m == 0: break
            cost = m * t 
            pcost = s + k * r
            total += min(cost, pcost)
            if pcost <= cost:
                segtree.update(0, n, -1)
                segtree2.update(0, n, -1)
            else:
                segtree.update(idx, n, -1)
                segtree2.update(idx, n, -1)
            while r < n and segtree.query(r) == 0: r += 1
        ans[i] = total
        H = q
    print(*ans)

if __name__ == "__main__":
    main()
```

```py
def main():
    N = int(input())
    if N % 2 == 0 or (N - 3) % 4 == 0: print("Lucija")
    else: print("Ivan")

if __name__ == "__main__":
    T = int(input())
    for _ in range(T):
        main()
```