
## A. Inversions

### Solution 1:  update value to be 1 for current permutation value + find the prefix sum from current value to the end + suffix sum

```py
from math import inf
from typing import Callable
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


def main():
    n = int(input())
    arr = map(int,input().split())
    sum_func = lambda x, y: x+y
    neutral = 0
    sumSeg = SegmentTree(n+1, neutral, sum_func)
    results = []
    for p in arr:
        results.append(sumSeg.query(p+1,n+1))
        sumSeg.update(p, 1)
    return ' '.join(map(str,results))

if __name__ == '__main__':
    print(main())
```

## B. Inversions II

### Solution 1:  count k ones to the left of current value

```py
from math import inf
from typing import Callable
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


def main():
    n = int(input())
    arr = list(map(int,input().split()))
    sum_func = lambda x, y: x+y
    neutral = 0
    kthSeg = SegmentTree(n, neutral, sum_func)
    for i in range(n):
        kthSeg.update(i,1)
    results = []
    for a in reversed(arr):
        index = kthSeg.k_query(a+1)
        results.append(index+1)
        kthSeg.update(index, 0)
    return ' '.join(map(str,reversed(results)))

if __name__ == '__main__':
    print(main())
```

## C. Nested Segments

### Solution 1:  prefix sum segment tree + greedy + left to right, on second occurrence of integer update the value to one at the first index it was found. 

```py
from math import inf
from typing import Callable
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


def main():
    n = int(input())
    arr = map(int, input().split())
    sum_func = lambda x, y: x+y
    neutral = 0
    sumSeg = SegmentTree(2*n, neutral, sum_func)
    unvisited = -1
    last_index = [unvisited]*(n+1)
    answer = [0]*(n+1)
    for i, num in enumerate(arr):
        if last_index[num] != unvisited:
            answer[num] = sumSeg.query(last_index[num], i)
            sumSeg.update(last_index[num], 1)
        last_index[num] = i
    return ' '.join(map(str, answer[1:]))


if __name__ == '__main__':
    print(main())
```

## D. Intersecting Segments

### Solution 1:  prefixsum segment tree + setting values to index, so that if only seen one it sums to 1 else sums to 0 after both ocurrences

```py
from math import inf
from typing import Callable
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


def main():
    n = int(input())
    arr = map(int, input().split())
    sum_func = lambda x, y: x+y
    neutral = 0
    sumSeg = SegmentTree(2*n, neutral, sum_func)
    unvisited = -1
    last_index = [unvisited]*(n+1)
    answer = [0]*(n+1)
    for i, num in enumerate(arr):
        if last_index[num] != unvisited:
            answer[num] = sumSeg.query(last_index[num], i)-1
            sumSeg.update(last_index[num], -1)
        sumSeg.update(i, 1)
        last_index[num] = i
    return ' '.join(map(str, answer[1:]))


if __name__ == '__main__':
    print(main())
```

## E. Addition to Segment

### Solution 1:  Compute the prefix sum for range query [0,i+1) to get value at the ith + update left val, and right with -val

```py
from math import inf
from typing import Callable
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


def main():
    n, m = map(int,input().split())
    sum_func = lambda x, y: x+y
    neutral = 0
    sumSeg = SegmentTree(n+1, neutral, sum_func, is_count=True)
    results = []
    for _ in range(m):
        query = list(map(int,input().split()))
        if query[0] == 1: # type 1 query
            _, left, right, val = query
            sumSeg.update(left,val)
            sumSeg.update(right,-val)
        else: # type 2 query
            _, i = query
            results.append(sumSeg.query(0,i+1))
    return '\n'.join(map(str,results))

if __name__ == '__main__':
    print(main())
```

## A. Sign alternation

### Solution 1:  store even and odd sum in the nodes of the segment tree datastructure

```py
from math import inf
from typing import Callable, Type, List
from copy import deepcopy
class Node:
    def __init__(self, even_val: int = 0, odd_val: int = 0):
        self.even_sum = even_val
        self.odd_sum = odd_val
    def __repr__(self) -> str:
        return f"even: {self.even_sum}, odd: {self.odd_sum}"
class SegmentTree:
    def __init__(self, n: int, neutral: Type[Node], func: Callable[[int, int], int]):
        self.neutral = neutral
        self.size = 1
        self.func = func
        self.n = n
        while self.size<n:
            self.size*=2
        self.tree = [deepcopy(neutral) for _ in range(self.size*2)]
        
    def update(self, idx: int, val: int, update_func: Callable[[List[List[int]],int,int], None]) -> None:
        is_odd = idx&1
        idx += self.size - 1
        update_func(self.tree, idx, val, is_odd)
        while idx > 0:
            idx -= 1
            idx >>= 1
            self.tree[idx] = self.func(self.tree[2*idx+1], self.tree[2*idx+2])
            
    def query(self, l: int, r: int) -> int:
        stack = [(0, self.size, 0)]
        result = self.neutral
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
            stack.extend([(mid, right_bound, 2*idx+2), (left_bound, mid, 2*idx+1)])
        return result
    
    def __repr__(self) -> str:
        return f"array: {self.tree}"

def update_func(tree: List[Node], idx: int, val: int, is_odd: bool) -> None:
    if is_odd:
        tree[idx].odd_sum = val
    else:
        tree[idx].even_sum = val

def sum_func(node_left: Type[Node], node_right: Type[Node]) -> Node:
    return Node(node_left.even_sum+node_right.even_sum, node_left.odd_sum+node_right.odd_sum)

def main():
    n = int(input())
    arr = map(int,input().split())
    m = int(input())
    neutral = Node(0)
    sumSeg = SegmentTree(n, neutral, sum_func)
    results = []
    for i, num in enumerate(arr):
        sumSeg.update(i, num, update_func)
    for _ in range(m):
        type_, x, y = map(int,input().split())
        if type_ == 1:
            node = sumSeg.query(x-1,y)
            result = node.even_sum - node.odd_sum if x&1 else node.odd_sum - node.even_sum
            results.append(result)
        else:
            sumSeg.update(x-1, y, update_func)
    return '\n'.join(map(str,results))

if __name__ == '__main__':
    print(main())
```

##

### Solution 1: 

```py

```

## C. Number of Inversions on Segment

### Solution 1:  segment tree with unique node + updating inversion count

```py
from math import inf
from typing import Callable, Type, List
from copy import deepcopy
class Node:
    def __init__(self, inversion_count: int, freq: List[int]):
        self.inversion_count = inversion_count
        self.freq = freq
class SegmentTree:
    def __init__(self, n: int, neutral: int, func: Callable[[int, int], int], node: Type[Node], is_count: bool = False):
        self.neutral = neutral
        self.size = 1
        self.is_count = is_count
        self.func = func
        self.n = n
        while self.size<n:
            self.size*=2
        self.tree = [deepcopy(node) for _ in range(self.size*2)]
        
    def update(self, idx: int, val: int, update_func: Callable[[List[List[int]],int,int], None]) -> None:
        idx += self.size - 1
        update_func(self.tree, idx, val)
        while idx > 0:
            idx -= 1
            idx >>= 1
            self.tree[idx] = self.func(self.tree[2*idx+1], self.tree[2*idx+2])
            
    def query(self, l: int, r: int) -> int:
        stack = [(0, self.size, 0)]
        result = self.neutral
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
            stack.extend([(mid, right_bound, 2*idx+2), (left_bound, mid, 2*idx+1)])
        return result
    
    def __repr__(self) -> str:
        return f"array: {self.tree}"

def update_func(tree: List[Node], idx: int, val: int) -> None:
    for i in range(40):
        tree[idx].freq[i] = 0
    tree[idx].freq[val] = 1

def inv_func(node_left: Type[Node], node_right: Type[Node]) -> Node:
    inv_count = node_left.inversion_count + node_right.inversion_count
    left_count = 0
    freq = [0]*40
    for i in reversed(range(40)):
        inv_count += node_right.freq[i]*left_count
        left_count += node_left.freq[i]
        freq[i] += node_right.freq[i] + node_left.freq[i]
    return Node(inv_count, freq)

def main():
    n, m = map(int,input().split())
    arr = map(int,input().split())
    neutral = Node(0, [0]*40)
    node = Node(0, [0]*40)
    invSeg = SegmentTree(n, neutral, inv_func, node = node)
    results = []
    for i, num in enumerate(arr):
        invSeg.update(i, num-1, update_func)
    for _ in range(m):
        type_, x, y = map(int,input().split())
        if type_ == 1:
            results.append(invSeg.query(x-1,y).inversion_count)
        else:
            invSeg.update(x-1, y-1, update_func)
    return '\n'.join(map(str,results))

if __name__ == '__main__':
    print(main())
```

## D. Number of Different on Segment

### Solution 1:  segment tree where the nodes hold the existence of elements in that range since only need array of size 40

```py
from math import inf
from typing import Callable, Type, List
class Node:
    def __init__(self, node_contents):
        self.val = node_contents
class SegmentTree:
    def __init__(self, n: int, neutral: int, func: Callable[[int, int], int], node: Type[Node], is_count: bool = False):
        self.neutral = neutral
        self.size = 1
        self.is_count = is_count
        self.func = func
        self.n = n
        while self.size<n:
            self.size*=2
        self.tree = [node.val.copy() for _ in range(self.size*2)]
        
    def update(self, idx: int, val: int, update_func: Callable[[List[List[int]],int,int], None]) -> None:
        idx += self.size - 1
        update_func(self.tree, idx, val)
        while idx > 0:
            idx -= 1
            idx >>= 1
            self.tree[idx] = self.func(self.tree[2*idx+1], self.tree[2*idx+2])
            
    def query(self, l: int, r: int) -> int:
        stack = [(0, self.size, 0)]
        result = self.neutral
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

def update_func(freq: List[List[int]], idx: int, val: int) -> None:
    for i in range(40):
        freq[idx][i] = 0
    freq[idx][val] = 1


def main():
    n, m = map(int,input().split())
    arr = map(int,input().split())
    different_func = lambda x, y: [u|v for u,v in zip(x,y)]
    neutral = [0]*40
    node = Node(neutral)
    diffSeg = SegmentTree(n, neutral, different_func, node = node)
    results = []
    for i, num in enumerate(arr):
        diffSeg.update(i, num-1, update_func)
    for _ in range(m):
        type_, x, y = map(int,input().split())
        if type_ == 1:
            results.append(sum(diffSeg.query(x-1,y)))
        else:
            diffSeg.update(x-1, y-1, update_func)
    return '\n'.join(map(str,results))

if __name__ == '__main__':
    print(main())
```

##

### Solution 1: 

```py

```

## A. Addition to Segment

### Solution 1:  lazy segment tree datastructure + range update + point query + commutative operation

```py
from math import inf
from typing import Callable, Type, List
from copy import deepcopy
class Node:
    def __init__(self, val: int):
        self.val = val

    def __repr__(self) -> str:
        return f"val: {self.val}"

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

    def operation(self, x: int, y: int) -> Node:
        return Node(x+y)

    def propagate(self, segment_idx: int, segment_left_bound: int, segment_right_bound: int) -> None:
        # do not want to propagate if it is a leaf node
        if self.is_leaf_node(segment_right_bound, segment_left_bound): return
        left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
        self.tree[left_segment_idx] = self.operation(self.tree[left_segment_idx].val, self.tree[segment_idx].val)
        self.tree[right_segment_idx] = self.operation(self.tree[right_segment_idx].val, self.tree[segment_idx].val)
        self.tree[segment_idx].val = self.noop
    
    def update(self, left: int, right: int, val: int) -> None:
        stack = [(0, self.size, 0)]
        while stack:
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # COMPLETE OVERLAP
            if segment_left_bound >= left and segment_right_bound <= right:
                self.tree[segment_idx] = self.operation(self.tree[segment_idx].val, val)
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

def main():
    n, m = map(int, input().split())
    neutral_node = Node(0)
    noop = 0
    assignSeg = LazySegmentTree(n, neutral_node, noop)
    results = []
    for _ in range(m):
        query = list(map(int, input().split()))
        if query[0] == 1:
            # range update
            _, left, right, val = query
            assignSeg.update(left, right, val)
        else:
            # point query
            _, i = query
            results.append(assignSeg.query(i))
    return '\n'.join(map(str,results))

if __name__ == '__main__':
    print(main())
```

## B. Applying MAX to Segment

### Solution 1: lazy segment tree + range update + point query + commutative operation

```py
from math import inf
from typing import Callable, Type, List
from copy import deepcopy
class Node:
    def __init__(self, val: int):
        self.val = val

    def __repr__(self) -> str:
        return f"val: {self.val}"

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

    def operation(self, x: int, y: int) -> Node:
        return Node(max(x,y))

    def propagate(self, segment_idx: int, segment_left_bound: int, segment_right_bound: int) -> None:
        # do not want to propagate if it is a leaf node
        if self.is_leaf_node(segment_right_bound, segment_left_bound): return
        left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
        self.tree[left_segment_idx] = self.operation(self.tree[left_segment_idx].val, self.tree[segment_idx].val)
        self.tree[right_segment_idx] = self.operation(self.tree[right_segment_idx].val, self.tree[segment_idx].val)
        self.tree[segment_idx].val = self.noop
    
    def update(self, left: int, right: int, val: int) -> None:
        stack = [(0, self.size, 0)]
        while stack:
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # COMPLETE OVERLAP
            if segment_left_bound >= left and segment_right_bound <= right:
                self.tree[segment_idx] = self.operation(self.tree[segment_idx].val, val)
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

def main():
    n, m = map(int, input().split())
    neutral_node = Node(0)
    noop = 0
    assignSeg = LazySegmentTree(n, neutral_node, noop)
    results = []
    for _ in range(m):
        query = list(map(int, input().split()))
        if query[0] == 1:
            # range update
            _, left, right, val = query
            assignSeg.update(left, right, val)
        else:
            # point query
            _, i = query
            results.append(assignSeg.query(i))
    return '\n'.join(map(str,results))

if __name__ == '__main__':
    print(main())
```

## C. Assignment to Segment

### Solution 1: lazy segment tree + range update + point query + non-commutative operation (order matters)

```py
from math import inf
from typing import Callable, Type, List
from copy import deepcopy
class Node:
    def __init__(self, val: int):
        self.val = val

    def __repr__(self) -> str:
        return f"val: {self.val}"

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

def main():
    n, m = map(int, input().split())
    neutral_node = Node(0)
    noop = -1
    assignSeg = LazySegmentTree(n, neutral_node, noop)
    results = []
    for _ in range(m):
        query = list(map(int, input().split()))
        if query[0] == 1:
            # range update
            _, left, right, val = query
            assignSeg.update(left, right, val)
        else:
            # point query
            _, i = query
            results.append(assignSeg.query(i))
    return '\n'.join(map(str,results))

if __name__ == '__main__':
    print(main())
```

## A. Addition and Minimum

### Solution 1: segment tree + range update + range query + update function and value function are distributive + update and value function are commutative

```py
from math import inf
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

def main():
    n, m = map(int, input().split())
    neutral = inf
    initial = 0
    segTree = SegmentTree(n, neutral, initial)
    results = []
    for _ in range(m):
        query = list(map(int, input().split()))
        if query[0] == 1:
            # range update
            _, left, right, val = query
            segTree.update(left, right, val)
        else:
            # point query
            _, left, right = query
            results.append(segTree.query(left, right))
    return '\n'.join(map(str,results))

if __name__ == '__main__':
    print(main())
```

## B. Multiplication and Sum

### Solution 1: segment tree + range update + range query + update function and value function are distributive + update and value function are commutative

```py
from math import inf
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
        return (x * y)%self.mod
    
    def calc_op(self, x: int, y: int) -> int:
        return (x + y)%self.mod

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
    n, m = map(int, input().split())
    neutral = 0
    initial = 1
    segTree = SegmentTree(n, neutral, initial)
    results = []
    for _ in range(m):
        query = list(map(int, input().split()))
        if query[0] == 1:
            # range update
            _, left, right, val = query
            segTree.update(left, right, val)
        else:
            # point query
            _, left, right = query
            results.append(segTree.query(left, right))
    return '\n'.join(map(str,results))

if __name__ == '__main__':
    print(main())
```

## C. Bitwise OR and AND

### Solution 1: segment tree + range update + range query + update function and value function are distributive + update and value function are commutative + or update function + and range query function

```py
from math import inf
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
        return x | y
    
    def calc_op(self, x: int, y: int) -> int:
        return x & y

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
    n, m = map(int, input().split())
    neutral = 2**31-1
    initial = 0
    segTree = SegmentTree(n, neutral, initial)
    results = []
    for _ in range(m):
        query = list(map(int, input().split()))
        if query[0] == 1:
            # range update
            _, left, right, val = query
            segTree.update(left, right, val)
        else:
            # point query
            _, left, right = query
            results.append(segTree.query(left, right))
    return '\n'.join(map(str,results))

if __name__ == '__main__':
    print(main())
```

## D. Addition and Sum

### Solution 1:  lazy segment tree + range update + range query

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

    def modify_op(self, x: int, y: int, segment_len: int = 1) -> int:
        return x + y*segment_len

    def calc_op(self, x: int, y: int) -> int:
        return x + y

    def is_leaf_node(self, segment_right_bound, segment_left_bound) -> bool:
        return segment_right_bound - segment_left_bound == 1

    def propagate(self, segment_idx: int, segment_left_bound: int, segment_right_bound: int) -> None:
        if self.is_leaf_node(segment_right_bound, segment_left_bound) or self.operations[segment_idx] == self.noop: return
        left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
        children_segment_len = (segment_right_bound - segment_left_bound) >> 1
        self.operations[left_segment_idx] = self.modify_op(self.operations[left_segment_idx], self.operations[segment_idx], 1)
        self.operations[right_segment_idx] = self.modify_op(self.operations[right_segment_idx], self.operations[segment_idx], 1)
        self.values[left_segment_idx] = self.modify_op(self.values[left_segment_idx], self.operations[segment_idx], children_segment_len)
        self.values[right_segment_idx] = self.modify_op(self.values[right_segment_idx], self.operations[segment_idx], children_segment_len)
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
                self.operations[segment_idx] = self.modify_op(self.operations[segment_idx], val, 1)
                segment_len = segment_right_bound - segment_left_bound
                self.values[segment_idx] = self.modify_op(self.values[segment_idx], val, segment_len)
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

def main():
    n, m = map(int, input().split())
    neutral = 0
    noop = 0
    st = LazySegmentTree(n, neutral, noop)
    results = []
    for _ in range(m):
        query = list(map(int, input().split()))
        if query[0] == 1:
            # range add update
            _, left, right, val = query
            st.update(left, right, val)
        elif query[0] == 2:
            # range sum query
            _, left, right = query
            results.append(st.query(left, right))
    return '\n'.join(map(str,results))

if __name__ == '__main__':
    print(main())
```

## E. Assignment and Minimum

### Solution 1:  lazy segment tree + range update + range query

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

def main():
    n, m = map(int, input().split())
    neutral = inf
    noop = -1
    st = LazySegmentTree(n, neutral, noop)
    results = []
    for _ in range(m):
        query = list(map(int, input().split()))
        if query[0] == 1:
            # range assign update
            _, left, right, val = query
            st.update(left, right, val)
        elif query[0] == 2:
            # range minimum query
            _, left, right = query
            results.append(st.query(left, right))
    return '\n'.join(map(str,results))

if __name__ == '__main__':
    print(main())
```

## F. Assignment and Sum

### Solution 1:  lazy segment tree + range update + range query + segment_length for the update of value + updating value is sum + updating operation is assignment

```py
class LazySegmentTree:
    def __init__(self, n: int, neutral: int, noop: int):
        self.neutral = neutral
        self.size = 1
        self.noop = noop
        self.n = n 
        while self.size<n:
            self.size*=2
        self.operations = [noop for _ in range(self.size*2)]
        self.values = [neutral for _ in range(self.size*2)]

    def modify_op(self, v: int, segment_len: int = 1) -> int:
        return segment_len*v

    def calc_op(self, x: int, y: int) -> int:
        return x + y

    def is_leaf_node(self, segment_right_bound, segment_left_bound) -> bool:
        return segment_right_bound - segment_left_bound == 1

    def propagate(self, segment_idx: int, segment_left_bound: int, segment_right_bound: int) -> None:
        # do not want to propagate if it is a leaf node or if it is no operation (means there are no updates stored there).
        if self.is_leaf_node(segment_right_bound, segment_left_bound) or self.operations[segment_idx] == self.noop: return
        left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
        children_segment_len = (segment_right_bound - segment_left_bound) >> 1
        self.operations[left_segment_idx] = self.modify_op(self.operations[segment_idx])
        self.operations[right_segment_idx] = self.modify_op(self.operations[segment_idx])
        self.values[left_segment_idx] = self.modify_op(self.operations[segment_idx], children_segment_len)
        self.values[right_segment_idx] = self.modify_op(self.operations[segment_idx], children_segment_len)
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
                self.operations[segment_idx] = self.modify_op(val)
                segment_len = segment_right_bound - segment_left_bound
                self.values[segment_idx] = self.modify_op(val, segment_len)
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

def main():
    n, m = map(int, input().split())
    neutral = 0
    noop = -1
    assignSeg = LazySegmentTree(n, neutral, noop)
    results = []
    for _ in range(m):
        query = list(map(int, input().split()))
        if query[0] == 1:
            # range update
            _, left, right, val = query
            assignSeg.update(left, right, val)
        else:
            # range query 
            _, left, right = query
            results.append(assignSeg.query(left, right))
    return '\n'.join(map(str,results))

if __name__ == '__main__':
    print(main())
```

## A. Assignment and Maximal Segment

### Solution 1:  lazy segment tree, node

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

def main():
    n, m = map(int, input().split())
    # set noop to -1, and initial_val is just empty node
    seg = LazySegmentTree(n, -math.inf) 
    for _ in range(m):
        L, R, x = map(int, input().split())
        seg.update(L, R, x)
        print(seg.query(0, n))

if __name__ == '__main__':
    main()
```

## B. Inverse and K-th one

### Solution 1:  lazy segment tree, flipping bits, kth query

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

def main():
    n, m = map(int, input().split())
    seg = LazySegmentTree(n, 0, 0, 0)
    for _ in range(m):
        query = list(map(int, input().split()))
        if query[0] == 1:
            L, R = query[1:]
            seg.update(L, R, 1)
        else:
            k = query[1]
            print(seg.kquery(k + 1))

if __name__ == '__main__':
    main()
```

## C. Addition and First element at least X

### Solution 1:  lazy segment tree, maximum, find smallest index within range satisfying condiition of being at least x

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

def main():
    n, m = map(int, input().split())
    seg = LazySegmentTree(n, 0, 0, 0)
    for _ in range(m):
        query = list(map(int, input().split()))
        if query[0] == 1:
            L, R, v = query[1:]
            seg.update(L, R, v)
        else:
            x, L = query[1:]
            print(seg.query(L, x))

if __name__ == '__main__':
    main()
```

```cpp
struct SegmentTree {
    int sz;
    int noop = 0, neutral = 0;
    vector<int> operations, values;
    void init(int n) {
        sz = 1;
        while (sz < n) {
            sz <<= 1;
        }
        operations.assign(sz << 1, noop);
        values.assign(sz << 1, neutral);
    }

    int calc_op(int &x, int &y) {
        return max(x, y);
    }

    bool is_leaf(int segment_right_bound, int segment_left_bound) {
        return segment_right_bound - segment_left_bound == 1;
    }

    void propagate(int segment_idx, int segment_left_bound, int segment_right_bound) {
        if (is_leaf(segment_right_bound, segment_left_bound) || operations[segment_idx] == noop) return;
        int left_segment_idx = 2 * segment_idx + 1;
        int right_segment_idx = 2 * segment_idx + 2;
        int children_segment_len = (segment_right_bound - segment_left_bound) >> 1;
        operations[left_segment_idx] += operations[segment_idx];
        operations[right_segment_idx] += operations[segment_idx];
        values[left_segment_idx] += operations[segment_idx];
        values[right_segment_idx] += operations[segment_idx];
        operations[segment_idx] = noop;
    }

    void update(int segment_left_bound, int segment_right_bound, int segment_idx, int left, int right, int val) {
        if (segment_left_bound >= right || segment_right_bound <= left) return;
        if (segment_left_bound >= left && segment_right_bound <= right) {
            int segment_len = segment_right_bound - segment_left_bound;
            operations[segment_idx] += val;
            values[segment_idx] += val;
            return;
        }
        propagate(segment_idx, segment_left_bound, segment_right_bound);
        int mid_point = (segment_left_bound + segment_right_bound) >> 1;
        int left_segment_idx = 2 * segment_idx + 1;
        int right_segment_idx = 2 * segment_idx + 2;
        update(segment_left_bound, mid_point, left_segment_idx, left, right, val);
        update(mid_point, segment_right_bound, right_segment_idx, left, right, val);
        values[segment_idx] = calc_op(values[left_segment_idx], values[right_segment_idx]);
    }

    void update(int left, int right, int val) {
        update(0, sz, 0, left, right, val);
    }

    int first_above(int segment_left_bound, int segment_right_bound, int segment_idx, int left, int x) {
        cout << segment_left_bound << " " << segment_right_bound << endl;
        if (segment_right_bound <= left) return -1;
        if (values[segment_idx] < x) return -1;
        if (is_leaf(segment_right_bound, segment_left_bound)) return segment_left_bound;
        int segment_len = segment_right_bound - segment_left_bound;
        propagate(segment_idx, segment_left_bound, segment_right_bound);
        int mid_point = (segment_left_bound + segment_right_bound) >> 1;
        int left_segment_idx = 2 * segment_idx + 1, right_segment_idx = 2 * segment_idx + 2;
        int res = first_above(segment_left_bound, mid_point, left_segment_idx, left, x);
        if (res == -1) {
            res = first_above(mid_point, segment_right_bound, right_segment_idx, left, x);
        }
        return res;
    }

    int first_above(int left, int x) {
        return first_above(0, sz, 0, left, x);
    }
};

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    int n, m;
    cin >> n >> m;
    SegmentTree seg;
    seg.init(n);
    int t, L, R, x;
    while (m--) {
        cin >> t;
        if (t == 1) {
            cin >> L >> R >> x;
            seg.update(L, R, x);
        } else {
            cin >> x >> L;
            cout << seg.first_above(L, x) << endl;
        }
    }
    return 0;
}
```

## A. Assignment, Addition, and Sum

### Solution 1:  lazy segment tree, two range update operaitons that are non-commutative with each other, range query is summation

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
    
def main():
    n, m = map(int, input().split())
    seg = LazySegmentTree(n, 0, 0)
    for _ in range(m):
        query = list(map(int, input().split()))
        t, L, R = query[0], query[1], query[2]
        if t == 1: # assign
            v = query[-1]
            seg.assign(L, R, v)
        elif t == 2: # add
            v = query[-1]
            seg.add(L, R, v)
        else: # query
            print(seg.query(L, R))

if __name__ == '__main__':
    main()
```

##

### Solution 1: 

```py

```

##

### Solution 1: 

```py

```


## Closest Equals

### D. Closest Equals

```c++
#include <bits/stdc++.h>
using namespace std;

inline int read()
{
	int x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}

int neutral = INT_MAX;

struct SegmentTree {
    vector<int> nodes;
    vector<int> next;
    int size;

    void init(int n, vector<int> &init_arr) {
        size = 1;
        while (size < n) size *= 2;
        nodes.assign(2 * size, neutral);
        next.assign(n, neutral);
        build(init_arr);
    }

    void build(vector<int> &init_arr) {
        unordered_map<int, int> last;
        for (int i = 0; i < init_arr.size(); i++) {
            int segment_idx = i + size -1;
            int val = init_arr[i];
            if (last.find(val) != last.end()) {
                nodes[segment_idx] = i - last[val];
                next[last[val]] = i;
            }
            last[val] = i;
            ascend(segment_idx);
        }
    }

    int calc_op(int x, int y) {
        return min(x, y);
    }

    void ascend(int segment_idx) {
        while (segment_idx > 0) {
            segment_idx--;
            segment_idx >>= 1;
            int left_segment_idx = 2 * segment_idx + 1, right_segment_idx = 2 * segment_idx + 2;
            nodes[segment_idx] = calc_op(nodes[left_segment_idx], nodes[right_segment_idx]);
        }
    }

    void update(int segment_idx, int val) {
        segment_idx += size - 1;
        nodes[segment_idx] = val;
        ascend(segment_idx);
    }

    int query(int left, int right) {
        vector<tuple<int, int, int>> stack;
        stack.push_back({0, size, 0});
        int result = neutral;
        while (!stack.empty()) {
            // BOUNDS FOR CURRENT INTERVAL AND IDX FOR TREE
            int segment_left_bound = get<0>(stack.back());
            int segment_right_bound = get<1>(stack.back());
            int segment_idx = get<2>(stack.back());
            stack.pop_back();
            // NO OVERLAP
            if (segment_left_bound >= right || segment_right_bound <= left) continue;
            // COMPLETE OVERLAP
            if (segment_left_bound >= left && segment_right_bound <= right) {
                result = calc_op(result, nodes[segment_idx]);
                continue;
            }
            // PARTIAL OVERLAP
            int left_segment_idx = 2 * segment_idx + 1, right_segment_idx = 2 * segment_idx + 2;
            int mid = (segment_left_bound + segment_right_bound) >> 1;
            stack.push_back({mid, segment_right_bound, right_segment_idx});
            stack.push_back({segment_left_bound, mid, left_segment_idx});
        }
        return result;
    }
};

int main() {
    int n = read(), m = read();
    vector<int> arr(n);
    for (int i = 0; i < n; i++) {
        arr[i] = read();
    }
    vector<tuple<int, int, int>> queries;
    for (int i = 0; i < m; i++) {
        int left = read(), right = read();
        queries.push_back({left-1, right-1, i});
    }
    sort(queries.begin(), queries.end());
    SegmentTree minSegTree;
    minSegTree.init(n, arr);
    vector<int> answer;
    answer.assign(m, -1);
    int index = 0;
    for (int i = 0; i < m; i++) {
        int left = get<0>(queries[i]);
        int right = get<1>(queries[i]);
        int id = get<2>(queries[i]);
        while (index < left) {
            if (minSegTree.next[index] != neutral) minSegTree.update(minSegTree.next[index], neutral);
            index++;
        }
        int query_res = minSegTree.query(left, right+1);
        if (query_res != neutral) answer[id] = query_res;
    }
    for (int ans : answer) {
        cout << ans << endl;
    }
}
```