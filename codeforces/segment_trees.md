
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

##

### Solution 1: 

```py

```

##

### Solution 1: 

```py

```

##

### Solution 1: 

```py

```

##

### Solution 1: 

```py

```

##

### Solution 1: 

```py

```

##

### Solution 1: 

```py

```

##

### Solution 1: 

```py

```

##

### Solution 1: 

```py

```

##

### Solution 1: 

```py

```