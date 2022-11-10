# Range Queries

## Hotel Queries

### Solution 1:  segment tree + point updates + range query + first element greater than x

```py
import os,sys
from io import BytesIO, IOBase
from typing import List

# Fast IO Region
BUFSIZE = 8192
class FastIO(IOBase):
    newlines = 0
    def __init__(self, file):
        self._fd = file.fileno()
        self.buffer = BytesIO()
        self.writable = "x" in file.mode or "r" not in file.mode
        self.write = self.buffer.write if self.writable else None
    def read(self):
        while True:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            if not b:
                break
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines = 0
        return self.buffer.read()
    def readline(self):
        while self.newlines == 0:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            self.newlines = b.count(b"\n") + (not b)
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines -= 1
        return self.buffer.readline()
    def flush(self):
        if self.writable:
            os.write(self._fd, self.buffer.getvalue())
            self.buffer.truncate(0), self.buffer.seek(0)
class IOWrapper(IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        self.write = lambda s: self.buffer.write(s.encode("ascii"))
        self.read = lambda: self.buffer.read().decode("ascii")
        self.readline = lambda: self.buffer.readline().decode("ascii")
sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)
input = lambda: sys.stdin.readline().rstrip("\r\n")

"""
Segment Tree 

- point updates
- range query
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
            self.tree[segment_idx] = max(self.tree[left_segment_idx], self.tree[right_segment_idx])
        
    def update(self, segment_idx: int, val: int) -> None:
        segment_idx += self.size - 1
        self.tree[segment_idx] += val
        self.ascend(segment_idx)

    def leftmost_query(self, k: int) -> int:
        left_segment_bound, right_segment_bound, segment_idx = 0, self.size, 0
        while right_segment_bound - left_segment_bound != 1:
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            mid_point = (left_segment_bound + right_segment_bound) >> 1
            if k <= self.tree[left_segment_idx]: # continue in the left branch
                segment_idx, right_segment_bound = left_segment_idx, mid_point
            else: # continue in the right branch
                segment_idx, left_segment_bound = right_segment_idx, mid_point
        return left_segment_bound if self.tree[segment_idx] >= k else -1
    
    def __repr__(self) -> str:
        return f"array: {self.tree}"

def main():
    n, m = map(int, input().split())
    hotels = list(map(int,input().split()))
    segTree = SegmentTree(n, 0)
    for i in range(n):
        segTree.update(i, hotels[i])
    result = [0]*m
    queries = map(int, input().split())
    for i, guests in enumerate(queries):
        index = segTree.k_query(guests)
        # returns index = -1 if there is no hotel that can support the guests count, thus don't update for that
        if index >= 0:
            segTree.update(index, -guests) # remove the guests from the room that they could get
        result[i] = index + 1
    return ' '.join(map(str, result))

if __name__ == '__main__':
    print(main())
```

## List Removals

### Solution 1:  segment tree + point updates + kth query

```py
import os,sys
from io import BytesIO, IOBase
from typing import List

# Fast IO Region
BUFSIZE = 8192
class FastIO(IOBase):
    newlines = 0
    def __init__(self, file):
        self._fd = file.fileno()
        self.buffer = BytesIO()
        self.writable = "x" in file.mode or "r" not in file.mode
        self.write = self.buffer.write if self.writable else None
    def read(self):
        while True:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            if not b:
                break
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines = 0
        return self.buffer.read()
    def readline(self):
        while self.newlines == 0:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            self.newlines = b.count(b"\n") + (not b)
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines -= 1
        return self.buffer.readline()
    def flush(self):
        if self.writable:
            os.write(self._fd, self.buffer.getvalue())
            self.buffer.truncate(0), self.buffer.seek(0)
class IOWrapper(IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        self.write = lambda s: self.buffer.write(s.encode("ascii"))
        self.read = lambda: self.buffer.read().decode("ascii")
        self.readline = lambda: self.buffer.readline().decode("ascii")
sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)
input = lambda: sys.stdin.readline().rstrip("\r\n")

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

def main():
    n = int(input())
    arr = list(map(int,input().split()))
    segTree = SegmentTree(n, 0)
    for idx in range(n):
        segTree.update(idx, 1)
    result = [0]*n
    queries = map(int, input().split())
    for i, idx in enumerate(queries):
        index = segTree.k_query(idx)
        segTree.update(index, 0)
        result[i] = arr[index]
    return ' '.join(map(str, result))

if __name__ == '__main__':
    print(main())
```

## Salary Queries

### Solution 1:  segment tree + point updates + range queries + coordinate compression

Python Solution TLEs

```py
import os,sys
from io import BytesIO, IOBase
from typing import List

# Fast IO Region
BUFSIZE = 8192
class FastIO(IOBase):
    newlines = 0
    def __init__(self, file):
        self._fd = file.fileno()
        self.buffer = BytesIO()
        self.writable = "x" in file.mode or "r" not in file.mode
        self.write = self.buffer.write if self.writable else None
    def read(self):
        while True:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            if not b:
                break
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines = 0
        return self.buffer.read()
    def readline(self):
        while self.newlines == 0:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            self.newlines = b.count(b"\n") + (not b)
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines -= 1
        return self.buffer.readline()
    def flush(self):
        if self.writable:
            os.write(self._fd, self.buffer.getvalue())
            self.buffer.truncate(0), self.buffer.seek(0)
class IOWrapper(IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        self.write = lambda s: self.buffer.write(s.encode("ascii"))
        self.read = lambda: self.buffer.read().decode("ascii")
        self.readline = lambda: self.buffer.readline().decode("ascii")
sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)
input = lambda: sys.stdin.readline().rstrip("\r\n")

"""
Assignment Segment Tree 

- point updates
- range queries

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
        self.tree[segment_idx] += val
        self.ascend(segment_idx)
            
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
    
    def __repr__(self) -> str:
        return f"array: {self.tree}"

def main():
    n, q = map(int,input().split())
    p = list(map(int,input().split()))
    seen = set(p)
    query_indicator, update_indicator = '?', '!'
    coordCompressed = dict()
    values = []
    queries = [None]*q
    for i in range(q):
        query = input().split()
        if query[0] == update_indicator:
            seen.add(int(query[2]))
        else:
            seen.update(map(int, [query[1], query[2]]))
        queries[i] = query
    for i, v in enumerate(sorted(list(seen))):
        coordCompressed[v] = i
        values.append(v)
    segTree = SegmentTree(len(values), 0)
    for v in map(lambda v: coordCompressed[v], p):
        segTree.update(v, 1)
    result = []
    for query in queries:
        if query[0] == update_indicator:
            employee_idx, salary = map(int, query[1:])
            employee_idx -= 1 # 1 indexed
            segTree.update(coordCompressed[p[employee_idx]], -1)
            segTree.update(coordCompressed[salary], 1)
            p[employee_idx] = salary
        else:
            left, right = map(lambda v: coordCompressed[int(v)], query[1:])
            result.append(segTree.query(left, right + 1))
    return '\n'.join(map(str, result))

if __name__ == '__main__':
    print(main())
```

### Solution 2: segment tree

```cpp

```

## Pizzeria Queries

### Solution 1:  segment tree + range query + point update + min modification + math

```py
import os,sys
from io import BytesIO, IOBase
from typing import List
from collections import deque
from math import inf

# Fast IO Region
BUFSIZE = 8192
class FastIO(IOBase):
    newlines = 0
    def __init__(self, file):
        self._fd = file.fileno()
        self.buffer = BytesIO()
        self.writable = "x" in file.mode or "r" not in file.mode
        self.write = self.buffer.write if self.writable else None
    def read(self):
        while True:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            if not b:
                break
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines = 0
        return self.buffer.read()
    def readline(self):
        while self.newlines == 0:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            self.newlines = b.count(b"\n") + (not b)
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines -= 1
        return self.buffer.readline()
    def flush(self):
        if self.writable:
            os.write(self._fd, self.buffer.getvalue())
            self.buffer.truncate(0), self.buffer.seek(0)
class IOWrapper(IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        self.write = lambda s: self.buffer.write(s.encode("ascii"))
        self.read = lambda: self.buffer.read().decode("ascii")
        self.readline = lambda: self.buffer.readline().decode("ascii")
sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)
input = lambda: sys.stdin.readline().rstrip("\r\n")

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
            self.tree[segment_idx] = min(self.tree[left_segment_idx], self.tree[right_segment_idx])
        
    def update(self, segment_idx: int, val: int) -> None:
        segment_idx += self.size - 1
        self.tree[segment_idx] = val
        self.ascend(segment_idx)
            
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
                result = min(result, self.tree[segment_idx])
                continue
            # PARTIAL OVERLAP
            mid_point = (left_segment_bound + right_segment_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            stack.extend([(left_segment_bound, mid_point, left_segment_idx), (mid_point, right_segment_bound, right_segment_idx)])
        return result
    
    def __repr__(self) -> str:
        return f"array: {self.tree}"

def main():
    n, q = map(int, input().split())
    pizzerias = list(map(int, input().split()))
    update, range_, neutral = 1, 2, inf
    st_left = SegmentTree(n, neutral)
    st_right = SegmentTree(n, neutral)
    for i, p in enumerate(pizzerias):
        st_left.update(i, p-i)
        st_right.update(i, p+i)
    result = []
    for _ in range(q):
        query = list(map(int, input().split()))
        query_type = query[0]
        if query_type == range_:
            j = query[1] - 1
            left_query, right_query = st_left.query(0, j) + j, st_right.query(j, n) - j
            result.append(min(left_query, right_query))
        elif query_type == update:
            idx, val = query[1], query[2]
            idx -= 1
            st_left.update(idx, val-idx)
            st_right.update(idx, val+idx)
    return '\n'.join(map(str, result))

if __name__ == '__main__':
    print(main())

```

## Subarray Sum Queries

### Solution 1:  segment tree + range query + point update + (sum, max subarray sum, max subarray sum including left, max subarray sum including right)

```py
import os,sys
from io import BytesIO, IOBase
from typing import List
from collections import deque
from math import inf

# Fast IO Region
BUFSIZE = 8192
class FastIO(IOBase):
    newlines = 0
    def __init__(self, file):
        self._fd = file.fileno()
        self.buffer = BytesIO()
        self.writable = "x" in file.mode or "r" not in file.mode
        self.write = self.buffer.write if self.writable else None
    def read(self):
        while True:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            if not b:
                break
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines = 0
        return self.buffer.read()
    def readline(self):
        while self.newlines == 0:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            self.newlines = b.count(b"\n") + (not b)
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines -= 1
        return self.buffer.readline()
    def flush(self):
        if self.writable:
            os.write(self._fd, self.buffer.getvalue())
            self.buffer.truncate(0), self.buffer.seek(0)
class IOWrapper(IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        self.write = lambda s: self.buffer.write(s.encode("ascii"))
        self.read = lambda: self.buffer.read().decode("ascii")
        self.readline = lambda: self.buffer.readline().decode("ascii")
sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)
input = lambda: sys.stdin.readline().rstrip("\r\n")

class Node:
    def __init__(self):
        self.segment_max = -inf
        self.maxLeftSubArray = -inf
        self.maxRightSubArray = -inf
        self.segment_sum = 0

    def __repr__(self):
        return f"Node(max: {self.segment_max}, max_left: {self.maxLeftSubArray}, max_right: {self.maxRightSubArray}, sum: {self.segment_sum})"

class SegmentTree:
    def __init__(self, n: int, neutral: int):
        self.neutral = neutral
        self.size = 1
        self.n = n
        while self.size<n:
            self.size*=2
        self.tree = [Node() for _ in range(self.size*2)]

    def ascend(self, segment_idx: int) -> None:
        while segment_idx > 0:
            segment_idx -= 1
            segment_idx >>= 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            node = self.tree[segment_idx]
            node.segment_sum = self.tree[left_segment_idx].segment_sum + self.tree[right_segment_idx].segment_sum
            node.maxLeftSubArray = max(self.tree[left_segment_idx].maxLeftSubArray, self.tree[left_segment_idx].segment_sum + self.tree[right_segment_idx].maxLeftSubArray)
            node.maxRightSubArray = max(self.tree[right_segment_idx].maxRightSubArray, self.tree[right_segment_idx].segment_sum + self.tree[left_segment_idx].maxRightSubArray)
            node.segment_max = max(self.tree[left_segment_idx].segment_max, self.tree[right_segment_idx].segment_max, self.tree[left_segment_idx].maxRightSubArray + self.tree[right_segment_idx].maxLeftSubArray)
        
    def update(self, segment_idx: int, val: int) -> None:
        segment_idx += self.size - 1
        self.tree[segment_idx].segment_sum = val
        self.tree[segment_idx].segment_max = val
        self.tree[segment_idx].maxLeftSubArray = val
        self.tree[segment_idx].maxRightSubArray = val
        self.ascend(segment_idx)
            
    def query(self, left: int, right: int) -> int:
        stack = [(0, self.n, 0)]
        result = self.neutral
        while stack:
            # BOUNDS FOR CURRENT INTERVAL and idx for tree
            left_segment_bound, right_segment_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if left_segment_bound >= right or right_segment_bound <= left: continue
            # COMPLETE OVERLAP
            if left_segment_bound >= left and right_segment_bound <= right:
                result = max(result, self.tree[segment_idx].segment_max)
                continue
            # PARTIAL OVERLAP
            mid_point = (left_segment_bound + right_segment_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            stack.extend([(left_segment_bound, mid_point, left_segment_idx), (mid_point, right_segment_bound, right_segment_idx)])
        return result
    
    def __repr__(self) -> str:
        return f"array: {self.tree}"

def main():
    n, m = map(int, input().split())
    arr = list(map(int, input().split()))
    neutral = 0
    st = SegmentTree(n, neutral)
    for i, x in enumerate(arr):
        st.update(i, x)
    result = []
    for _ in range(m):
        idx, val = map(int, input().split())
        idx -= 1
        st.update(idx, val)
        result.append(st.query(0, n))
    return '\n'.join(map(str, result))

if __name__ == '__main__':
    print(main())
```

## Distinct Values Queries

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