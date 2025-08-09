# Range Queries

## Static Range Minimum Queries

### Solution 1:  RMQ + sparse tables + O(nlogn) precompute sparse tables + O(1) query since the ranges can overlap without affecting the in

```py
sys.setrecursionlimit(1_000_000)
import math


def main():
    n, q = map(int, input().split())    
    arr = list(map(int, input().split()))
    lg = [0] * (n + 1)
    lg[1] = 0
    for i in range(2, n + 1):
        lg[i] = lg[i//2] + 1
    max_power_two = 18
    sparse_table = [[math.inf]*n for _ in range(max_power_two + 1)]
    for i in range(max_power_two + 1):
        j = 0
        while j + (1 << i) <= n:
            if i == 0:
                sparse_table[i][j] = arr[j]
            else:
                sparse_table[i][j] = min(sparse_table[i - 1][j], sparse_table[i - 1][j + (1 << (i - 1))])
            j += 1
    def query(left: int, right: int) -> int:
        length = right - left + 1
        power_two = lg[length]
        return min(sparse_table[power_two][left], sparse_table[power_two][right - (1 << power_two) + 1])
    res = []
    for _ in range(q):
        a, b = map(int, input().split())
        res.append(query(a - 1, b - 1))
    return '\n'.join(map(str, res))

if __name__ == '__main__':
    print(main())
```

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
    def __init__(self, num_elements: int, neutral: int):
        self.neutral = neutral
        self.size = 1
        self.num_elements
        while self.size<num_elements:
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
        stack = [(0, self.num_elements, 0)]
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

### Solution 1:  offline queries, Mo's algorithm, sorting, odd-even trick, coordinate compression, frequency array

```cpp
int N, Q, cnt;
vector<int> A, values, freq;

int block_size;

struct Query {
    int l, r, idx;

    bool operator<(const Query &other) const {
        int b1 = l / block_size, b2 = other.l / block_size;
        if (b1 != b2) return b1 < b2;
        if (b1 & 1) return r > other.r;
        return r < other.r;
    }
};

void remove(int idx) {
    if (--freq[A[idx]] == 0) --cnt;
}

void add(int idx) {
    if (++freq[A[idx]] == 1) ++cnt;
}

int getAnswer() {
    return cnt;
}

vector<int> mo_s_algorithm(vector<Query> queries) {
    block_size = max(1, (int)(N / max(1.0, sqrt(Q))));
    vector<int> answers(Q);
    sort(queries.begin(), queries.end());
    cnt = 0;
    freq.assign(values.size(), 0);
    int curL = 0, curR = -1;
    for (const Query& q : queries) {
        while (curL > q.l) add(--curL);
        while (curR < q.r) add(++curR);
        while (curL < q.l) { remove(curL); ++curL; }
        while (curR > q.r) { remove(curR); --curR; }
        answers[q.idx] = getAnswer();
    }
    return answers;
}

vector<Query> queries;

void solve() {
    cin >> N >> Q;
    A.resize(N);
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
        values.emplace_back(A[i]);
    }
    sort(values.begin(), values.end());
    values.erase(unique(values.begin(), values.end()), values.end());
    for (int i = 0; i < N; ++i) {
        A[i] = lower_bound(values.begin(), values.end(), A[i]) - values.begin();
    }
    for (int i = 0; i < Q; ++i) {
        int l, r;
        cin >> l >> r;
        --l, --r;
        queries.emplace_back(l, r, i);
    }
    vector<int> ans = mo_s_algorithm(queries);
    for (int x : ans) {
        cout << x << endl;
    }
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Solution 2:  fenwick tree, coordinate compression, sorting, offline queries

1. The trick is to scan from left to right for the right endpoint of the query, and track the latest occurrence of each value using a Fenwick tree.
1. This way you can query the number of distinct values in the range [l, r] efficiently, cause it will be 0 if that value appears later in the current interval ending at r.  

```cpp
int N, Q;
vector<int> A, values;

struct Query {
    int l, r, idx;

    bool operator<(const Query &other) const {
        return r < other.r;
    }
};

template <typename T>
struct FenwickTree {
    vector<T> nodes;
    T neutral;

    FenwickTree() : neutral(T(0)) {}

    void init(int n, T neutral_val = T(0)) {
        neutral = neutral_val;
        nodes.assign(n + 1, neutral);
    }

    void update(int idx, T val) {
        while (idx < (int)nodes.size()) {
            nodes[idx] += val;
            idx += (idx & -idx);
        }
    }

    T query(int idx) {
        T result = neutral;
        while (idx > 0) {
            result += nodes[idx];
            idx -= (idx & -idx);
        }
        return result;
    }

    T query(int left, int right) {
        return right >= left ? query(right) - query(left - 1) : T(0);
    }
};

void solve() {
    cin >> N >> Q;
    A.resize(N);
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
        values.emplace_back(A[i]);
    }
    sort(values.begin(), values.end());
    values.erase(unique(values.begin(), values.end()), values.end());
    for (int i = 0; i < N; ++i) {
        A[i] = lower_bound(values.begin(), values.end(), A[i]) - values.begin();
    }
    FenwickTree<int> fenwick;
    fenwick.init(N);
    vector<Query> queries;
    vector<int> latest(values.size(), -1);
    for (int i = 0; i < Q; ++i) {
        int l, r;
        cin >> l >> r;
        --l, --r;
        queries.emplace_back(l, r, i);
    }
    sort(queries.begin(), queries.end());
    vector<int> ans(Q);
    for (int i = 0, j = 0; i < N; ++i) {
        int idx = A[i];
        if (latest[idx] != -1) {
            fenwick.update(latest[idx] + 1, -1);
        }
        latest[idx] = i;
        fenwick.update(i + 1, 1);
        while (j < Q && queries[j].r == i) {
            ans[queries[j].idx] = fenwick.query(queries[j].l + 1, queries[j].r + 1);
            ++j;
        }
    }
    for (int x : ans) {
        cout << x << endl;
    }
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Distinct Values Queries II

### Solution 1:  point update, range query segment tree, next pointer, sets, coordinate compression, 

```cpp
const int INF = numeric_limits<int>::max();
int N, Q;
vector<int> A, nxt;
vector<set<int>> pos;

struct Query {
    int x, y, z;
};

struct SegmentTree {
    int size;
    int neutral = INF;
    vector<int64> nodes;

    void init(int num_nodes) {
        size = 1;
        while (size < num_nodes) size *= 2;
        nodes.assign(size * 2, 0);
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
    void update(int segment_idx, int64 val) {
        segment_idx += size;
        nodes[segment_idx] = val;
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

SegmentTree seg;

void setNext(int idx, int val) {
    nxt[idx] = val;
    seg.update(idx, val);
}

void remove(int idx, int val) {
    set<int> &s = pos[val];
    auto it = s.find(idx);
    int pred = -1, succ = INF;
    if (it != s.begin()) {
        pred = *prev(it);
    }
    auto nextIt = next(it);
    if (nextIt != s.end()) succ = *nextIt;
    if (pred != -1) {
        setNext(pred, succ);
    }
    setNext(idx, INF);
    s.erase(it);
}

void add(int idx, int val) {
    set<int> &s = pos[val];
    auto it = s.lower_bound(idx);
    int pred = -1, succ = INF;
    if (it != s.begin()) pred = *prev(it);
    if (it != s.end()) succ = *it;
    s.insert(idx);
    if (pred != -1) setNext(pred, idx);
    setNext(idx, succ);
}

void solve() {
    cin >> N >> Q;
    A.resize(N);
    vector<int> values;
    for (int i = 0; i < N; i++) {
        cin >> A[i];
        values.emplace_back(A[i]);
    }
    vector<Query> queries;
    while (Q--) {
        int x, y, z;
        cin >> x >> y >> z;
        if (x == 1) values.emplace_back(z);
        queries.emplace_back(x, y, z);
    }
    sort(values.begin(), values.end());
    values.erase(unique(values.begin(), values.end()), values.end());
    int M = values.size();
    for (int i = 0; i < N; ++i) {
        A[i] = lower_bound(values.begin(), values.end(), A[i]) - values.begin();
    }
    for (Query &q : queries) {
        if (q.x == 1) {
            q.y--;
            q.z = lower_bound(values.begin(), values.end(), q.z) - values.begin();
        } else {
            q.y--, q.z--;
        }
    }
    nxt.assign(N, INF);
    pos.assign(M, set<int>());
    for (int i = 0; i < N; ++i) {
        pos[A[i]].insert(i);
    }
    // calculate the nxt array
    for (int i = 0; i < M; ++i) {
        set<int> &s = pos[i];
        for (auto it = s.begin(); it != s.end(); ++it) {
            auto nextIt = next(it);
            if (nextIt != s.end()) nxt[*it] = *nextIt;
        }
    }
    seg.init(N);
    for (int i = 0; i < N; ++i) {
        seg.update(i, nxt[i]);
    }
    for (auto &[x, y, z] : queries) {
        if (x == 1) {
            remove(y, A[y]);
            A[y] = z;
            add(y, A[y]);
        } else {
            int ans = seg.query(y, z);
            if (ans > z) {
                cout << "YES" << endl;
            } else {
                cout << "NO" << endl;
            }
        }
    }
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Increasing Array Queries

### Solution 1:

```py

```

## Forest Queries II

### Solution 1:

```py

```

##  Range Updates and Sums

### Solution 1:  lazy segment tree + two range updates and lazy operations + range queries

```py
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

def main():
    n, m = map(int, input().split())
    arr = list(map(int, input().split()))
    neutral = 0
    noop = 0
    st = LazySegmentTree(n, neutral, noop, arr)
    results = []
    for _ in range(m):
        query = list(map(int, input().split()))
        if query[0] == 1:
            # range increment update
            _, left, right, val = query
            left -= 1
            right -= 1
            st.update(left, right+1, val, 'add')
        elif query[0] == 2:
            # range assign update
            _, left, right, val = query
            left -= 1
            right -= 1
            st.update(left, right+1, val, 'assign')
        else:
            # range sum query 
            _, left, right = query
            left -= 1
            right -= 1
            results.append(st.query(left, right+1))
    return '\n'.join(map(str,results))

if __name__ == '__main__':
    print(main())
```

## Range XOR Queries

### Solution 1:  prefix xor sum

```cpp
int N, Q;
vector<int> arr, pxor;

int xor_sum(int l, int r) {
    return pxor[r] ^ (l > 0 ? pxor[l - 1] : 0);
}

signed main() {
    cin >> N >> Q;
    arr.resize(N);
    pxor.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> arr[i];
        pxor[i] = arr[i];
        if (i > 0) pxor[i] ^= pxor[i - 1];
    }
    while (Q--) {
        int l, r;
        cin >> l >> r;
        l--; r--;
        cout << xor_sum(l, r) << endl;
    }
}
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