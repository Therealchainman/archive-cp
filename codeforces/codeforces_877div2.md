# Codeforces Round 877 Div 2

## Notes

if the implementation is in python it will have this at the top of the python script for fast IO operations

```py
import os,sys
from io import BytesIO, IOBase
from typing import *
import pypyjit
pypyjit.set_param('max_unroll_recursion=-1')
 
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
```

```cpp
#include <bits/stdc++.h>
using namespace std;
#define int long long

inline int read() {
	int x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}
```

## A - Game with Board

### Solution 1: 

```py
def main():
    n = int(input())
    if n < 5:
        print('Bob')
    else:
        print('Alice')
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## B - Keep it Beautiful

### Solution 1: 

```py
def main():
    q = int(input())
    arr = list(map(int, input().split()))
    res = [0] * q
    ascending = False
    prev = -1
    for i in range(q):
        if not ascending and arr[i] >= prev:
            res[i] = 1
            prev = arr[i]
        elif not ascending and arr[0] >= arr[i]:
            ascending = True
            res[i] = 1
            prev = arr[i]
        elif arr[i] >= prev and arr[0] >= arr[i]:
            res[i] = 1
            prev = arr[i]
    print(''.join(map(str, res)))

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## C - Ranom Numbers

### Solution 1: 

```py

```

## D. Pairs of Segments

### Solution 1:  dynamic programming + sort + coordinate compression + O(n^2 + nlogn) time

dp[i] = maximum number of pairs formed by the ith index, which corresponds with an endpoint of a segment. So when you are considering a segment, look through all th previous segments and find the minimum left endpoint index, and then take the dp value at that index and add 1, cause you are able to add another overlapping pair to some non-overlapping pairs from earlier. 

```py
from collections import defaultdict

def main():
    n = int(input())
    segments = [None] * n
    points = set()
    for i in range(n):
        left, right = map(int, input().split())
        segments[i] = (left, right)
        points.update([left, right])
    # sort in based on the right endpoint of the segment, then the left endpoint
    segments.sort(key = lambda x: (x[1], x[0]))
    m = len(points)
    # coordinate compression
    point_map = {}
    for i, p in enumerate(sorted(points), start = 1):
        point_map[p] = i
    # maximum number of pairs of segments that do not intersect at the ith index point which corresponds with 
    # a point on the number line
    dp = [0] * (m + 1)
    # the segments with endpoint at this index
    segments_at_index = defaultdict(list)
    for i, (_, right) in enumerate(segments):
        segments_at_index[point_map[right]].append(i)
    intersects = lambda a, b: max(segments[a][0], segments[b][0]) <= min(segments[a][1], segments[b][1])
    for i in range(1, m + 1):
        for segment_index in segments_at_index[i]:
            for j in range(segment_index):
                if intersects(segment_index, j):
                    leftmost_endpoint = min(segments[segment_index][0], segments[j][0])
                    leftmost_index = point_map[leftmost_endpoint]
                    dp[i] = max(dp[i], dp[leftmost_index - 1] + 1)
        dp[i] = max(dp[i], dp[i - 1])
    res = n - 2 * max(dp)
    print(res)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## E. Fill the Matrix

### Solution 1:  sparse table + range minimum query (rmq) + O(nlogn) time + divide and conquer + stack

Change the values of the array to store the n - x, because this will give you the number of empty cells in that column.  So before the goal would have been to process the cells with maximum non-empty cells.  Now you can change it to the problem of process the cell with minimum empty cells in current range and use that to divide into two segments.  So can use the rmq algorithm with O(1) lookups. But this has a slight variation, we need to track the index of these elements, so that can divide at that index into left and right segment for process later. 

The goal is to compute all the segments at most there will be n segments because it is going to be removing an element at each step.  Also need to track a decrement for that segment or range, that represents how many empty cells already been used and so for these ranges will need to be decrement by that value.  

Then wnat to sort the segments based on segment length because the best solution is to take the minimum number of segments. Each segment results in a loss of one value for the result. so take longest segments and continue processing until no more elements remain to be placed in the matrix. At this point you will have found minimum number of segments needed and just subtract that from m, so the result is m - minimum number of segments

```py
import math

"""
n is size of array input
range query is [left, right]
"""
class RMQ:
    def __init__(self, n, arr):
        self.lg = [0] * (n + 1)
        self.lg[1] = 0
        for i in range(2, n + 1):
            self.lg[i] = self.lg[i//2] + 1
        max_power_two = 18
        self.sparse_table = [[math.inf]*n for _ in range(max_power_two + 1)]
        for i in range(max_power_two + 1):
            j = 0
            while j + (1 << i) <= n:
                if i == 0:
                    self.sparse_table[i][j] = arr[j]
                else:
                    self.sparse_table[i][j] = min(self.sparse_table[i - 1][j], self.sparse_table[i - 1][j + (1 << (i - 1))])
                j += 1
                
    def query(self, left: int, right: int) -> int:
        length = right - left + 1
        power_two = self.lg[length]
        return min(self.sparse_table[power_two][left], self.sparse_table[power_two][right - (1 << power_two) + 1])

def main():
    n = int(input())
    arr = map(int, input().split())
    arr = [(n - x, i) for i, x in enumerate(arr)]
    m = int(input())
    rmq = RMQ(n, arr)
    # (left, right, range_decrement)
    stack = [(0, n - 1, 0)]
    segments = []
    while stack:
        left, right, range_decrement = stack.pop()
        if left > right: continue
        segment_len = right - left + 1
        val, idx = rmq.query(left, right)
        val -= range_decrement
        if val > 0:
            segments.append((segment_len, val))
        range_decrement += val
        stack.extend([(left, idx - 1, range_decrement), (idx + 1, right, range_decrement)])
    rem = m
    num_segments = 0
    for segment_len, repeat in sorted(segments, reverse = True):
        num_times = rem // segment_len + (rem % segment_len != 0)
        num_times = min(num_times, repeat)
        num_segments += num_times
        rem -= num_times * segment_len
        if rem <= 0: break
    res = m - num_segments
    print(res)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## F. Monocarp and a Strategic Game

### Solution 1: 

```py

```