# Atcoder Beginner Contest 306

## What is used at the top of each submission

```py
import os,sys
from io import BytesIO, IOBase
sys.setrecursionlimit(10**6)
from typing import *
# only use pypyjit when needed, it usese more memory, but speeds up recursion in pypy
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')
# sys.stdout = open('output.txt', 'w')

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
                    
if __name__ == '__main__':
    print(main())
    # main()
    # sys.stdout.close()
```

## A - Echo 

### Solution 1: 

```py
def main():
    n = int(input())
    s = input()
    res = []
    for ch in s:
        res.extend([ch] * 2)
    res = ''.join(res)
    print(res)

if __name__ == '__main__':
    main()
```

## B - Base 2 

### Solution 1: 

```py
def main():
    arr = list(map(int, input().split()))
    res = 0
    n = len(arr)
    for i in range(n):
        res += (arr[i] << i)
    print(res)

if __name__ == '__main__':
    main()
```

## C - Centers 

### Solution 1: 

```py
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    indices = [[] for _ in range(n + 1)]
    for i, x in enumerate(arr, start = 1):
        indices[x].append(i)
    res = sorted(range(1, n + 1), key = lambda x: indices[x][1])
    print(' '.join(map(str, res)))

if __name__ == '__main__':
    main()
```

## D - Poisonous Full-Course 

### Solution 1: 

```py
import math

def main():
    n = int(input())
    anti, pois = 0, 1
    arr = [None] * n
    for i in range(n):
        x, y = map(int, input().split())
        arr[i] = (x, y)
    dp = [[-math.inf] * 2 for _ in range(n + 1)]
    dp[0][0] = 0
    for i in range(n):
        x, y = arr[i]
        # skipping the course
        dp[i + 1][0] = dp[i][0]
        dp[i + 1][1] = dp[i][1]
        if x == anti:
            dp[i + 1][0] = max(dp[i + 1][0], dp[i][1] + y, dp[i][0] + y)
        else:
            dp[i + 1][1] = max(dp[i + 1][1], dp[i][0] + y)
    print(max(dp[-1]))

if __name__ == '__main__':
    main()
```

## E - Best Performances 

### Solution 1: 

```py
import heapq

def main():
    n, k, q = map(int, input().split())
    queries = [None] * q
    for i in range(q):
        x, y = map(int, input().split())
        x -= 1
        queries[i] = (x, y)
    arr = [0] * n # current value
    queried = [0] * n # number of times queried
    activity_level = [0] * n # if in k largest elements or not
    min_heap, max_heap = [], [(0, i, 0) for i in range(n)] # (value, index, query id)
    heapq.heapify(max_heap)
    sum_ = num_active = 0
    cnt = 0
    for i, y in queries:
        cnt += 1
        delta = y - arr[i]
        arr[i] = y
        queried[i] += 1
        if num_active < k:
            num_active += (activity_level[i] == 0)
            activity_level[i] = 1
            sum_ += delta
            heapq.heappush(min_heap, (arr[i], i, queried[i]))
        else:
            if activity_level[i] == 0:
                sum_ += arr[i]
            else:
                sum_ += delta
            heapq.heappush(min_heap, (arr[i], i, queried[i]))
            num_active += (activity_level[i] == 0)
            activity_level[i] = 1
            # balance them
            while min_heap and queried[min_heap[0][1]] != min_heap[0][2]:
                heapq.heappop(min_heap)
            while max_heap and queried[max_heap[0][1]] != max_heap[0][2]:
                heapq.heappop(max_heap)
            # swap
            if min_heap and max_heap and abs(max_heap[0][0]) > min_heap[0][0]:
                v1, i1, q1 = heapq.heappop(min_heap)
                v2, i2, q2 = heapq.heappop(max_heap)
                v2 = abs(v2)
                sum_ += v2 - v1
                activity_level[i1] = 0
                activity_level[i2] = 1
                heapq.heappush(max_heap, (-v1, i1, q1))
                heapq.heappush(min_heap, (v2, i2, q2))
            # balance them
            while min_heap and queried[min_heap[0][1]] != min_heap[0][2]:
                heapq.heappop(min_heap)
            while max_heap and queried[max_heap[0][1]] != max_heap[0][2]:
                heapq.heappop(max_heap)
            # remove one from active
            if num_active > k:
                v, i, q = heapq.heappop(min_heap)
                activity_level[i] = 0
                num_active -= 1
                sum_ -= v
                heapq.heappush(max_heap, (-v, i, q))
        print(sum_)

if __name__ == '__main__':
    main()
```

## F - Merge Sets 

### Solution 1: 

```py

```

## G - Return to 1 

### Solution 1: 

```py

```