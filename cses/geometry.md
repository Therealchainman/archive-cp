# Geometry

## USED IN SUBMISSIONS

```py
import os,sys
from io import BytesIO, IOBase
from typing import *
sys.setrecursionlimit(1_000_000)
# only use pypyjit when needed, it usese more memory, but speeds up recursion in pypy
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

inline long long readll() {
	long long x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}
```

## Point Location Test

### Solution 1:  outer product test

```py
def main():
    x1, y1, x2, y2, x3, y3 = map(int, input().split())
    outer_product = lambda v1, v2: v1[0]*v2[1] - v1[1]*v2[0]
    line_vector = (x2-x1, y2-y1)
    point_vector = (x3-x1, y3-y1)
    outer_prod = outer_product(line_vector, point_vector)
    if outer_prod > 0:
        print('LEFT')
    elif outer_prod < 0:
        print('RIGHT')
    else:
        print("TOUCH")
        
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## Line Segment Intersection

### Solution 1:  outer product

The easiest thing is to check if it doesn't intersect, so for instance if the both points of a line segment are on the same side of the other line segment, then they don't intersect.  You can do this for both so you swap the points and do from other perspective. 

Then you need to check when the outer product is 0 for both which indicates colinear lines.  You need to just check if the lines overlap with each other by taking the max of one and min of other, It is guaranteed to not overlap if the max is less than the min of a single coordinate, either x or y coordinate.  Check each dimension independently. Imagine if the max x element is less than the min x element from another line segment then they do not intersect.

```py
def main():
    x1, y1, x2, y2, x3, y3, x4, y4 = map(int, input().split())
    p1, p2, p3, p4 = (x1, y1), (x2, y2), (x3, y3), (x4, y4)
    outer_product = lambda v1, v2: v1[0]*v2[1] - v1[1]*v2[0]
    def intersects(p1, p2, p3, p4):
        for _ in range(2):
            v1, v2, v3 = (p2[0]-p1[0], p2[1]-p1[1]), (p3[0]-p1[0], p3[1]-p1[1]), (p4[0]-p1[0], p4[1]-p1[1])
            outer_prod1 = outer_product(v1, v2)
            outer_prod2 = outer_product(v1, v3)
            if (outer_prod1 < 0 and outer_prod2 < 0) or (outer_prod1 > 0 and outer_prod2 > 0): return False
            if outer_prod1 == outer_prod2 == 0 and (max(p1[0], p2[0]) < min(p3[0], p4[0])) or (max(p1[1], p2[1]) < min(p3[1], p4[1])): return False
            p1, p2, p3, p4 = p3, p4, p1, p2
        return True
    return "YES" if intersects(p1, p2, p3, p4) else "NO"
        
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        print(main())
```

## Polygon Area

### Solution 1:

```py

```

## Point in Polygon

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