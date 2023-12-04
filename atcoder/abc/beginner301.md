# Atcoder Beginner Contest 301

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

##

### Solution 1:  

```py

```

##

### Solution 1:  

```py

```

## C - AtCoder Cards

### Solution 1:  frequency arrays + counters + anagram

```py
from itertools import product
import math
import heapq
from collections import Counter
import string
 
def main():
    s = input()
    t = input()
    freq_s, freq_t = Counter(s), Counter(t)
    free_s, free_t = s.count('@'), t.count('@')
    chars = 'atcoder'
    for ch in string.ascii_lowercase:
        if freq_s[ch] != freq_t[ch] and ch not in chars: return 'No'
        if freq_s[ch] < freq_t[ch]:
            delta = freq_t[ch] - freq_s[ch]
            if free_s < delta: return 'No'
            free_s -= delta
        elif freq_s[ch] > freq_t[ch]:
            delta = freq_s[ch] - freq_t[ch]
            if free_t < delta: return 'No'
            free_t -= delta
    return 'Yes'
 
                    
if __name__ == '__main__':
    print(main())
```

## D - Bitmask 

### Solution 1:  bit manipulation + greedy

```py
def main():
    s = list(reversed(input()))
    n = int(input())
    res = 0
    for i in range(len(s)):
        if s[i] == '1':
            res |= (1 << i)
    if res > n: return -1
    for i in reversed(range(len(s))):
        if s[i] == '?' and (res | (1 << i)) <= n:
            res |= (1 << i)
    return res
if __name__ == '__main__':
    print(main())
```

## E - Pac-Takahashi 

### Solution 1:  traveling salesman problem + dp bitmask + (n^2 * 2^n) time

Given a list of cities and the distances between each pair of cities, what is the shortest possible route that visits each city exactly once and returns to the origin city?

minimize number of moves for the dp[i][mask], where i is the current node you are visiting and mask contains the set of all nodes that have been visited, which in this case will tell how many candies have been collected. 

Find the shortest distance between all pairs of vertices with bfs with the adjacency matrix

loop through each mask or set of visited vertices, then loop through the src and dst vertex and consider this. 

```py
from itertools import product
import math
from collections import deque

def main():
    H, W, T = map(int, input().split())
    A = [input() for _ in range(H)]
    start, goal, empty, wall, candy = ['S', 'G', '.', '#', 'o']
    in_bounds = lambda r, c: 0 <= r < H and 0 <= c < W
    neighborhood = lambda r, c: [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
    # CONSTRUCT THE VERTICES
    start_pos = goal_pos = None
    nodes = []
    index = {}
    for i, j in product(range(H), range(W)):
        if A[i][j] == start:
            start_pos = (i, j)
        elif A[i][j] == goal:
            goal_pos = (i, j)
        elif A[i][j] == candy:
            nodes.append((i, j))
    nodes.extend([goal_pos, start_pos])
    for i, (r, c) in enumerate(nodes):
        index[(r, c)] = i
    V = len(nodes)
    adj_matrix = [[math.inf]*V for _ in range(V)] # complete graph so don't need adjacency list
    # CONSTRUCT THE EDGES WITH BFS
    for i, (r, c) in enumerate(nodes):
        queue = deque([(r, c)])
        dist = 0
        vis = set([(r, c)])
        while queue:
            dist += 1
            for _ in range(len(queue)):
                row, col = queue.popleft()
                for nr, nc in neighborhood(row, col):
                    if not in_bounds(nr, nc) or A[nr][nc] == wall or (nr, nc) in vis:
                        continue
                    if (nr, nc) in index:
                        adj_matrix[i][index[(nr, nc)]] = dist
                    vis.add((nr, nc))
                    queue.append((nr, nc))
    # TRAVELING SALESMAN PROBLEM
    dp = [[math.inf]*(1 << (V - 1)) for _ in range(V)]
    dp[V - 1][0] = 0 # start at node 0 with no candy
    for mask in range(1 << (V - 1)):
        for i in range(V):
            if dp[i][mask] == math.inf: continue
            for j in range(V - 1):
                nmask = mask | (1 << j)
                dp[j][nmask] = min(dp[j][nmask], dp[i][mask] + adj_matrix[i][j])
    res = -1
    for mask in range(1 << (V - 1)):
        if dp[V - 2][mask] > T: continue
        res = max(res, bin(mask).count('1') - 1)
    return res
                    
if __name__ == '__main__':
    print(main())

```

##

### Solution 1:  

```py

```
 
## G - Worst Picture 

### Solution 1:  3 dimensional space geometry + computational geometry + line intersection in 3 dimensional space

This problem can be solved in O(n^3) time

summary of steps

1.  find lines from pairs of points
1.  add lines to `lines_of_interest` list if it is not parallel to the yz plane



A line is parallel to a plane if the direction vector of the line is orthogonal to the normal vector of the plane. So for the case when the line has a direction vector that is orthogonal to the normal vector of the yz plane.  Any line that would be parallal to the yz plane is not created because it will never intersect region x < 0.  In other words, you can check if a line would be like this by just looking at if the x1 = x2 of the points that the line goes through.  If they are equal, then the line is parallel to the yz plane and you can skip adding it to lines of interest.

good test case cause it include many points that can be colinear, see image of what it looks like in 3d space.  This image includes the line to the point of intersection in the x < 0  region where 4 lines an intersect and only 4 points are visible. 

```txt
11
1 1 1
1 1 -1
1 -1 1
1 -1 -1
3 2 2
3 2 -2
3 -2 2
3 -2 -2
5 3 3
7 4 4
9 5 5
```

![visualization of points](images/worst_picture1.png)



```py

import math
from itertools import product
from collections import defaultdict

def cross(u, v):
    """
    Returns the cross product of two 3D vectors u and v.
    """
    x = u[1] * v[2] - u[2] * v[1]
    y = u[2] * v[0] - u[0] * v[2]
    z = u[0] * v[1] - u[1] * v[0]
    return x, y, z

def norm(u):
    """
    Returns the norm of a 3D vector u.
    """
    return math.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)

def intersect(line1, line2):
    """
    Returns point if line segments line1 and line2 intersect in x < 0 region.
    returns None if the lines line1 and line2 do not intersect in x < 0 region or are parallel.
    """
    # Compute direction vectors and a point on each line
    v1 = (line1[1][0] - line1[0][0], line1[1][1] - line1[0][1], line1[1][2] - line1[0][2])
    # v1 = line1[1] - line1[0]
    p1 = line1[0]
    v2 = (line2[1][0] - line2[0][0], line2[1][1] - line2[0][1], line2[1][2] - line2[0][2])
    # v2 = line2[1] - line2[0]
    p2 = line2[0]

    # Compute normal vector to plane containing both lines
    n = cross(v1, v2)

    # Check if lines are parallel
    if norm(n) < 1e-6:
        return None

    # Compute intersection point
    p2p1 = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
    p1p2 = (p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])
    t1 = dot_product(cross(p2p1, v2), n) / dot_product(cross(v1, v2), n)
    t2 = dot_product(cross(p1p2, v1), n) / dot_product(cross(v2, v1), n)


    # point = p1 + t1 * v1
    # point2 = p2 + t2 * v2
    point1 = (p1[0] + t1*v1[0], p1[1] + t1*v1[1], p1[2] + t1*v1[2])
    point2 = (p2[0] + t2*v2[0], p2[1] + t2*v2[1], p2[2] + t2*v2[2])

    # check for skew lines
    if any(abs(v1 - v2) > 1e-6 for v1, v2 in zip(point1, point2)): return None

    # Check if intersection point is in x < 0 region
    if point1[0] < 0:
        return point1

    return None

def dot_product(u, v):
    """
    Returns the dot product of two 3D vectors u and v.
    """
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]

def parallel(line1, line2):
    """
    Returns True if line1 and line2 are parallel.
    """
    # Compute direction vectors of lines
    v1 = (line1[1][0] - line1[0][0], line1[1][1] - line1[0][1], line1[1][2] - line1[0][2])
    v2 = (line2[1][0] - line2[0][0], line2[1][1] - line2[0][1], line2[1][2] - line2[0][2])
    # v1 = line1[1] - line1[0]
    # v2 = line2[1] - line2[0]

    # Compute cross product of direction vectors
    norm_vec = cross(v1, v2)

    # Check if cross product is zero
    if norm(norm_vec) < 1e-6:
        return True

    return False

def main():
    n = int(input())
    points = [None] * n
    for i in range(n):
        x, y, z = map(int, input().split())
        points[i] = (x, y, z)
    # form all the lines that have delta_x != 0, else the line is in the yz plane, and will never cross into the x < 0 region.
    lines_of_interest = []
    blocked = 0
    counts = []
    for i, j in product(range(n), repeat = 2):
        p1, p2 = points[i], points[j]
        if p1[0] >= p2[0]: continue # must be not equal on x, and p2.x > p1.x
        line = [p1, p2]
        lines_of_interest.append(line)
        cnt = 1
        for k in range(n):
            p3 = points[k]
            if p3[0] <= p1[0] or p3[0] <= p2[0]: continue # requiring that p3.x > p2.x > p1.x
            p2p1 = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
            p3p1 = (p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2])
            normal_vector = cross(p2p1, p3p1)
            if norm(normal_vector) < 1e-6: cnt += 1
        counts.append(cnt)
        blocked = max(blocked, cnt)
    intersections = defaultdict(set)
    # find all the line intersections
    for i, j in product(range(len(lines_of_interest)), repeat = 2):
        line1, line2 = lines_of_interest[i], lines_of_interest[j]
        # point is None means in the region x < 0
        point = intersect(line1, line2)
        if point is None: continue
        good_i = good_j = True
        cur_lines = intersections[tuple(point)]
        not_good = []
        for k in cur_lines:
            line3 = lines_of_interest[k]
            if parallel(line1, line3):
                if counts[i] < counts[k]:
                    good_i = False
                else:
                    not_good.append(k)
            elif parallel(line2, line3):
                if counts[j] < counts[k]:
                    good_j = False
                else:
                    not_good.append(k)
        for k in not_good:
            cur_lines.discard(k)
        if good_i:
            cur_lines.add(i)
        if good_j:
            cur_lines.add(j)
    for intersecting_lines in intersections.values():
        blocked = max(blocked, sum(counts[i] for i in intersecting_lines))
    return n - blocked

if __name__ == '__main__':
    print(main())
```