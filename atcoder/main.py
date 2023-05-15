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
    lines = []
    blocked = 0
    counts = []
    for i, j in product(range(n), repeat = 2):
        p1, p2 = points[i], points[j]
        if p1[0] >= p2[0]: continue # must be not equal on x, and p2.x > p1.x
        line = [p1, p2]
        lines.append(line)
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
    for i, j in product(range(len(lines)), repeat = 2):
        line1, line2 = lines[i], lines[j]
        # point is None means in the region x < 0
        point = intersect(line1, line2)
        if point is None: continue
        good_i = good_j = True
        cur_lines = intersections[tuple(point)]
        not_good = []
        for k in cur_lines:
            line3 = lines[k]
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
    # main()
    # sys.stdout.close()