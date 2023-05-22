import os,sys
from io import BytesIO, IOBase
from typing import *

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

sys.setrecursionlimit(1_000_000)
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')

def main():
    n, m = map(int, input().split())
    inside, outside, boundary = "INSIDE", "OUTSIDE", "BOUNDARY"
    outer_product = lambda v1, v2: v1[0]*v2[1] - v1[1]*v2[0]
    def is_boundary(p, p1, p2):
        # is p on the boundary of p1p2
        x, y = p
        x1, y1 = p1
        x2, y2 = p2
        v1 = (x2 - x1, y2 - y1)
        v2 = (x - x1, y - y1)
        return outer_product(v1, v2) == 0 and min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)
    def intersects(p1, p2, p3, p4):
        for _ in range(2):
            v1, v2, v3 = (p2[0]-p1[0], p2[1]-p1[1]), (p3[0]-p1[0], p3[1]-p1[1]), (p4[0]-p1[0], p4[1]-p1[1])
            outer_prod1 = outer_product(v1, v2)
            outer_prod2 = outer_product(v1, v3)
            if (outer_prod1 < 0 and outer_prod2 < 0) or (outer_prod1 > 0 and outer_prod2 > 0): return False
            p1, p2, p3, p4 = p3, p4, p1, p2
        return True
    polygon = []
    for _ in range(n):
        polygon.append(tuple(map(int, input().split())))
    points = []
    for _ in range(m):
        points.append(tuple(map(int, input().split())))
    res = []
    for p1 in points:
        x1, y1 = p1
        # L1 (x1, y1) -> (x2, y2)
        p2 = (10**9 + 7, y1 + 1)
        intersections = 0
        on_boundary = False
        for i in range(n):
            p3 = polygon[i]
            p4 = polygon[(i + 1)%n]
            # L2 (x3, y3) -> (x4, y4)
            if is_boundary(p1, p3, p4):
                on_boundary = True
                break
            if intersects(p1, p2, p3, p4):
                intersections += 1
        if on_boundary:
            res.append(boundary)
        elif intersections%2 == 0:
            res.append(outside)
        else:
            res.append(inside)
    return "\n".join(res)

if __name__ == '__main__':
    print(main())
    # T = int(input())
    # for _ in range(T):
    #     print(main())
