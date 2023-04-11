import os,sys
from io import BytesIO, IOBase
sys.setrecursionlimit(10**6)
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

def main():
    n = int(input())
    points = [tuple(map(int, input().split())) for _ in range(n)]
    line_segments = []
    vertical_line_segments = {}
    for i in range(1, n + 1):
        (x1, y1), (x2, y2) = points[i-1], points[i%n]
        if x2 < x1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        if x1 != x2:
            line_segments.append((x1, x2, y1, y2))
        else:
            if y2 < y1: y1, y2 = y2, y1
            vertical_line_segments[x1] = (y1, y2)
    line_segments.sort()
    q = int(input())
    queries = []
    for i in range(q):
        a, b = map(int, input().split())
        queries.append((a, b, i))
    queries.sort()
    ans = ["OUT"]*q
    ptr = 1
    inner_product = lambda v1, v2: sum(x1*x2 for x1, x2 in zip(v1, v2))
    outer_product = lambda v1, v2: v1[0]*v2[1] - v1[1]*v2[0]
    def on_line_segment(outer_product, line_vector, point_vector):
        if outer_product != 0: return False
        line_inner_product = inner_product(line_vector, line_vector)
        point_inner_product = inner_product(line_vector, point_vector)
        return 0 <= point_inner_product <= line_inner_product
    for a, b, i in queries:
        # ADVANCE LINE SEGMENT POINTER
        while ptr + 1 < len(line_segments) and line_segments[ptr][0] < a:
            ptr += 1
        # CHECK IF POINT IS IN POLYGON
        line_seg1, line_seg2 = line_segments[ptr-1], line_segments[ptr]
        # print('line_seg1', line_seg1, 'line_seg2', line_seg2)
        # print('a', a, 'b', b)
        if not (line_seg1[0] <= a <= line_seg1[1] and line_seg2[0] <= a <= line_seg2[1]):
            continue
        # print('a', a, 'b', b)
        line_vector1 = (line_seg1[1]-line_seg1[0], line_seg1[3]-line_seg1[2])
        line_vector2 = (line_seg2[1]-line_seg2[0], line_seg2[3]-line_seg2[2])
        point_vector1 = (a-line_seg1[0], b-line_seg1[2])
        point_vector2 = (a-line_seg2[0], b-line_seg2[2])
        # p1, p2, p3, p4 = (line_seg1[0], line_seg1[2]), (line_seg1[1], line_seg1[3]), (line_seg2[0], line_seg2[2]), (line_seg2[1], line_seg2[3])
        outer_product1 = outer_product(line_vector1, point_vector1)
        outer_product2 = outer_product(line_vector2, point_vector2)
        # COLINEAR WITH A LINE SEGMENT
        if on_line_segment(outer_product1, line_vector1, point_vector1) or on_line_segment(outer_product2, line_vector2, point_vector2):
            ans[i] = "ON"
            continue
        # SAME SIDE OF LINE SEGMENT
        orientation1, orientation2 = outer_product1 > 0, outer_product2 > 0
        # BETWEEN LINE SEGMENTS MEANS INSIDE POLYGON
        if orientation1^orientation2:
            ans[i] = "IN"
            # CHECK COLINEAR WITH VERTICAL LINE SEGMENT
            if a in vertical_line_segments:
                y1, y2 = vertical_line_segments[a]
                # print('y1', y1, 'y2', y2)
                if y1 <= b <= y2:
                    ans[i] = "ON"
    return '\n'.join(ans)

if __name__ == '__main__':
    print(main())

"""
5
0 4
-2 2
-2 0
-1 0
3 1
7
-1 3
0 2
2 0
-2 1
-3 1
4 1
3 1

on
in
out
on
out
out
on

"""