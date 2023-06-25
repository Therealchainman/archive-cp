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

from itertools import product

# i corresponds to the row, and also the position on the y axis
# j corresponds to the column, and also the position on the x axis
def convert(H, W, grid):
    s = set()
    for i, j in product(range(H), range(W)):
        if grid[i][j] == '#':
            s.add((i, j))
    return s

"""
convert everything to the first quadrant, and with minimim black squares on x = 0 and y = 0
"""
def normalize(s):
    min_x, min_y = min(x for y, x in s), min(y for y, x in s)
    return set((y - min_y, x - min_x) for y, x in s)

def main():
    n = int(input())
    HA, WA = map(int, input().split())
    A = normalize(convert(HA, WA, [input() for _ in range(HA)]))
    HB, WB = map(int, input().split())
    B = normalize(convert(HB, WB, [input() for _ in range(HB)]))
    HX, WX = map(int, input().split())
    X = normalize(convert(HX, WX, [input() for _ in range(HX)]))
    res = False
    for dx, dy in product(range(-HX, HX + 1), range(-WX, WX + 1)):
        res |= normalize(A.union(set((y + dy, x + dx) for y, x in B))) == X
    print('Yes' if res else 'No')

if __name__ == '__main__':
    main()