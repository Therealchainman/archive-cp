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

# sys.setrecursionlimit(1_000_000)

"""
Euler Tour Technique to a random tree

relabel or index of the tree nodes


Necessary and sufficient conditions

An undirected graph has a closed Euler tour if and only if it is connected and each vertex has an even degree.

An undirected graph has an open Euler tour (Euler path) if it is connected, and each vertex, except for exactly two vertices, has an even degree. The two vertices of odd degree have to be 
the endpoints of the tour.

A directed graph has a closed Euler tour if and only if it is strongly connected and the in-degree of each vertex is equal to its out-degree.

Similarly, a directed graph has an open Euler tour (Euler path) if and only if for each vertex the difference between its in-degree and out-degree is 0, except for two vertices, 
where one has difference +1 (the start of the tour) and the other has difference -1 (the end of the tour) and, if you add an edge from the end to the start, the graph is strongly connected.

Definition 13.1.1.  A walk is closed if it begins and ends with the same vertex.
A trail is a walk in which no two vertices appear consecutively (in either order) more than once. (That is, no edge is used more than once.)

A tour is a closed trail.

An Euler trail is a trail in which every pair of adjacent vertices appear consecutively. (That is, every edge is used exactly once.)

An Euler tour is a closed Euler trail.
"""

class FenwickTree:
    def __init__(self, N):
        self.sums = [0 for _ in range(N+1)]

    def update(self, i, delta):
        while i < len(self.sums):
            self.sums[i] += delta
            i += i & (-i)

    def query(self, i):
        res = 0
        while i > 0:
            res += self.sums[i]
            i -= i & (-i)
        return res

    def __repr__(self):
        return f"array: {self.sums}"

def main():
    n, q = map(int, input().split())
    arr = list(map(int, input().split()))


if __name__ == '__main__':
    main()
