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
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')

"""
probability cell is empty is 1 - probability of robot existing on cell
"""
from itertools import product

def main():
    k = int(input())
    n = 8
    board = [[[0] * n for _ in range(n)] for _ in range(n * n)]
    neighborhood = lambda r, c: [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    in_bounds = lambda r, c: 0 <= r < n and 0 <= c < n
    for i in range(n * n):
        r, c = i // n, i % n
        board[i][r][c] = 1
    on_corner = lambda r, c: (r == 0 and c == 0) or (r == 0 and c == n - 1) or (r == n - 1 and c == 0) or (r == n - 1 and c == n - 1)
    on_boundary = lambda r, c: r == 0 or r == n - 1 or c == 0 or c == n - 1
    for _ in range(k):
        nboard = [[[0] * n for _ in range(n)] for _ in range(n * n)]
        for i, r, c in product(range(n * n), range(n), range(n)):
            p = 3 if on_boundary(r, c) else 4
            p = 2 if on_corner(r, c) else p
            for nr, nc in neighborhood(r, c):
                if in_bounds(nr, nc):
                    nboard[i][nr][nc] += board[i][r][c] / p
        board = nboard
    """
    probability that first robot is not in that cell at kth step, 1 - probability robot exists in that cell at kth step
    so it should be multiplied, because you want the probability of the sequence that robot1, robot2, robot3 are all not at that cell
    so how to do this for all.
    low*high = low 
    low*low = low
    high*high = high
    """
    res = [[1] * n for _ in range(n)]
    for i, r, c in product(range(n * n), range(n), range(n)):
        res[r][c] *= (1 - board[i][r][c])
    """
    expectation value is sum of all probabilities of each cell
    using linearity of expectation
    E[x+y] = E[x] + E[y
    that is expecation value of all cells is equal to expectation value of each cell that it is empty
    """
    sum_ = sum(sum(row) for row in res)
    print(f"{sum_:0.6f}")

if __name__ == '__main__':
    main()
    # T = int(input())
    # for _ in range(T):
    #     print(main())
