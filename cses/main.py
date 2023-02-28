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
import math


def main():
    n, q = map(int, input().split())    
    arr = list(map(int, input().split()))
    lg = [0] * (n + 1)
    lg[1] = 0
    for i in range(2, n + 1):
        lg[i] = lg[i//2] + 1
    max_power_two = 18
    sparse_table = [[math.inf]*n for _ in range(max_power_two + 1)]
    for i in range(max_power_two + 1):
        j = 0
        while j + (1 << i) <= n:
            if i == 0:
                sparse_table[i][j] = arr[j]
            else:
                sparse_table[i][j] = min(sparse_table[i - 1][j], sparse_table[i - 1][j + (1 << (i - 1))])
            j += 1
    def query(left: int, right: int) -> int:
        length = right - left + 1
        power_two = lg[length]
        return min(sparse_table[power_two][left], sparse_table[power_two][right - (1 << power_two) + 1])
    res = []
    for _ in range(q):
        a, b = map(int, input().split())
        res.append(query(a - 1, b - 1))
    return '\n'.join(map(str, res))

if __name__ == '__main__':
    print(main())
