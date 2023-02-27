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

import math

def main():
    n, m = map(int, input().split())
    teleporters = [''] + [input() for _ in range(n)]
    min_teleports = [math.inf]*n
    dist_from_start, dist_from_end = [math.inf]*(n + 1), [math.inf]*(n + 1)
    dist_from_start[1] = dist_from_end[n] = 0
    for i in range(2, n + 1):
        for j in range(max(1, i - m), i):
            if teleporters[j][i - j - 1] == '1':
                dist_from_start[i] = min(dist_from_start[i], dist_from_start[j] + 1)
    for i in range(n - 1, 0, -1):
        for j in range(i + 1, min(n + 1, i + m + 1)):
            if teleporters[i][j - i - 1] == '1':
                dist_from_end[i] = min(dist_from_end[i], dist_from_end[j] + 1)
    for k in range(2, n):
        for i in range(max(1, k - m), k):
            for j in range(k + 1, min(n + 1, i + m + 1)):
                if teleporters[i][j - i - 1] == '1':
                    min_teleports[k] = min(min_teleports[k], dist_from_start[i] + dist_from_end[j] + 1)
    min_teleports = [t if t < math.inf else -1 for t in min_teleports]
    return ' '.join(map(str, min_teleports[2:]))

if __name__ == '__main__':
    print(main())