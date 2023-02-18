import os,sys
from io import BytesIO, IOBase
from typing import *
# sys.setrecursionlimit(1_000_000)

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
    n, k = map(int, input().split())
    cand_segments = [None]*n
    for i in range(n):
        cand_segments[i] = tuple(map(int, input().split()))
    segments = []
    for left, right in cand_segments:
        if left <= k <= right: 
            segments.append((left, right))
    events = []
    for left, right in segments:
        events.append((left, 1))
        events.append((right + 1, -1))
    events.sort()
    counts = [0]*52
    for ev, incr in events:
        counts[ev] += incr
    delta = max_count_other = 0
    for i in range(1, 51):
        delta += counts[i]
        if i != k:
            max_count_other = max(max_count_other, delta)
        counts[i] = delta
    return 'YES' if counts[k] > max_count_other else 'NO'
if __name__ == '__main__':
    t = int(input())
    for _ in range(t):
        print(main())