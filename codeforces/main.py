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

import math

def main():
    n = int(input())
    arr = list(map(int, input().split()))
    q = int(input())
    gcd_count = Counter()
    tmp_gcd_count = Counter()
    for val in arr:
        cur_gcd_count = Counter({val: 1})
        gcd_count[val] += 1
        for gcd, cnt in tmp_gcd_count.items():
            cur_gcd_count[math.gcd(val, gcd)] += cnt
            gcd_count[math.gcd(val, gcd)] += cnt
        tmp_gcd_count = cur_gcd_count
    res = [0]*q
    for i in range(q):
        x = int(input())
        res[i] = gcd_count[x]
    return '\n'.join(map(str, res))

if __name__ == '__main__':
    print(main())