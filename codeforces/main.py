import os,sys
from io import BytesIO, IOBase
from typing import *
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')
 
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

"""
n is size of array input
range query is [left, right]
"""
class RMQ:
    def __init__(self, n, arr):
        self.lg = [0] * (n + 1)
        self.lg[1] = 0
        for i in range(2, n + 1):
            self.lg[i] = self.lg[i//2] + 1
        max_power_two = 18
        self.sparse_table = [[math.inf]*n for _ in range(max_power_two + 1)]
        for i in range(max_power_two + 1):
            j = 0
            while j + (1 << i) <= n:
                if i == 0:
                    self.sparse_table[i][j] = arr[j]
                else:
                    self.sparse_table[i][j] = min(self.sparse_table[i - 1][j], self.sparse_table[i - 1][j + (1 << (i - 1))])
                j += 1
                
    def query(self, left: int, right: int) -> int:
        length = right - left + 1
        power_two = self.lg[length]
        return min(self.sparse_table[power_two][left], self.sparse_table[power_two][right - (1 << power_two) + 1])

import bisect

def main():
    n = int(input())
    arr = list(map(int, input().split()))
    rmq = RMQ(n, arr)
    smax = [0] * (n + 1)
    for i in reversed(range(n)):
        smax[i] = max(smax[i + 1], arr[i])  
    print('smax', smax)
    def binary_search(start, target):
        left, right = start, n - 1
        while left < right:
            mid = (left + right) >> 1
            if smax[mid] <= target:
                right = mid
            else:
                left = mid + 1
        return left
    pmax = 0
    for i in range(n - 2):
        pmax = max(pmax, arr[i])
        j = binary_search(i + 2, pmax)
        mmin = rmq.query(i + 1, j - 1)
        if pmax == smax[j] == mmin:
            print("YES")
            x, y, z = i + 1, j - i - 1, n - j
            print(x, y, z)
            return
    print("NO")
    
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()