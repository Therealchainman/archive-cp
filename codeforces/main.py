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

from itertools import product

def count(n):
    digits = str(n)
    num_digits = len(digits)
    # dp(i, j, t), ith index in digits, j nonzero digits, t represents tight bound
    dp = [[[0]* 2 for _ in range(num_digits + 1)] for _ in range(num_digits + 1)]
    for i in range(int(digits[0]) + 1):
        dp[1][1 if i > 0 else 0][1 if i == int(digits[0]) else 0] += 1
    for i, t in product(range(1, num_digits), range(2)):
        for j in range(i + 1):
            for k in range(10): # digits
                if t and k > int(digits[i]): break
                dp[i + 1][j + (1 if k > 0 else 0)][t and k == int(digits[i])] += dp[i][j][t]
    return sum(dp[num_digits][j][t] for j, t in product(range(min(num_digits, 3) + 1), range(2)))

def main():
    left, right = map(int, input().split())
    res = count(right) - count(left - 1)
    print(res)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()