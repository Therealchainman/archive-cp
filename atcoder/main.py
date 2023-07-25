import os,sys
from io import BytesIO, IOBase
sys.setrecursionlimit(10**6)
from typing import *
# only use pypyjit when needed, it usese more memory, but speeds up recursion in pypy
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')
sys.stdout = open('output.txt', 'w')

# Fast IO Region
# BUFSIZE = 8192
# class FastIO(IOBase):
#     newlines = 0
#     def __init__(self, file):
#         self._fd = file.fileno()
#         self.buffer = BytesIO()
#         self.writable = "x" in file.mode or "r" not in file.mode
#         self.write = self.buffer.write if self.writable else None
#     def read(self):
#         while True:
#             b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
#             if not b:
#                 break
#             ptr = self.buffer.tell()
#             self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
#         self.newlines = 0
#         return self.buffer.read()
#     def readline(self):
#         while self.newlines == 0:
#             b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
#             self.newlines = b.count(b"\n") + (not b)
#             ptr = self.buffer.tell()
#             self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
#         self.newlines -= 1
#         return self.buffer.readline()
#     def flush(self):
#         if self.writable:
#             os.write(self._fd, self.buffer.getvalue())
#             self.buffer.truncate(0), self.buffer.seek(0)
# class IOWrapper(IOBase):
#     def __init__(self, file):
#         self.buffer = FastIO(file)
#         self.flush = self.buffer.flush
#         self.writable = self.buffer.writable
#         self.write = lambda s: self.buffer.write(s.encode("ascii"))
#         self.read = lambda: self.buffer.read().decode("ascii")
#         self.readline = lambda: self.buffer.readline().decode("ascii")
# sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)
# input = lambda: sys.stdin.readline().rstrip("\r\n")
# from sys import stdin
# input = stdin.readline

n, m = map(int, input().split())
a = [list(map(int, input().split())) for _ in range(n)]

cumsum = [[0] * (m + 1) for _ in range(n + 1)]
for x in range(n):
    for y in range(m):
        cumsum[x][y] = cumsum[x][y - 1] + a[x][y]
for x in range(n):
    for y in range(m):
        cumsum[x][y] += cumsum[x - 1][y]

ans = 0
for v in range(1, 301):
    print("==========================================================")
    print('v', v)
    h = [0] * m
    order = list(range(m))
    for x in range(n):
        new_order = []
        zero_columns = []
        done = [0] * (m + 1)
        left = list(range(m))
        right = list(range(m))
        print('order', order)
        for y in order:
            if a[x][y] < v:
                h[y] = 0
                zero_columns.append(y)
                continue
            else:
                h[y] += 1
                new_order.append(y)
            done[y] = 1
            l = left[y - 1] if done[y - 1] else y
            r = right[y + 1] if done[y + 1] else y
            print('y', y, y - 1, 'l', l, 'r', r, 'done', done)
            ans = max(ans, v * (cumsum[x][r] - cumsum[x][l - 1] - cumsum[x - h[y]][r] + cumsum[x - h[y]][l - 1]))
            print('ans', ans, 'h', h[y])
            right[l] = r
            left[r] = l
        order = new_order + zero_columns

print(ans)
sys.stdout.close()

# def main():
#     pass

# if __name__ == '__main__':
#     main()