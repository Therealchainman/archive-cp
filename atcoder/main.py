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

def main():
    n, q = map(int, input().split())
    degrees = [0] * (n + 1)
    adj_list = [set() for _ in range(n + 1)]
    cnt = n
    res = [None] * q
    for i in range(q):
        query = list(map(int, input().split()))
        if query[0] == 1:
            u, v = query[1:]
            cnt -= degrees[u] == 0
            cnt -= degrees[v] == 0
            degrees[u] += 1
            degrees[v] += 1
            adj_list[u].add(v)
            adj_list[v].add(u)
        else:
            u = query[1]
            cnt += degrees[u] > 0
            degrees[u] = 0
            for v in adj_list[u]:
                degrees[v] -= 1
                cnt += degrees[v] == 0
                adj_list[v].discard(u)
            adj_list[u].clear()
        res[i] = cnt
    return '\n'.join(map(str, res))

if __name__ == '__main__':
    print(main())
    # main()
    # sys.stdout.close()