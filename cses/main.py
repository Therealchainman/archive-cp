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

def main():
    n, m = map(int, input().split())
    adj_list = [[] for _ in range(n + 1)]
    for _ in range(m):
        a, b = map(int, input().split())
        adj_list[a].append(b)
    time = num_scc = 0
    scc_ids = [0]*(n + 1)
    disc, low, on_stack = [0]*(n + 1), [0]*(n + 1), [0]*(n + 1)
    stack = []
    def dfs(node):
        nonlocal time, num_scc
        time += 1
        disc[node] = time
        low[node] = disc[node]
        on_stack[node] = 1
        stack.append(node)
        for nei in adj_list[node]:
            if not disc[nei]: dfs(nei)
            if on_stack[nei]: low[node] = min(low[node], low[nei])
        # found scc
        if disc[node] == low[node]:
            num_scc += 1
            while stack:
                snode = stack.pop()
                on_stack[snode] = 0
                low[snode] = low[node]
                scc_ids[snode] = num_scc
                if snode == node: break
    for i in range(1, n + 1):
        if disc[i]: continue
        dfs(i)
    print(num_scc)
    print(*scc_ids[1:])

if __name__ == '__main__':
    main()
