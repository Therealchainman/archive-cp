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
    n = int(input())
    adj_list = [[] for _ in range(n)]
    for _ in range(n - 1):
        u, v = map(int, input().split())
        u -= 1
        v -= 1
        adj_list[u].append(v)
        adj_list[v].append(u)
    leaf_lens1, leaf_lens2 = [0] * n, [0] * n
    path_node1, path_node2 = [-1] * n, [-1] * n
    def dfs1(node: int, parent: int) -> int:
        for child in adj_list[node]:
            if child == parent: continue
            leaf_len = dfs1(child, node)
            if leaf_len > leaf_lens1[node]:
                leaf_lens2[node] = leaf_lens1[node]
                path_node2[node] = path_node1[node]
                leaf_lens1[node] = leaf_len
                path_node1[node] = child
            elif leaf_len > leaf_lens2[node]:
                leaf_lens2[node] = leaf_len
                path_node2[node] = child
        return leaf_lens1[node] + 1
    dfs1(0, -1)
    parent_lens = [0] * n
    def dfs2(node: int, parent: int) -> None:
        parent_lens[node] = parent_lens[parent] + 1 if parent != -1 else 0
        if parent != -1 and node != path_node1[parent]:
            parent_lens[node] = max(parent_lens[node], leaf_lens1[parent] + 1)
        if parent != -1 and node != path_node2[parent]:
            parent_lens[node] = max(parent_lens[node], leaf_lens2[parent] + 1)
        for child in adj_list[node]:
            if child == parent: continue
            dfs2(child, node)
    dfs2(0, -1)
    res = [max(leaf, pleaf) for leaf, pleaf in zip(leaf_lens1, parent_lens)]
    print(*res)

if __name__ == '__main__':
    main()
