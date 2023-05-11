import os,sys
from io import BytesIO, IOBase
from typing import *
# only use pypyjit when needed, it usese more memory, but speeds up recursion in pypy
sys.setrecursionlimit(1_000_000)
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

from collections import defaultdict, Counter
import math

def main():
    n = int(input())
    values = [0] + list(map(int, input().split()))
    adj_list = [[] for _ in range(n + 1)]
    for _ in range(n - 1):
        u, v = map(int, input().split())
        adj_list[u].append(v)
        adj_list[v].append(u)
    # store minimum operations (node, xor_sum): min operations for this state
    min_ops = defaultdict(lambda: math.inf)
    # a leaf node cannot be the root node, and guarantee tree will have two nodes
    is_leaf = lambda node, parent: len(adj_list[node]) == 1 and parent != -1
    def dfs(node, parent):
        # base case
        if is_leaf(node, parent):
            min_ops[(node, values[node])] = 0
            return values[node]
        freq = Counter()
        operation_count = 0
        for child in adj_list[node]:
            if child == parent: continue
            val = dfs(child, node)
            ops = min_ops[(child, val)]
            freq[val] += 1
            operation_count += ops
        # update the values
        total_cost = operation_count + sum(freq.values()) - max(freq.values())
        max_value = max(freq.keys(), key = lambda x: freq[x])
        new_val = max_value ^ values[node]
        min_ops[(node, new_val)] = total_cost
        min_ops[(node, 0)] = total_cost + (1 if values[node] != new_val else 0)
        return new_val
    dfs(1, -1)
    return min_ops[(1, 0)]

if __name__ == '__main__':
    # T = int(input())
    # for _ in range(T):
        # print(main())
    print(main())