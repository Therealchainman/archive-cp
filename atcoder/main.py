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
from collections import deque
from itertools import zip_longest, product

mod = 998_244_353

def convolution(arr1: List[int], arr2: List[int]) -> List[int]:
    """
    Convolution of two sequences of numbers modulo mod
    """
    n = len(arr1)
    m = len(arr2)
    res = [0] * (n + m - 1)
    for i, j in product(range(n), range(m)):
        res[i + j] += arr1[i] * arr2[j]
        res[i + j] %= mod
    return res

def add_sequence(arr1: List[int], arr2: List[int]) -> List[int]:
    """
    Adds two sequences of numbers modulo mod
    """
    return [(a + b) % mod for a, b in zip_longest(arr1, arr2, fillvalue = 0)]

def merge(arrs: List[List[int]]) -> List[int]:
    """
    Merges a list of sequences of numbers modulo mod
    """
    res = [1]
    if len(arrs) == 0: return res # base case
    queue = deque(arrs)
    while len(queue) > 1:
        arr1 = queue.popleft()
        arr2 = queue.popleft()
        queue.append(convolution(arr1, arr2))
    return queue[0]

def main():
    n = int(input())
    adj_list = [[] for _ in range(n)]
    for _ in range(n - 1):
        u, v = map(int, input().split())
        u -= 1
        v -= 1
        adj_list[u].append(v)
        adj_list[v].append(u)
    children = [[] for _ in range(n)]
    dist = [math.inf]*n
    dist[0] = 0 # root tree at node 0
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for child in adj_list[node]:
            if dist[child] == math.inf:
                dist[child] = dist[node] + 1
                children[node].append(child)
                queue.append(child)
    # sorted in decreasing distance from root node 0, descending order based on distance
    vertex = sorted(range(n), key=lambda x: dist[x], reverse=True)
    # memo[i][0][j] corresponds to the number of ways to form j connected components with the subtree rooted at i when skipping node i
    # memo[i][1][j] corresponds to the number of ways to form j connected components with the subtree rooted at i when not skipping node i
    memo = [[[], []] for _ in range(n)] 
    for v in vertex:
        # merge all the children's values for number of ways to form up j connected components
        # shen skipping node v you can just add the number of ways 
        memo[v][0] = merge([add_sequence(memo[child][0], memo[child][1]) for child in children[v]])
        """
        example to understand this logic, given simple example

        n1
        |
        n2

        n1 has child node n2

        suppose
        component_ways_for_keeping_node_n2 =  [0, 10, 5, 5]
        component_ways_for_skipping_node_n2 = [1, 10, 5, 1]
                                   components  0, 1,  2, 3

        then what should be the transition state to find these for node n1, given we know for node n2?
        component_ways_for_keeping_node_n1 
        =
        [0, 10, 5, 5]
        [0, 1, 10, 5, 1]
        = [0, 11, 15, 10, 1]
        basically, if you skipped n2, and you are keeping n1, then you are incrementing number of components by 1, so you shift the entire array to the right by 1.
        So now what was for 1 component, is now for 2 components and added for when you keep n2

        component_ways_for_skipping_node_n1
        =
        [0, 10, 5, 5]
        [1, 10, 5, 1]
        = [1, 11, 10, 6]
        """ 
        memo[v][1] = [0] + merge([add_sequence(memo[child][0], memo[child][1][1:]) for child in children[v]])
    ans = [0]*(n + 1)
    # add number of ways for when including node and not including node 0
    for take in [0, 1]:
        # j corresponds to number of components
        # ways corresponds to number of ways to form j components
        for j, ways in enumerate(memo[0][take]):
            ans[j] += ways
            ans[j] %= mod
    print(*ans[1:], sep = '\n')

if __name__ == '__main__':
    main()