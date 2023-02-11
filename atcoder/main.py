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

import heapq
import math
from collections import deque

def main():
    n, m = map(int, input().split())
    colors = [0] + list(map(int, input().split()))
    adj_list = [[] for _ in range(n + 1)]
    for _ in range(m):
        u, v = map(int, input().split())
        adj_list[u].append(v)
        adj_list[v].append(u)
    dist = [[math.inf]*(n + 1) for _ in range(n + 1)]
    dist[1][n] = 0
    queue = deque([(1, n, 0)])
    while queue:
        u, v, d = queue.popleft()
        if (u, v) == (n, 1): return d
        for nei_u in adj_list[u]:
            for nei_v in adj_list[v]:
                if colors[nei_u] == colors[nei_v]: continue
                if d + 1 < dist[nei_u][nei_v]:
                    dist[nei_u][nei_v] = d + 1
                    queue.append((nei_u, nei_v, d + 1))
    return -1

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        print(main())





from typing import *

class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
    
    def __repr__(self):
        return f'val: {self.val}, random val: {self.random.val if self.random else -1}'

def main(head: 'Node'):
    # INTERLEAVING OLD WITH NEW NODES
    cur = head
    while cur:
        new_node = Node(cur.val, cur.next)
        cur.next = new_node
        cur = cur.next.next
    # SETTING THE RANDOM POINTERS FOR NEW NODES
    cur = head
    while cur:
        new_node = cur.next
        if cur.random:
            new_node.random = cur.random.next
        cur = cur.next.next
    # SEPARATE THE NODES LISTS
    sentinel_node = Node(0)
    new_cur = sentinel_node
    cur = head
    # old -> new -> None
    while cur:
        new_node = cur.next
        new_cur.next = new_node 
        cur.next = cur.next.next
        cur = cur.next
        new_cur = new_cur.next
    return sentinel_node.next
        


def create_dataset(arr):
    nodes = []
    for u, _ in arr:
        nodes.append(Node(u))
    for node, (_, random_ptr) in zip(nodes, arr):
        if random_ptr is not None:
            node.random = nodes[random_ptr]
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
    return nodes[0]

if __name__ == '__main__':
    data1 = [[7,None],[13,0],[11,4],[10,2],[1,0]] 
    data2 = [[1,1],[2,1]]
    data3 = [[3,None],[3,0],[3,None]]
    dataset1, dataset2, dataset3 = map(create_dataset, [data1, data2, data3])
    main(dataset1)
    main(dataset2)
    main(dataset3)