# Atcoder Beginner Contest 299

## What is used at the top of each submission

```py
import os,sys
from io import BytesIO, IOBase
sys.setrecursionlimit(10**6)
from typing import *
# only use pypyjit when needed, it usese more memory, but speeds up recursion in pypy
import pypyjit
pypyjit.set_param('max_unroll_recursion=-1')
 
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
```

## A - Treasure Chest

### Solution 1:  loop + string index and rindex

```py
def main():
    n = int(input())
    s = input()
    first, last = s.index('|'), s.rindex('|')
    for i in range(n):
        if s[i] == '*' and first < i < last:
            return 'in'
    return 'out'
 
if __name__ == '__main__':
    print(main())
```

## B - 	Trick Taking

### Solution 1:  max

custom max based on if correct color and maximized on rank but return the player or index

```py
def main():
    n, t = map(int, input().split())
    colors = list(map(int, input().split()))
    ranks = list(map(int, input().split()))
    if t not in colors:
        t = colors[0]
    if t in colors:
        return max(range(n), key = lambda i: ranks[i] if colors[i] == t else 0) + 1
 
if __name__ == '__main__':
    print(main())
```

## C - 	Dango

### Solution 1:  sliding window

The tricky part to this one is that anytime you see a - character you need to move the left = right + 1, cause you only want to count the o characters.  Also another thing to be careful of is when -oooo, which should be level 4 dango.  This means you need that if left > 0 condition to check for when there appeard a - before the o characters. even in this case it is important oooo-oooooo, cause the best one is after that last - character.

```py
def main():
    n = int(input())
    s = input()
    left = res = 0
    for right in range(n):
        if s[right] == '-':
            res = max(res, right - left)
            left = right + 1
    if left > 0:
        res = max(res, n - left)
    res = res if res > 0 else -1
    print(res)
 
if __name__ == '__main__':
    main()
```

## D - Find by Query

### Solution 1:  binary search

Can binary search because guaranteed that s1 = 0 and sn = 1, so really just need to find the rightmost 0 in a sense that precedes a 1. kind of local maximum idea, cause the truth is if you are looking for last T 

0010110000011
TTFTFFTTTTTFF
so you might say this doesn't work for binary search, but it is fine, if it get's stuck in a local region that region will still contain some form of 
T...TF...F regardless

```py
def main():
    n = int(input())
    left, right = 1, n - 1
    while left < right:
        mid = (left + right + 1) >> 1
        print(f"? {mid}", flush = True)
        resp = int(input())
        if resp == 0:
            left = mid
        else:
            right = mid - 1
    print(f"! {left}", flush = True)
 
if __name__ == '__main__':
    main()
```

## E - Nearest Black Vertex

### Solution 1:  max heap + multisource bfs + memoization 

1. max heap to paint all nodes white that would not satisfy the minimum distance to a node painted black.
1. multisource bfs to find that the minimum distance to every black node is still equal to minimum distance

set all nodes to be painted black, than set up a max heap based on the distance required from the current node that all nodes need to be painted white. Then paint all then nodes white. By using the max heap and storing the max_dist, it prevents it from recomputing on nodes cause it will have already painted on neighbor nodes farther than distance from that node.  the max heap makes certain you use the larger distances first. 

After painting all the nodes that need to be white, you just need to check that there is at least one remaining painted black node, and perform a multisource bfs from each black node and record the minimum distance to every node in the graph. If the minimum distance is greater than the required minimum distance for node then you need to return False, there is no solution.  What this means is there was no valid way to have a node painted black at the minimum distance and satisfy the distance requirements of all the nodes. 

To understand this further the best thing to do is draw out a simple undirected graph and have a few nodes black.  Although some reason I just got this one and didn't need to do a crazy proof. All I did was think about the previous two steps above.  

One concern I had was that I can just iterate from every node that has a minimun distance to a node painted black. Because there would be so much recomputation, if I visite a node 2 times that doesn't make sense, I made observation that if ai visit a node a second time but the remaining distance to the nearest node painted black is less than the previous time, there is no reason to revisit, cause the previous visit will reach farther and all the nodes that are needed to be painted white.  I also realize the best way to guarantee that it uses the larger distances first is to use a max heap.  That will avoid recomputation.  Cause if a node has distance = 5, and another one distance = 2, I will explore from the distance = 5 first, until it becomes smaller than or equal to distance = 2, and on the same playing field.  Therefore technically the first time I visit a node it is guaranteed to also be the largest distance.  This means I probably don't need the max_dist array but just to store visited. 

Now that I've painted all the nodes white that need to be white, cause if they were not the minimum distance to a node painted black would be too small.  Now you just need to check that the minimum distances are correct and in the course of doing this you didn't make it that from a node the minimum distance to a node painted black is now greater than what is required.  Which could happen because of the requirements of another node prevents it.  BFS implemented to do this because can store minimum number of edges traversed as I explore out from all the nodes painted black. 

```py
from collections import deque
import heapq
import math

def main():
    n, m = map(int, input().split())
    adj_list = [[] for _ in range(n)]
    for _ in range(m):
        u, v = map(int, input().split())
        adj_list[u-1].append(v-1)
        adj_list[v-1].append(u-1)
    k = int(input())
    max_heap = []
    max_dist = [0]*n
    min_dist = [math.inf]*n
    for _ in range(k):
        p, d = map(int, input().split())
        p -= 1
        min_dist[p] = d
        if d > 0:
            heapq.heappush(max_heap, (-d, p))
            max_dist[p] = d
    result = [1]*n
    while max_heap:
        dist, node = heapq.heappop(max_heap)
        result[node] = 0
        dist = -dist
        if dist < max_dist[node]: continue
        for nei in adj_list[node]:
            if dist - 1 <= max_dist[nei]: continue
            max_dist[nei] = dist - 1
            heapq.heappush(max_heap, (-(dist-1), nei))
    def bfs():
        vis = [0]*n
        queue = deque()
        for i in range(n):
            if result[i] == 1:
                queue.append(i)
                vis[i] = 1
        dist = 0
        while queue:
            for _ in range(len(queue)):
                node = queue.popleft()
                if dist > min_dist[node]: return False
                for nei in adj_list[node]:
                    if vis[nei]: continue
                    vis[nei] = 1
                    queue.append(nei)
            dist += 1
        return True
    if sum(result) == 0 or not bfs():
        print('No')
        return
    print('Yes')
    print(''.join(map(str, result)))

if __name__ == '__main__':
    main()
```

##  F - Square Subsequence

### Solution 1:

```py

```

## G - Minimum Permutation

### Solution 1:

```py

```