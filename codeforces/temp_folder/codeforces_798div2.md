# Codeforces Round 780 div3

## Summary

## Lex String

### Solution 1: Greedy + always take 1 to k depending if it becomes not beneficial

```py
def main():
    n, m, k = map(int,input().split())
    s_arr = [sorted(list(input()), reverse=True), sorted(list(input()), reverse=True)]
    index = 0^(s_arr[0][-1]>s_arr[1][-1])
    result = []
    while len(s_arr[0]) > 0 and len(s_arr[1]) > 0:
        for i in range(k):
            if len(s_arr[index]) == 0: break
            if i > 0 and s_arr[index][-1] > s_arr[index^1][-1]: break
            result.append(s_arr[index].pop())
        index^=1
    return ''.join(result)
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        print(main())
```

## Mystic Permutation

### Solution 1: Stack for the result for the edge case + Consider best and second option

```py
def main():
    n = int(input())
    P = list(map(int,input().split()))
    sP = sorted(P, reverse=True)
    stack = []
    for i in range(n):
        best_option = sP.pop()
        if P[i] != best_option:
            stack.append(best_option)
        elif sP:
            second_option = sP.pop()
            stack.append(second_option)
            sP.append(best_option)
        elif not stack:
            return -1
        else:
            last = stack.pop()
            stack.append(best_option)
            stack.append(last)
    return ' '.join(map(str,stack))
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        print(main())
```

## Infected Nodes

### Solution 1: BFS through binary tree + represent binary tree as undirected graph with adjacency list

```py
import os,sys
from io import BytesIO, IOBase
from collections import deque
 
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
    n = int(input())
    graph = [[] for _ in range(n+1)]
    for _ in range(n-1):
        u, v = map(int,input().split())
        graph[u].append(v)
        graph[v].append(u)
    
    children = [0]*(n+1)
    children[1] = len(graph[1])
    for i in range(2,n+1):
        children[i] = len(graph[i])-1
    
    depth = [0]*(n+1)
    queue = deque([(0,1)])
    while queue:
        parent, node = queue.popleft()
        for child in graph[node]:
            if child == parent: continue
            queue.append((node, child))
            depth[child] = depth[node] + 1
    min_casualties = n
    for i in range(1,n+1):
        if children[i] < 2:
            cur_casualties = 2*depth[i] + children[i] + 1
            min_casualties = min(min_casualties, cur_casualties)
    return n-min_casualties
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        print(main())
```