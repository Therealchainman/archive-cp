# Atcoder Beginner Contest 293

## What is used at the top of each submission

```py
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
```

## A - Swap Odd and Even 

### Solution 1: loop

```py
def main():
    s = input()
    n = len(s)
    arr = ['$'] + list(s)
    for i in range(1, n//2 + 1):
        arr[2*i], arr[2*i - 1] = arr[2*i - 1], arr[2*i]
    return ''.join(arr[1:])

if __name__ == '__main__':
    print(main())
```

## B - Call the ID Number

### Solution 1:  loop + memoized who is called

```py
def main():
    n = int(input())
    arr = [0] + list(map(int, input().split()))
    called = [0] * (n + 1)
    for i in range(1, n + 1):
        if called[i]: continue
        called[arr[i]] = 1
    res = [i for i in range(1, n + 1) if not called[i]]
    print(len(res))
    return ' '.join(map(str, res))
```

## C - Make Takahashi Happy 

### Solution 1:  bitmask + brute force enumerate all possible paths since it is 2^18 paths at most + O((h+w-2)2^(h+w-2)) time

Enumerate through all valid paths, by representing with 001100 in binary, where 0 is left and 1 is down.  Generate all paths by using bin function and zfill to pad with 0s to the left.  Then just have to check that path is valid by traversing the path and checking it never is out of bounds and contains only unique numbers.

```py
def main():
    h, w = map(int, input().split())
    matrix = [list(map(int, input().split())) for _ in range(h)]
    left, down = '0', '1'
    res = 0
    len_path = h + w - 2
    for mask in range(1 << len_path):
        vis = set([matrix[0][0]])
        path = bin(mask)[2:].zfill(len_path)
        r = c = 0
        valid_path = True
        for move in path:
            if move == left:
                c += 1
            else:
                r += 1
            if r >= h or c >= w or matrix[r][c] in vis: 
                valid_path = False
                break
            vis.add(matrix[r][c])
        res += valid_path
    return res

if __name__ == '__main__':
    print(main())
```

## D - Tying Rope 

### Solution 1:  undirected graph + degree + path graph + bfs + O(n + m) time

Consider the N ropes as N vertices of a graph, and connecting ropes a and b as an egdge connecting vertices a and b; then the problem is rephrased as follows.

Given a graph with N vertices and M edges.

You want to find the count of cycles and paths, since each component is either a cycle or a path.

A connected component is a cycle if and only if the degree of every vertex is two.  

The path graph P_n is a tree with two nodes of vertex degree 1, and the other n-2 nodes of vertex degree 2. A path graph is therefore a graph that can be drawn so that all of its vertices and edges lie on a single straight line. 

So if all the vertices have degree 2, then the graph is a not a path graph and a cycle.

```py
from collections import deque

def main():
    n, m = map(int, input().split())
    adj_list = [[] for _ in range(n)]
    degrees = [0]*n
    for _ in range(m):
        rope1, _, rope2, _ = input().split()
        rope1 = int(rope1) - 1
        rope2 = int(rope2) - 1
        adj_list[rope1].append(rope2)
        adj_list[rope2].append(rope1)
        degrees[rope1] += 1
        degrees[rope2] += 1
    visited = [0]*n
    def is_cyclic(rope):
        cycle = True
        queue = deque([rope])
        visited[rope] = 1
        while queue:
            rope = queue.popleft()
            if degrees[rope] != 2: # must be a path graph
                cycle = False
            for neighbor in adj_list[rope]:
                if visited[neighbor]: continue
                visited[neighbor] = 1
                queue.append(neighbor)
        return cycle
    num_paths = num_cycles = 0
    for rope in range(n):
        if visited[rope]: continue
        if is_cyclic(rope):
            num_cycles += 1
        else:
            num_paths += 1
    return f'{num_cycles} {num_paths}'

if __name__ == '__main__':
    print(main())
```

## E - Geometric Progression 

### Solution 1:  matrix exponentiation + mathematics + summation + transition state + O(logn) (n = number of terms)

```py
"""
matrix multiplication with modulus
"""
def mat_mul(mat1: List[List[int]], mat2: List[List[int]], mod: int) -> List[List[int]]:
    result_matrix = []
    for i in range(len(mat1)):
        result_matrix.append([0]*len(mat2[0]))
        for j in range(len(mat2[0])):
            for k in range(len(mat1[0])):
                result_matrix[i][j] += (mat1[i][k]*mat2[k][j])%mod
    return result_matrix

"""
matrix exponentiation with modulus
matrix is represented as list of lists in python
"""
def mat_pow(matrix: List[List[int]], power: int, mod: int) -> List[List[int]]:
    if power<=0:
        print('n must be non-negative integer')
        return None
    if power==1:
        return matrix
    if power==2:
        return mat_mul(matrix, matrix, mod)
    t1 = mat_pow(matrix, power//2, mod)
    if power%2 == 0:
        return mat_mul(t1, t1, mod)
    return mat_mul(t1, mat_mul(matrix, t1, mod), mod)

def main():
    base, num_terms, mod = map(int, input().split())
    # exponentiated_matrix*base_matrix = solution_matrix
    # exponentiated_matrix = transition_matrix^num_terms
    transition_matrix = [[base, 1], [0, 1]]
    base_matrix = [[0], [1]]
    exponentiated_matrix = mat_pow(transition_matrix, num_terms, mod)
    solution_matrix = mat_mul(exponentiated_matrix, base_matrix, mod)
    return solution_matrix[0][0]

if __name__ == '__main__':
    print(main())
```

## F - Zero or One 

I can't understand this one, it is very mathematical and using the bases of number theory.
Using number bases

I'll try again later to understand the proof to this one.  

```py

```