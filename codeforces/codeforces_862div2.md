# Codeforces Round 862 Div 2

## Notes

if the implementation is in python it will have this at the top of the python script for fast IO operations

```py
import os,sys
from io import BytesIO, IOBase
from typing import *
sys.setrecursionlimit(10**6)
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

```cpp
#include <bits/stdc++.h>
using namespace std;

inline int read()
{
	int x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}

inline long long readll() {
	long long x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}
```

## A. We Need the Zero

### Solution 1:  bit manipulation + xor

It is simple if it is even number of elements it will only work if frequency of all bits from the array is already even, represent by sum(cnts) = 0 because using xor to track if even or odd, so if all 0 that represents all bits are even. and you can always choose the 0 to xor with. 
But when it comes to the n is odd, then you want to find all the bits frequency that are odd, and use that to build your xor, but really all you need to do is xor all the elements to find what you need to xor with. 

```py
import operator
from functools import reduce
 
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    cnts = [0]*8
    for num in arr:
        for i in range(8):
            if (num>>i)&1:
                cnts[i] ^= 1
    if n%2 == 0:
        return 0 if sum(cnts) == 0 else -1
    else:
        return reduce(operator.xor, arr)
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        print(main())
```

## B. The String Has a Target

### Solution 1:  string

The idea is find the smallest character smaller than first character, and find the last index and move that one to the start of the string.

```py
def main():
    n = int(input())
    s = input()
    last_index = -1
    cur_char = 'z'
    for i in range(1, n):
        if s[i] <= s[0] and s[i] <= cur_char:
            last_index = i
            cur_char = s[i]
    return s if last_index == -1 else s[last_index] + s[:last_index] + s[last_index+1:]
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        print(main())
```

## C. Place for a Selfie

### Solution 1:  math + quadratic equation + determinant + binary search

using the property of the discrimant you want to find the case when the discrimant is negative because those are lines that don't intersect parabola

A line can meet a parabola in at most two points.
If the discriminant of the resulting quadratic is negative, the line does not meet the parabola at all.
If the discriminant of the resulting quadratic is positive, the line meets the parabola in two distinct points.
If the discriminant of the resulting quadratic is zero, then the line is tangent to the parabola.

Steps done to solve is to set the kx = ax^2+bx+c, then this leads to you want to solve for this quadratic equation 0 = ax^2+(b-k)x+c. 
Then what you can do is  get the determinant and you have b^2+4ax < 0.  This gives a range of values for b and therfore k, slope of the lines that will not intersect with the current parabola.  So just use left and right to store that range that is valid.   Then just binary search through the slopes to find a line that doesn't intersect the parabola. bisect_right so that it is guaranteed to be greater if it exists, then just need to check it is less than right. 

```py
import math
import bisect
 
def main():
    n, m = map(int, input().split())
    lines = [0]*n
    for i in range(n):
        lines[i] = int(input())
    lines.sort()
    parabolas = [None]*m
    for i in range(m):
        a, b, c = map(int, input().split())
        left = b - 2*math.sqrt(a*c) if a*c >= 0 else math.inf
        right = b + 2*math.sqrt(a*c) if a*c >= 0 else -math.inf
        parabolas[i] = (left, right)
    ans = [None]*m
    for i in range(m):
        left, right = parabolas[i]
        j = bisect.bisect_right(lines, left)
        if j < n and lines[j] < right:
            ans[i] = lines[j]
    for i in range(m):
        if ans[i] is not None:
            print('Yes')
            print(ans[i])
        else:
            print('No')
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## D. A Wide, Wide Graph

### Solution 1:  reroot tree + two dfs to compute maximum distance from each node + iterate from k = n to 1 and find how many more elements absorb into the hive connected component 

The idea here is that initially when k > max_distance between any node then all nodes belong to single connected component.  But once you reach some nodes that will be connected together they will form a connecte component, from here forth any other nodes that connect to this component will reduce the total count of connected components.  so for when k = y it and suppose that max distance between nodes is x it is something like number of connected components = n - (freq(y) + freq(y+1) + ... + freq(x)) + 1.  this computes how many nodes are in the super or hive of connected components and additionally how many nodes have still not joined that large connected component that is absorbing them all.

The other thing is you use the two dfs to find the maximum distance from a node to another node in the tree.  

unfortunately the python version fails with recursion because of mle with pypy in codeforces

```py
def main():
    n = int(input())
    adj_list = [[] for _ in range(n + 1)]
    for _ in range(n - 1):
        u, v = map(int, input().split())
        adj_list[u].append(v)
        adj_list[v].append(u)
    ans = [n] * (n + 1)
    freq = [0] * (n + 1)
    max_dist_subtree1 = [0] * (n + 1)
    max_dist_subtree2 = [0] * (n + 1)
    child1 = [0] * (n + 1)
    child2 = [0] * (n + 1)
    # PHASE 1: DFS to find the max distance from each node to the leaf in it's subtree but find first and second max distance
    def dfs1(node, parent):
        for child in adj_list[node]:
            if child == parent: continue
            max_dist_subtree = dfs1(child, node)
            if max_dist_subtree > max_dist_subtree1[node]:
                max_dist_subtree2[node] = max_dist_subtree1[node]
                child2[node] = child1[node]
                max_dist_subtree1[node] = max_dist_subtree
                child1[node] = child
            elif max_dist_subtree > max_dist_subtree2[node]:
                max_dist_subtree2[node] = max_dist_subtree
                child2[node] = child
        return max_dist_subtree1[node] + 1
    dfs1(1, 0)
    parent_max_dist = [-1] * (n + 1)
    # PHASE 2: 
    def dfs2(node, parent):
        parent_max_dist[node] = parent_max_dist[parent] + 1
        if parent != 0:
            parent_max_dist[node] = max(parent_max_dist[node], max_dist_subtree1[parent] + 1) if node != child1[parent] else max(parent_max_dist[node], max_dist_subtree2[parent] + 1)
        for child in adj_list[node]:
            if child == parent: continue
            dfs2(child, node)
    dfs2(1, 0)
    # PHASE 3: compute the frequency for each max distance for each node
    for i in range(1, n + 1):
        freq[max(max_dist_subtree1[i], parent_max_dist[i])] += 1
    suffix_freq = 0
    for i in range(n, 0, -1):
        suffix_freq += freq[i]
        if suffix_freq > 0:
            ans[i] = n - suffix_freq + 1
    print(*ans[1:])

if __name__ == '__main__':
    main()
```

This solution actually worked, note that memset is particular you can't use memset to set array to n, it is different, only works with values that are known at compile time I think.  So that's why it works with the -1 for parent_max_dist. can use fill to fill the array instead and only need to fill up to the n + 1. 

This is really just same as code above but implemented in C++. 

```cpp
const int MAXN = 200'005;

int n, ans[MAXN], freq[MAXN], max_dist_subtree1[MAXN], max_dist_subtree2[MAXN], child1[MAXN], child2[MAXN], parent_max_dist[MAXN];
vector<int> adj_list[MAXN];

int dfs1(int node, int parent) {
    for (int child : adj_list[node]) {
        if (child == parent) continue;
        int max_dist_subtree = dfs1(child, node);
        if (max_dist_subtree > max_dist_subtree1[node]) {
            max_dist_subtree2[node] = max_dist_subtree1[node];
            child2[node] = child1[node];
            max_dist_subtree1[node] = max_dist_subtree;
            child1[node] = child;
        } else if (max_dist_subtree > max_dist_subtree2[node]) {
            max_dist_subtree2[node] = max_dist_subtree;
            child2[node] = child;
        }
    }
    return max_dist_subtree1[node] + 1;
}

void dfs2(int node, int parent) {
    parent_max_dist[node] = parent_max_dist[parent] + 1;
    if (parent != 0) {
        if (node != child1[parent]) {
            parent_max_dist[node] = max(parent_max_dist[node], max_dist_subtree1[parent] + 1);
        } else {
            parent_max_dist[node] = max(parent_max_dist[node], max_dist_subtree2[parent] + 1);
        }
    }
    for (int child : adj_list[node]) {
        if (child == parent) continue;
        dfs2(child, node);
    }
}

int main() {
	n = read();
    for (int i = 1; i < n; i++) {
        int u = read(), v = read();
        adj_list[u].push_back(v);
        adj_list[v].push_back(u);
    }
    dfs1(1, 0);
    memset(parent_max_dist, -1, sizeof(parent_max_dist));
    dfs2(1, 0);
    for (int i = 1; i <= n; i++) {
        freq[max(max_dist_subtree1[i], parent_max_dist[i])] += 1;
    }
	fill(ans, ans + n + 1, n);
    int suffix_freq = 0;
    for (int i = n; i >= 1; i--) {
        suffix_freq += freq[i];
        if (suffix_freq > 0) {
            ans[i] = n - suffix_freq + 1;
        }
    }
    for (int i = 1; i <= n; i++) {
        cout << ans[i] << " ";
    }
    cout << endl;
    return 0;
}
```

## E. There Should Be a Lot of Maximums

### Solution 1:

```py

```