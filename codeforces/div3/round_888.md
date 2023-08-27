# Codeforces Round 888 Div 3

## Notes

if the implementation is in python it will have this at the top of the python script for fast IO operations

```py
import os,sys
from io import BytesIO, IOBase
from typing import *
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
#define int long long

inline int read() {
	int x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}
```

## A. Escalator Conversations

### Solution 1: modulus + math

```py
def main():
    n, m, k, H = map(int, input().split())
    heights = list(map(int, input().split()))
    res = 0
    for h in map(lambda h: abs(H - h), heights):
        if h % k or (m - 1) * k < h or h == 0: continue
        res += 1
    print(res)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## B. Parity Sort

### Solution 1: sort

```py
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    odd, even = [], []
    for num in arr:
        if num & 1:
            odd.append(num)
        else:
            even.append(num)
    odd.sort()
    even.sort()
    o = e = 0
    for i in range(n):
        if arr[i] & 1:
            arr[i] = odd[o]
            o += 1
        else:
            arr[i] = even[e]
            e += 1
    for i in range(1, n):
        if arr[i] < arr[i - 1]:
            return print('NO')
    print('YES')

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## C. Tiles Comeback

### Solution 1:  greedy

```py
import math

def main():
    n, k = map(int, input().split())
    colors = list(map(int, input().split()))
    if k == 1: return print('YES')
    color = colors[0]
    cnt = 1
    left = math.inf
    for i in range(1, n):
        cnt += color == colors[i]
        if cnt == k: 
            left = i
            break
    if left == math.inf: return print('NO')
    if colors[0] == colors[-1]: return print('YES')
    right = math.inf
    color = colors[-1]
    cnt = 1
    for i in range(n - 2, -1, -1):
        cnt += color == colors[i]
        if cnt == k: 
            right = i
            break
    if right == math.inf: return print('NO')
    res = 'YES' if left < right else 'NO'
    print(res)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## D. Prefix Permutation Sums

### Solution 1:  sort + difference array + deque

```py
from collections import deque
import math

"""
sx = math.inf means it has not been used
sx = -math.inf means it has already been used 
"""

def main():
    n = int(input())
    psum = [0] + list(map(int, input().split()))
    s1 = s2 = math.inf
    diff = [0] * (n - 1)
    for i in range(1, n):
        diff[i - 1] = psum[i] - psum[i - 1]
    diff.sort()
    queue = deque(range(1, n + 1))
    for dif in diff:
        while queue and queue[0] < dif: 
            num = queue.popleft()
            if s1 == math.inf:
                s1 = num
            elif s2 == math.inf:
                s2 = num
            else:
                return print('NO')
        if queue and queue[0] == dif:
            queue.popleft()
        else:
            if s1 + s2 != dif: return print('NO')
            s1 = s2 = -math.inf
    print('YES')

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## E. Nastya and Potions

### Solution 1:  dynamic programming + topological sort + directed graph

```py
"""
dp on topological sort of directed graph
"""

import math
from collections import deque

def main():
    n, k = map(int, input().split())
    costs = [0] + list(map(int, input().split()))
    potions = set(map(int, input().split()))
    # if you have the potion is is free to obtain
    for p in potions:
        costs[p] = 0
    dp = [0] * (n + 1)
    adj_list = [[] for _ in range(n + 1)]
    indegree = [0] * (n + 1)
    for v in range(1, n + 1):
        neighbors = list(map(int, input().split()))
        for u in neighbors[1:]:
            adj_list[u].append(v)
            indegree[v] += 1
    queue = deque()
    for i in range(1, n + 1):
        if indegree[i] == 0:
            dp[i] = costs[i]
            queue.append(i)
    while queue:
        node = queue.popleft()
        dp[node] = min(dp[node], costs[node])
        for nei in adj_list[node]:
            dp[nei] += dp[node]
            indegree[nei] -= 1
            if indegree[nei] == 0: queue.append(nei)
    for i in range(1, n + 1):
        dp[i] = min(dp[i], costs[i])
    print(*dp[1:])

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## F. Lisa and the Martians

### Solution 1:  bit manipulation + trie

This solution gives correct answer but it gets MLE error

```py
"""
consider case of two opposites existing as separate case
"""

from collections import defaultdict
import math

class TrieNode(defaultdict):
    def __init__(self):
        super().__init__(TrieNode)
        self.count = 0 # how many integers have this bit

    def __repr__(self) -> str:
        return f'count: {self.count}, children: {self.keys()}'

def main():
    n, k = map(int, input().split())
    arr = list(map(int, input().split()))
    val_index = {}
    trie = TrieNode()
    for i, num in enumerate(arr):
        val_index[num] = i
        node = trie
        for i in range(k - 1, -1, -1):
            node = node[(num >> i) & 1]
            node.count += 1
    max_i = max_j = max_x = 0
    max_val = -math.inf
    for i in range(n):
        bound = True
        node = trie
        x = j_val = 0
        for j in range(k - 1, -1, -1):
            bit_i = (arr[i] >> j) & 1
            if node[bit_i].count > 1 :
                node = node[bit_i]
                j_val |= bit_i * (1 << j)
            elif not bound and node[bit_i].count > 0:
                node = node[bit_i]
                j_val |= bit_i * (1 << j)
            else:
                node = node[bit_i ^ 1]
                j_val |= (bit_i ^ 1) * (1 << j)
                bound = False
        for j in range(k):
            bit_i = (arr[i] >> j) & 1
            bit_j = (j_val >> j) & 1
            if bit_i == bit_j == 0:
                x |= 1 << j
        val = (arr[i] ^ x) & (j_val ^ x)
        if val > max_val:
            max_val = val
            max_x = x
            max_i = i + 1
            max_j = val_index[j_val] + 1
        # to find index need a map of value to index
    print(max_i, max_j, max_x)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## G. Vlad and the Mountains

### Solution 1:  

```py

```