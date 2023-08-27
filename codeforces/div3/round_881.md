# Codeforces Round 881 Div 3

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

## A. Sasha and Array Coloring

### Solution 1: 

```py
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    arr.sort()
    res = 0
    left, right = 0, n - 1
    while left < right:
        delta = arr[right] - arr[left]
        res += delta
        left += 1
        right -= 1
    print(res)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## B. Long Long

### Solution 1: 

```py
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    ops = 0
    pos = True
    for num in arr:
        if num < 0:
            ops += pos
            pos = False
        elif num > 0:
            pos = True
    res = sum(map(abs, arr))
    print(res, ops)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## C. Sum in Binary Tree

### Solution 1: 

```py
def main():
    n = int(input())
    res = 0
    while n > 0:
        res += n
        n //= 2
    print(res)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## D. Apple Tree

### Solution 1: 

```cpp
vector<vector<int>> adj_list;
vector<int> num_leaves;

int dfs(int node, int parent) {
    bool is_leaf = true;
    int cnt = 0;
    for (int child : adj_list[node]) {
        if (child == parent)
            continue;
        is_leaf = false;
        cnt += dfs(child, node);
    }
    if (is_leaf)
        cnt = 1;
    num_leaves[node] = cnt;
    return cnt;
}

int32_t main() {
    int T = read();
    
    while (T--) {
        int n = read();
        
        adj_list.clear();
        adj_list.resize(n + 1);
        
        for (int i = 0; i < n - 1; i++) {
            int u = read(), v = read();
            adj_list[u].push_back(v);
            adj_list[v].push_back(u);
        }
        
        int q = read();
        
        vector<pair<int, int>> queries(q);
        for (int i = 0; i < q; i++) {
            int x = read(), y = read();    
            queries[i] = make_pair(x, y);
        }
        
        num_leaves.clear();
        num_leaves.resize(n + 1);
        
        dfs(1, 0);
        
        for (const auto& query : queries) {
            int x = query.first;
            int y = query.second;
            
            int res = num_leaves[x] * num_leaves[y];
            cout << res << endl;
        }
    }
    
    return 0;
}
```

## E. Tracking Segments

### Solution 1: 

```py
import math
from itertools import accumulate

def main():
    n, m = map(int, input().split())
    arr = [0] * n
    queries = [math.inf] * n
    segments = [None] * m
    for i in range(m):
        left, right = map(int, input().split())
        segments[i] = (left, right)
    q = int(input())
    for i in range(1, q + 1):
        x = int(input()) - 1
        arr[x] += 1
        queries[x] = i
    psum = [0] + list(accumulate(arr))
    # print('psum', psum, 'queries', queries)
    # print('arr', arr)
    res = math.inf
    for left, right in segments:
        sum_ = psum[right] - psum[left - 1]
        # print('left', left, 'right', right, 'sum_', sum_)
    res = res if res < math.inf else -1

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## F2. Omsk Metro (hard version)

### Solution 1: 

```py

```