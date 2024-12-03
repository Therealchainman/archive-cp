# Codeforces Round 886 Div 4

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

## A. To My Critics

### Solution 1:  min

```py
def main():
    a, b, c = map(int, input().split())
    res = "YES" if a + b + c - min(a, b, c) >= 10 else "NO"
    print(res)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## B. Ten Words of Wisdom

### Solution 1:  max

```py
def main():
    n = int(input())
    res = best = 0
    for i in range(1, n + 1):
        a, b = map(int, input().split())
        if a <= 10 and b > best:
            best = b
            res = i
    print(res)


if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## C. Word on the Paper

### Solution 1:  grid

```py
def main():
    n = 8
    grid = [input() for _ in range(n)]
    for c in range(n):
        word = ""
        for r in range(n):
            if grid[r][c] == '.':
                continue
            word += grid[r][c]
        if word: return print(word)
    return -1

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## D. Balanced Round

### Solution 1:  sliding window

```py
ef main():
    n, k = map(int, input().split())
    arr = sorted(map(int, input().split()))
    res = left = 0
    for right in range(n):
        if right > left:
            if abs(arr[right] - arr[right - 1]) > k:
                left = right
        res = max(res, right - left + 1)
    print(n - res)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## E. Cardboard for Pictures

### Solution 1:  binary search + math + quadratic formula

```py
def main():
    n, c = map(int, input().split())
    arr = list(map(int, input().split()))
    def possible(target):
        cardboard = 0
        for num in arr:
            cardboard += (num + 2 * target) * (num + 2 * target)
            if cardboard > c: return False
        return True
    left, right = 0, 10**18
    while left < right:
        mid = (left + right + 1) >> 1
        if possible(mid):
            left = mid
        else:
            right = mid - 1
    print(left)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## F. We Were Both Children

### Solution 1:  sieve + harmonic series + counter

```py
"""
The time complexity of this is nlogn because it is the harmonic series 
The summation of the 1/1 + 1/2 + 1/3 +... + 1/n is logn, and this is what happens in the inner loop
The inner loop is executed n/i times for each i so that is 
n/1+n/2 + n/n times, which is nlogn, it is n times harmonic series.
"""

def main():
    n = int(input())
    arr = list(filter(lambda x: x <= n, map(int, input().split())))
    freq = [0] * (n + 1)
    for num in arr:
        freq[num] += 1
    counts = [0] * (n + 1)
    for i in range(1, n + 1):
        # runs n/i times
        for j in range(i, n + 1, i):
            counts[j] += freq[i]
    print(max(counts), flush = True)


if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## G. The Morning Star

### Solution 1:  math + counters

just need to find how many points are along each horizontal line x = c, vertical line y = c, bisector y - x = c, and anti-bisector y + x = c

```cpp

int32_t main() {
    int t = read();
    while (t--) {
        int n = read();
        
        std::map<int, int> horizontal, vertical, bisector, anti_bisector;
        
        for (int i = 0; i < n; ++i) {
            int x = read(), y = read();
            horizontal[x]++;
            vertical[y]++;
            bisector[y - x]++;
            anti_bisector[y + x]++;
        }
        
        int res = 0;
        for (const auto& entry : horizontal) {
            int v = entry.second;
            if (v > 1) res += v * (v - 1);
        }
        for (const auto& entry : vertical) {
            int v = entry.second;
            if (v > 1) res += v * (v - 1);
        }
        for (const auto& entry : bisector) {
            int v = entry.second;
            if (v > 1) res += v * (v - 1);
        }
        for (const auto& entry : anti_bisector) {
            int v = entry.second;
            if (v > 1) res += v * (v - 1);
        }
        
        std::cout << res << std::endl;
    }
    return 0;
}
```

## H. The Third Letter

### Solution 1:  weighted bidirectional graph + bfs 

Can arbitrarily assign the first element starting a search with to be at the 0 location.  Then assign the rest based on the edge weight.  If there is ever a contradiction that is problematic.  Note there will be multiple connected components in the graph,  and each one can arbitrarily assign one of the nodes position to be 0.

```py
from collections import deque

def main():
    n, m = map(int, input().split())
    adj_list = [[] for _ in range(n + 1)]
    for _ in range(m):
        u, v, w = map(int, input().split())
        adj_list[u].append((v, w))
        adj_list[v].append((u, -w))
    vis = [0] * (n + 1)
    pos = [-1] * (n + 1)
    def bfs(node):
        queue = deque([node])
        vis[node] = 1
        pos[node] = 0
        while queue:
            node = queue.popleft()
            for nei, wei in adj_list[node]:
                npos = pos[node] + wei
                if vis[nei] and pos[nei] != npos: 
                    return False
                if vis[nei]: continue
                vis[nei] = 1
                pos[nei] = npos
                queue.append(nei)
        return True
    for i in range(1, n + 1):
        if vis[i]: continue
        if not bfs(i): return print('NO')
    print("YES")

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```