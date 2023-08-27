# Codeforces Round 878 Div 3

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

## A. Cipher Shifer

### Solution 1:

```py
def main():
    n = int(input())
    s = input()
    res = []
    flag = False
    for ch in s:
        if not flag:
            res.append(ch)
            flag = True
        elif flag and ch == res[-1]:
            flag = False
    print(''.join(res))

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## E. Character Blocking

### Solution 1:

At time 0 store an array that indicates if column contains equal characters between string s1, and s2, and have a variable that tracks how many indices that they have differing characters. 

when a swap query takes place, you just need to update two locations in the array and the strings, So you need to check if characters were equal, and are not equal now increase 

ch1 = ch2, but ch1 is being swapped away, increment the number of differing characters
for character swapping in you need to decide what to do increment or decrement depending if character is equal or not equal. 

blocks of differing characters will decrement the count, and when blocking frees up it will increase the count, so when count is 0 you good, 
also need to add these blocks to a queue so that can remove from queue once time equals first element in queue.  

```py
from collections import deque

def main():
    strs = list(map(list, (input(), input())))
    t, q = map(int, input().split())
    blocked_queue = deque()
    diff_count = sum(1 for ch1, ch2 in zip(strs[0], strs[1]) if ch1 != ch2)
    for i in range(q):
        query = tuple(map(int, input().split()))
        if blocked_queue and blocked_queue[0][0] == i:
            _, idx = blocked_queue.popleft()
            diff_count += (strs[0][idx] != strs[1][idx])
        if query[0] == 1:
            pos = query[1] - 1
            diff_count -= (strs[0][pos] != strs[1][pos])
            blocked_queue.append((i + t, pos))
        elif query[0] == 2:
            st1, pos1, st2, pos2 = query[1:]
            st1 -= 1
            st2 -= 1
            pos1 -= 1
            pos2 -= 1
            if st1 == st2:
                diff_count -= (strs[st1][pos1] != strs[st1 ^ 1][pos1])
                diff_count -= (strs[st1][pos2] != strs[st1 ^ 1][pos2])
                strs[st1][pos1], strs[st2][pos2] = strs[st2][pos2], strs[st1][pos1]
                diff_count += (strs[st1][pos1] != strs[st1 ^ 1][pos1])
                diff_count += (strs[st1][pos2] != strs[st1 ^ 1][pos2])
            else:
                diff_count -= (strs[st1][pos1] != strs[st2][pos1])
                diff_count -= (strs[st1][pos2] != strs[st2][pos2])
                strs[st1][pos1], strs[st2][pos2] = strs[st2][pos2], strs[st1][pos1]
                diff_count += (strs[st1][pos1] != strs[st2][pos1])
                diff_count += (strs[st1][pos2] != strs[st2][pos2])
        else:
            if diff_count == 0:
                print("YES")
            else:
                print("NO")

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## D. Wooden Toy Festival

### Solution 1:

```py
def main():
    n = int(input())
    arr = sorted(list(map(int, input().split())))
    def possible(target):
        i = 0
        for _ in range(3):
            end = arr[i] + 2 * target
            while i < n and arr[i] <= end:
                i += 1
            if i == n: break
        return i == n
    left, right = 0, arr[-1]
    while left < right:
        mid = (left + right) >> 1
        if possible(mid):
            right = mid
        else:
            left = mid + 1
    return left

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## C. Ski Resort

### Solution 1:

```py
from itertools import groupby

def main():
    n, k, q = map(int, input().split())
    arr = list(map(int, input().split()))
    num_ways = lambda x: x * (x + 1) // 2
    res = 0
    for key, grp in groupby(arr, key = lambda x: x <= q):
        if key:
            x = len(list(grp)) - k + 1
            if x < 0: continue
            res += num_ways(x)
    print(res)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## F. Railguns

### Solution 1:

```py
def main():
    n, m = map(int, input().split())
    r = int(input())
    free = [[[True] * (r + 1) for _ in range(m + 1)] for _ in range(n + 1)]
    for _ in range(r):
        t, d, coord = map(int, input().split())
        if d == 1: # horizontal
            for i in range(m + 1):
                if 0 <= t - coord - i <= r:
                    free[coord][i][t - coord - i] = False
        else:
            for i in range(n + 1):
                if 0 <= t - coord - i <= r:
                    free[i][coord][t - coord - i] = False
    # dp(i, j, stops) = can reach this state
    dp = [[[0]*(r + 1) for _ in range(m + 1)] for _ in range(n + 1)]
    dp[0][0][0] = 1 # base case
    for i in range(n + 1):
        for j in range(m + 1):
            for k in range(r + 1):
                if not free[i][j][k]: continue
                # 3 MOVEMENTS
                if i > 0: 
                    dp[i][j][k] |= dp[i - 1][j][k]
                if j > 0:
                    dp[i][j][k] |= dp[i][j - 1][k]
                dp[i][j][k] |= dp[i][j][k - 1]
    for i in range(r + 1):
        if dp[n][m][i]: return print(n + m + i)
    print(-1)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## B. Binary Cafe

### Solution 1:  math + observation

You can form any groups as long as it doesn't sum above n, or you can just take all the possiblities which are 2^k, so just take the minimum, which one is limiting the number of ways. 

```py
def main():
    n, k = map(int, input().split())
    k = min(k, 30)
    res = min(n + 1, (1 << k))
    print(res)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```