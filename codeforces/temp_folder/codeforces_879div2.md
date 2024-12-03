# Codeforces Round 879 Div 2

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

## A. Unit Array

### Solution 1:  greedy

```py
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    sum_ = sum(arr)
    cnt_negatives = sum(1 for x in arr if x < 0)
    res = 0
    while cnt_negatives & 1 or sum_ < 0:
        res += 1
        sum_ += 2
        cnt_negatives -= 1
    print(res)
    
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## B. Maximum Strength

### Solution 1:  greedy + math

Find the first index at which the prefix of left and right disagree, then you can add 9 to rest digits for left and 0 to all rest digits for right. 

```py
def main():
    left, right = input().split()
    n = len(right)
    left = left.zfill(n)
    right = list(map(int, right))
    left = list(map(int, left))
    res = 0
    for i in range(n):
        if left[i] == right[i]: continue
        res = abs(left[i] - right[i]) + 9 * (n - i - 1)
        return print(res)
    print(res)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

### Solution 2:  digit dp

This is works but time limit exceeds

```py
import math
from itertools import product

def main():
    left, right = input().split()
    n = len(right)
    left = left.zfill(n)
    dp = [[[[[-math.inf] * 2 for _ in range(2)] for _ in range(2)] for _ in range(2)] for _ in range(n + 1)] # (i, left_lower, left_upper, right_lower, right_upper)
    # tight is 1
    dp[0][1][1][1][1] = 0
    for i in range(n):
        L, R = int(left[i]), int(right[i])            
        for left_lower, left_upper in product(range(2), repeat=2):
            for d1 in range(10):
                if left_lower and d1 < L: continue
                if left_upper and d1 > R: break
                for right_lower, right_upper in product(range(2), repeat=2):
                    for d2 in range(10):
                        if right_lower and d2 < L: continue
                        if right_upper and d2 > R: break
                        nleft_lower = left_lower and d1 == L
                        nleft_upper = left_upper and d1 == R
                        nright_lower = right_lower and d2 == L
                        nright_upper = right_upper and d2 == R
                        dp[i + 1][nleft_lower][nleft_upper][nright_lower][nright_upper] = max(dp[i + 1][nleft_lower][nleft_upper][nright_lower][nright_upper], dp[i][left_lower][left_upper][right_lower][right_upper] + abs(d1 - d2))
    res = 0
    for left_lower, left_upper, right_lower, right_upper in product(range(2), repeat = 4):
        res = max(res, dp[n][left_lower][left_upper][right_lower][right_upper])
    print(res)   

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

```cpp
const int inf = 1e15;

int32_t main() {
    int T = read();
    while (T--) {
        string left, right;
        cin >> left >> right;

        int n = right.length();
        left = string(n - left.length(), '0') + left;

        vector<vector<vector<vector<vector<int>>>>> dp(n + 1,
            vector<vector<vector<vector<int>>>>(2,
                vector<vector<vector<int>>>(2,
                    vector<vector<int>>(2,
                        vector<int>(2, -inf)
                    )
                )
            )
        );

        // (i, left_lower, left_upper, right_lower, right_upper)
        dp[0][1][1][1][1] = 0;

        for (int i = 0; i < n; i++) {
            int L = left[i] - '0';
            int R = right[i] - '0';

            for (int left_lower = 0; left_lower < 2; left_lower++) {
                for (int left_upper = 0; left_upper < 2; left_upper++) {
                    for (int d1 = 0; d1 < 10; d1++) {
                        if (left_lower && d1 < L) continue;
                        if (left_upper && d1 > R) break;

                        for (int right_lower = 0; right_lower < 2; right_lower++) {
                            for (int right_upper = 0; right_upper < 2; right_upper++) {
                                for (int d2 = 0; d2 < 10; d2++) {
                                    if (right_lower && d2 < L) continue;
                                    if (right_upper && d2 > R) break;

                                    int nleft_lower = left_lower && d1 == L;
                                    int nleft_upper = left_upper && d1 == R;
                                    int nright_lower = right_lower && d2 == L;
                                    int nright_upper = right_upper && d2 == R;

                                    dp[i + 1][nleft_lower][nleft_upper][nright_lower][nright_upper] =
                                        max(dp[i + 1][nleft_lower][nleft_upper][nright_lower][nright_upper],
                                            dp[i][left_lower][left_upper][right_lower][right_upper] + abs(d1 - d2)
                                        );
                                }
                            }
                        }
                    }
                }
            }
        }

        int res = 0;

        for (int left_lower = 0; left_lower < 2; left_lower++) {
            for (int left_upper = 0; left_upper < 2; left_upper++) {
                for (int right_lower = 0; right_lower < 2; right_lower++) {
                    for (int right_upper = 0; right_upper < 2; right_upper++) {
                        res = max(res, dp[n][left_lower][left_upper][right_lower][right_upper]);
                    }
                }
            }
        }

        cout << res << endl;
    }

    return 0L;
}
```

## C. Game with Reversing

### Solution 1:  string + math

```py
import math

def main():
    n = int(input())
    s, t = input(), input()
    res = math.inf
    hamming_distance = lambda s1, s2: sum(1 for x, y in zip(s1, s2) if x != y)
    m1 = hamming_distance(s, t)
    m2 = max(1, hamming_distance(s, reversed(t)))
    res = min(res, m1 + (m1 if m1 % 2 == 0 else m1 - 1))
    res = min(res, m2 + (m2 - 1 if m2 % 2 == 0 else m2))
    print(res)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## D. Survey in Class

### Solution 1:  min and max + interval overlapping + reduce to few cases

There are 4 types of relationships of segments to study for this problem. 
prefix intersection
suffix intersection
nested intersection
non-intersection

for each segment consider actually only 3 cases.

1. prefix intersection
But the problem can be reduced further because for the prefix intersection you just want the one that has smallest prefix intersection, so consider the minimum right endpoint for all segments.  This will also include case when non-intersection, but it also maximizes on the prefix intersection as well.  

2. suffix intersection
For suffix intersection you just need to consider the maximum left endpoint for all segments, this will also include non-intersection when the right endpoint of current segment is before the maximum left endpoint.  in that case it will just be the length of current segment. 

3. nested intersection
for nested intersection you need to consider the smallest segment in the input.  And just subtract it from all segments. 

```py
import math

def main():
    n, m = map(int, input().split())
    lefts, rights = [None] * n, [None] * n
    smallest_segment = math.inf
    for i in range(n):
        left, right = map(int, input().split())
        lefts[i] = left
        rights[i] = right
        smallest_segment = min(smallest_segment, right - left + 1)
    min_right = min(rights)
    max_left = max(lefts)
    res = 0
    for i in range(n):
        res = max(res, rights[i] - max(min_right, lefts[i] - 1), min(max_left, rights[i] + 1) - lefts[i], rights[i] - lefts[i] - smallest_segment + 1)
    print(2 * res)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## E. MEX of LCM

### Solution 1: 

```py

```

## F. Typewriter

### Solution 1: 

```py

```