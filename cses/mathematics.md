# Mathematics

## Notes

if the implementation is in python it will have this at the top of the python script for fast IO operations

```py
import os,sys
from io import BytesIO, IOBase
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

## Solutions

## Throwing Dice

### Solution 1:  linear reccurence relation + matrix exponentiation + linear algebra + matrix

f(n) = f(n - 1) + f(n - 2) + f(n - 3) + f(n - 4) + f(n - 5) + f(n - 6)

![image](images/throwing_dice.PNG)

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
    n = int(input())
    mod = 10**9+7
    matrix = [
        [1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0]
    ]
    base_matrix = [
        [1],
        [0],
        [0],
        [0],
        [0],
        [0]
    ]
    exponentiated_matrix = mat_pow(matrix, n, mod)
    result_matrix = mat_mul(exponentiated_matrix, base_matrix, mod)
    return result_matrix[0][0]

if __name__ == '__main__':
    print(main())
```

## Dice Probability

### Solution 1:  math + probability + and or logic of probability

```py
def main():
    n, a, b = map(int, input().split())
    dp = [0] * (b + 1)
    for i in range(1, min(7, b + 1)):
        dp[i] = 1 / 6
    for _ in range(n - 1):
        ndp = [0] * (b + 1)
        for i in range(b):
            for j in range(1, 7):
                if i + j > b: continue
                ndp[i + j] += dp[i] / 6
        dp = ndp
    res = f"{sum(dp[a : b + 1]):0.6f}"
    print(res)

if __name__ == '__main__':
    main()
```

## Binomial Coefficients

### Solution 1: python math module + math.comb

TLE here, but sometimes it can work. 

```py
import math

def main():
    n = int(input())
    mod = int(1e9) + 7
    for _ in range(n):
        a, b = map(int, input().split())
        res = math.comb(a, b) % mod
        print(res)

if __name__ == '__main__':
    main()
```

### Solution 2:  precompute factorial and inverse factorial + O(1) to compute binomial coefficient + O(m) time, where m is the max(n, k)

```py
def mod_inverse(num, mod):
    return pow(num, mod - 2, mod)

def main():
    n = int(input())
    m = int(1e6) + 1
    mod = int(1e9) + 7
    fact = [1]*(m + 1)
    for i in range(1, m + 1):
        fact[i] = (fact[i - 1] * i) % mod
    inv_fact = [1]*(m + 1)
    inv_fact[-1] = mod_inverse(fact[-1], mod)
    for i in range(m - 1, -1, -1):
        inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % mod
    for _ in range(n):
        a, b = map(int, input().split())
        res = fact[a] * inv_fact[b] * inv_fact[a - b] % mod
        print(res)

if __name__ == '__main__':
    main()
```

## 

### Solution 1:

```py

```

## 

### Solution 1:

```py

```

## 

### Solution 1:

```py

```

## 

### Solution 1:

```py

```

## 

### Solution 1:

```py

```


## 

### Solution 1:

```py

```

## 

### Solution 1:

```py

```

## 

### Solution 1:

```py

```

## 

### Solution 1:

```py

```

## Stick Game

### Solution 1:

```py

```

## Nim Game I

### Solution 1:  grundy numbers + independent sub games for each heap so take the nim sum + grundy number is equal to number of sticks in pile + O(n) time

```py
from functools import reduce
import operator

def main():
    n = int(input())
    heaps = list(map(int, input().split()))
    return 'first' if reduce(operator.xor, heaps) > 0 else 'second'

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        print(main())
```

## Nim Game II

### Solution 1:

```py

```

## Stair Game

### Solution 1:

```py
from functools import cache
sys.stdout = open('output.txt', 'w')


@cache
def main(n, stairs):
    if sum(stairs[1:]) == 0: return 0
    grundy_numbers = set()
    for i in range(1, n):
        if stairs[i] == 0: continue
        new_stairs = list(stairs)
        for _ in range(stairs[i]):
            new_stairs[i] -= 1
            new_stairs[i - 1] += 1
            grundy_numbers.add(main(n, tuple(new_stairs)))
    mex = 0
    while mex in grundy_numbers:
        mex += 1
    if mex != 0:
        print('stairs', stairs, 'mex =', mex)
    return mex

if __name__ == '__main__':
    n = 6
    stairs = [20, 20, 20, 20, 20, 20]
    main(n, tuple(stairs))
    sys.stdout.close()
```

## Grundy's Game

### Solution 1:  grundy numbers + memoization + mex function + independent subgames that are summed up + observe pattern that grundy only equal to 0 for n < 1300 + O(n^2) time complexity

```cpp
#include <bits/stdc++.h>
using namespace std;

int memo[1301];
int thres = 10000;

int grundy(int coins) {
	if (memo[coins] != -1) return memo[coins];
	if (coins <= 2) return 0;
	vector<int> grundy_numbers(thres, 0);
	for (int i = 1; i <= coins/2; i++) {
		if (i == coins - i) continue;
		grundy_numbers[grundy(i) ^ grundy(coins - i)] = 1;
	}
	for (int grundy_number = 0; grundy_number < thres; grundy_number++) {
		if (grundy_numbers[grundy_number] == 0) {
			return memo[coins] = grundy_number;
		}
	}
	return -1;
}

int main() {
	cin.tie(0)->sync_with_stdio(0);
	int n, t;
	cin >> t;
	memset(memo, -1, sizeof(memo));
	grundy(1300);
	while (t--) {
		cin >> n;
		if (n <= 1300) {
			cout << (memo[n] > 0 ? "first" : "second") << endl;
		} else {
			cout << "first" << endl;
		}
	}
	return 0;
}
```


## Another Game

### Solution 1: brute force + expontential time + sprague grundy theorem + use to observe the pattern to derive the simpler solution

```py
from functools import cache
import sys
sys.setrecursionlimit(1_000_000)
sys.stdout = open('output.txt', 'w')

@cache
def coin_game(n, heaps):
    if all(h == 0 for h in heaps):
        return 0
    grundy_numbers = set()
    for mask in range(1, 1 << n):
        nei_heaps = list(heaps)
        below_zero = False
        for i in range(n):
            if (mask>>i)&1:
                if nei_heaps[i] == 0:
                    below_zero = True
                    break
                nei_heaps[i] -= 1
        if not below_zero:
            grundy_numbers.add(coin_game(n, tuple(nei_heaps)))
    # mex function implementation
    grundy_number = 0
    while grundy_number in grundy_numbers:
        grundy_number += 1
    print(heaps, 'g =', grundy_number)
    return grundy_number

if __name__ == '__main__':
    n = 4
    heaps = tuple([8, 8, 8, 8])
    print(coin_game(n, heaps))
    sys.stdout.close
```

### Solution 2:  periodicity observed from using sprague grund's theorem + if all the numbers are even then the grundy number is 0 and a losing state for first player + first player only wins if at least one number is odd

```py
def main():
    n = int(input())
    heaps = list(map(int, input().split()))
    return 'first' if any(h&1 for h in heaps) else 'second'
    
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        print(main())
```
