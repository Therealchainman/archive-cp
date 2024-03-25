# Rolling Hash


## Rolling hash to find pattern in string

This works for when you have only lowercase english letters, 
This is an implementation of a rolling hash to find pattern in 
string

This can lead to collision sometimes, so it is not 100% guaranteed to work

```py
def find_pattern_rolling_hash(string: str, pattern: str) -> int:
    p, MOD = 31, int(1e9)+7
    coefficient = lambda x: ord(x) - ord('a') + 1
    pat_hash = 0
    for ch in pattern:
        pat_hash = ((pat_hash*p)%MOD + coefficient(ch))%MOD
    POW = 1
    for _ in range(len(pattern)-1):
        POW = (POW*p)%MOD
    cur_hash = 0
    for i, ch in enumerate(string):
        cur_hash = ((cur_hash*p)%MOD + coefficient(ch))%MOD
        if i>=len(pattern)-1:
            if cur_hash == pat_hash: return i-len(pattern)+1
            cur_hash = (cur_hash - (POW*coefficient(string[i-len(pattern)+1]))%MOD + MOD)%MOD
    return -1
```

## Double hash

if rolling hash fails, you could get accepted with a double hash, here is an example of it used in a problem. 

```py
p, MOD1, MOD2 = 31, int(1e9) + 7, int(1e9) + 9
coefficient = lambda x: ord(x) - ord('a') + 1
shashes = {}
add = lambda h, mod, ch: ((h * p) % mod + coefficient(ch)) % mod
n = len(wordsContainer)
for i in reversed(range(n)):
    word = wordsContainer[i]
    hash1 = hash2 = 0
    if len(word) <= shashes.get((hash1, hash2), (0, math.inf))[1]:
        shashes[(hash1, hash2)] = (i, len(word))
    for ch in reversed(word):
        hash1 = add(hash1, MOD1, ch)
        hash2 = add(hash2, MOD2, ch)
        if len(word) <= shashes.get((hash1, hash2), (0, math.inf))[1]:
            shashes[(hash1, hash2)] = (i, len(word))
```

## Rolling Hash when you have -1 in array

Example of very similar rolling hash implementation but for an array containing [-1, 0, 1] elements.  So you encode the coefficient by adding 2,  cause you can't have a 0 I believe. 

```py
n, m = len(nums), len(pattern)
p, MOD = 31, int(1e9)+7
coefficient = lambda x: x + 2
pat_hash = 0
for v in pattern:
    pat_hash = (pat_hash * p + coefficient(v)) % MOD
diff = [0] * (n - 1)
for i in range(n - 1):
    if nums[i + 1] > nums[i]: diff[i] = 1
    elif nums[i + 1] < nums[i]: diff[i] = -1
POW = 1
for _ in range(m - 1):
    POW = (POW * p) % MOD
ans = cur_hash = 0
for i, v in enumerate(diff):
    cur_hash = (cur_hash * p + coefficient(v)) % MOD
    if i >= m - 1:
        if cur_hash == pat_hash: ans += 1
        cur_hash = (cur_hash - coefficient(diff[i - m + 1]) * POW) % MOD
return ans
```

## Rolling Hash on Segment Tree

using segment tree with queries of [l, r)

```py
class SegmentTree:
    def __init__(self, n, neutral, func, initial_arr):
        self.func = func
        self.neutral = neutral
        self.size = 1
        self.n = n
        while self.size<n:
            self.size*=2
        self.nodes = [neutral for _ in range(self.size*2)] 
        self.build(initial_arr)

    def build(self, initial_arr):
        for i, segment_idx in enumerate(range(self.n)):
            segment_idx += self.size - 1
            val = initial_arr[i]
            self.nodes[segment_idx] = val
            self.ascend(segment_idx)

    def ascend(self, segment_idx):
        while segment_idx > 0:
            segment_idx -= 1
            segment_idx >>= 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.nodes[segment_idx] = self.func(self.nodes[left_segment_idx], self.nodes[right_segment_idx])
        
    def update(self, segment_idx, val):
        segment_idx += self.size - 1
        self.nodes[segment_idx] = val
        self.ascend(segment_idx)
            
    def query(self, left, right):
        stack = [(0, self.size, 0)]
        result = self.neutral
        while stack:
            # BOUNDS FOR CURRENT INTERVAL and idx for tree
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # COMPLETE OVERLAP
            if segment_left_bound >= left and segment_right_bound <= right:
                result = self.func(result, self.nodes[segment_idx])
                continue
            # PARTIAL OVERLAP
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
        return result
    
    def __repr__(self):
        return f"nodes array: {self.nodes}, next array: {self.nodes}"

import random
import math    

mod = 2**61 - 1
base0 = 3

while True:
    k = random.randint(1, mod - 1)
    base = pow(base0, k, mod)
    if base <= ord("z"): continue
    if math.gcd(base, mod - 1) != 1: continue
    break

def main():
    N, Q = map(int, input().split())
    S = input()
    pw = [1] * (N + 1)
    for i in range(N):
        pw[i + 1] = (pw[i] * base) % mod
    segfunc = lambda a, b: ((a[0] + (b[0] * pw[a[1]]) % mod) % mod, a[1] + b[1])
    seg = SegmentTree(N, (0, 0), segfunc, [(ord(ch), 1) for ch in S])
    seg_rev = SegmentTree(N, (0, 0), segfunc, [(ord(ch), 1) for ch in reversed(S)])
    for _ in range(Q):
        t, l, r = input().split()
        if t == "1":
            x, c = int(l) - 1, ord(r)
            seg.update(x, (c, 1))
            seg_rev.update(N - x - 1, (c, 1))
        else:
            l, r = int(l) - 1, int(r) - 1
            h = seg.query(l, r + 1)
            l_rev, r_rev = N - r - 1, N - l - 1
            h_rev = seg_rev.query(l_rev, r_rev + 1)
            if h == h_rev: print("Yes")
            else: print("No")
```

```cpp

```