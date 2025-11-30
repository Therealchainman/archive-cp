# Atcoder Beginner Contest 206

## Notes:

if the implementation is in python it will have this at the top of the python script for fast IO operations

```py
import os,sys
from io import BytesIO, IOBase
sys.setrecursionlimit(1_000_000)

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

## F. Interval Game 2 

### Solution 1:  Sprague Grundy Theorem + recursive dp + nimbers + independent sub games + O(mn^2), n = range, m = number of segments

```py
from functools import lru_cache
from itertools import dropwhile

def main():
    n = int(input())
    segments = [None]*n
    for i in range(n):
        left, right = map(int, input().split())
        segments[i] = (left, right)
    
    @lru_cache(None)
    def dp(start: int, end: int) -> int:
        if start >= end: return 0
        nimbers = [False]*101
        for left_seg, right_seg in segments:
            if start <= left_seg and right_seg <= end:
                nimbers[dp(start, left_seg) ^ dp(right_seg, end)] = True
        return next(dropwhile(lambda i: nimbers[i], range(len(nimbers))))
    outcome = dp(1, 100)
    if outcome:
        return 'Alice'
    return 'Bob'

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        print(main())
```

# Atcoder Beginner Contest 245

## Summary

This was an interesting contest, very rough one for me.  

I kind of got stuck on the polynomial division problem which required just little algebra 
to solve. Then there is the problem E, which is an interesting one that requires some 
skills as well.  




# Atcoder Beginner Contest 249

## Summary

index trio was a silly problem that I couldn't figure out either.  I imagine it may be a form of prime sieve algorithm because then I can find the possible divisibles for each integer
it will just be the primes, but of course I don't know that will work cause 

I need to understand run length encoding more.  Of course this problem is asking more about when does run length encoding result in a shorter string and I suppose lossless data compression? 
So you need some characters to repeat a certain number of times to reduce.  My observations from this were the following. 

Suppose you have x characters adjacent that are the same character
xxx => x3
xxxx => x4
xxxxx => x5
for length of 1 you are saving -1 character
for length of 2 you are saving 0 character
for length of 3 you are saving 1 character
for length of 4 you are saving 2 characters
for length of 5 you are saving 3 characters
Thus we can get the following mathematical equation

space_saved = length_of_identical_adjacent - 2
thus space is only saved for at least length of 3 or greater characters

Now to be honest with this equation for run length encoding, just a simple linear equation, it probably is not difficult to derive the number of ways to have efficient run length encoding. 

## Jogging

### Solution 1:  Simulation

```py
DRAW = "Draw"
TAK = "Takahashi"
AO = "Aoki"
 
def main():
    A, B, C, D, E, F, X = map(int, input().split())
    tak_dist = get_distance(A,B,C, X)
    ao_dist = get_distance(D,E,F, X)
    if tak_dist > ao_dist:
        return TAK
    if ao_dist > tak_dist:
        return AO
    return DRAW
 
def get_distance(walk, speed, rest, time):
    distance = 0
    while time > 0:
        travel = min(time, walk)
        distance += speed*travel
        time = time - travel - rest
    return distance
 
if __name__ == '__main__':
    print(main())
```

## Perfect String

### Solution 1: hash table + bitwise for boolean

```py
def main():
    S = input()
    seen = set()
    has_upper = has_lower = False
    for ch in S:
        if ch in seen: return False
        seen.add(ch)
        has_upper |= ch.isupper()
        has_lower |= ch.islower()
    return has_upper and has_lower
 
if __name__ == '__main__':
    if main():
        print("Yes")
    else:
        print("No")
```

## Just K

### Solution 1: Bit Masking 

```py
def main():
    N, K = map(int, input().split())
    S = [input() for _ in range(N)]
    best = 0
    for i in range(1, 1<<N):
        freq = [0]*26
        for j in range(N):
            if (i>>j)&1:
                for x in map(lambda x: ord(x)-ord('a'), S[j]):
                    freq[x] += 1
        best = max(best, sum(x==K for x in freq))
    return best
 
 
if __name__ == '__main__':
    print(main())
```

## Index Trio

### Solution 1:  Get all factors + simple combinatoric of number of ways when you have 2 distinct categories to choose from

O(nsqrt(n))

```py
from collections import Counter
from math import sqrt
def main():
    N = int(input())
    A = list(map(int,input().split()))
    counter = Counter(A)
    ways = 0
    for Ai in A:
        div_arr = divisors(Ai)
        for Aj in div_arr:
            Ak = Ai//Aj
            ways += counter.get(Ak, 0) * counter.get(Aj, 0)
    return ways

def divisors(num):
    div_arr = []
    for i in range(1, int(sqrt(num))+1):
        if num%i==0:
            div_arr.append(i)
            div_arr.append(num//i)
    return list(set(div_arr))
 
 
if __name__ == '__main__':
    print(main())
```

## RLE

### Solution 1: Dynamic Programming 

this is a hard one for me

```py

```

## Ignore Operations

### Solution 1: difference array 

I was wrong it looks greedy with heap being used, still don't really understand it. 

```py

```

# Atcoder Beginner Contest 278

## Shift

### Solution 1:  deque + simulation

```py
from collections import deque
def main():
    n, k = map(int, input().split())
    queue = deque(map(int, input().split()))
    for _ in range(k):
        queue.popleft()
        queue.append(0)
    return ' '.join(map(str, queue))

if __name__ == '__main__':
    print(main())
```

## Misjudge the Time

### Solution 1:  math + string + modular arithmetic

```py
def is_valid(h, m):
    return h >= 0 and h <= 23 and m >= 0 and m <= 59

def is_confusing(minutes):
    h = minutes // 60
    m = minutes % 60
    last_digit = str(h%10)
    h_str = str(h).zfill(2)
    m_str = str(m).zfill(2)
    first_digit = m_str[0]
    h_swap = int(h_str[0] + first_digit)
    m_swap = int(last_digit + m_str[1])
    return is_valid(h, m) and is_valid(h_swap, m_swap)

def main():
    h, m = map(int, input().split())
    minutes = h * 60 + m
    while not is_confusing(minutes):
        minutes = (minutes + 1)%1440
    h = minutes // 60
    m = minutes % 60
    return f'{h} {m}'

if __name__ == '__main__':
    print(main())
```

## FF

### Solution 1:  set

```py
def main():
    n, q = map(int, input().split())
    following = set()
    result = []
    for _ in range(q):
        t, a, b = map(int, input().split())
        if t == 1:
            following.add((a, b))
        elif t == 2:
            following.discard((a, b))
        else:
            if (a, b) in following and (b, a) in following:
                result.append('Yes')
            else:
                result.append('No')
    return '\n'.join(result)

if __name__ == '__main__':
    print(main())
```

## All Assign Point Add

### Solution 1: dictionary + greedy

```py
from math import inf
def main():
    n = int(input())
    arr = map(int, input().split())
    q = int(input())
    result = []
    updated = {i: x for i, x in enumerate(arr)}
    assigned = 0
    for _ in range(q):
        query = list(map(int, input().split()))
        if query[0] == 1:
            assigned = query[1]
            updated.clear()
        elif query[0] == 2:
            i, x = query[1:]
            i -= 1
            if i in updated:
                updated[i] += x
            else:
                updated[i] = assigned + x
        elif query[0] == 3:
            i = query[1] - 1
            if i in updated:
                result.append(updated[i])
            else:
                result.append(assigned)
    return '\n'.join(map(str, result))

if __name__ == '__main__':
    print(main())
```

## Grid Filling

### Solution 1:  dictionary + find overlap of integers

```py
from itertools import product
def main():
    H, W, N, h, w = map(int, input().split())
    grid = [[0] * W for _ in range(H)]
    for i in range(H):
        row = map(int, input().split())
        for j, x in enumerate(row):
            grid[i][j] = x
    bounding_boxes = {} # (min(i), min(j), max(i), max(j))
    for i, j in product(range(H), range(W)):
        n = grid[i][j]
        if n in bounding_boxes:
            min_i, min_j, max_i, max_j = bounding_boxes[n]
            bounding_boxes[n] = (min(min_i, i), min(min_j, j), max(max_i, i), max(max_j, j))
        else:
            bounding_boxes[n] = (i, j, i, j)
    total_distinct = len(bounding_boxes)
    ans = [[total_distinct]*(W-w+1) for _ in range(H-h+1)]
    for i, j in product(range(H-h+1), range(W-w+1)):
        for n, (min_i, min_j, max_i, max_j) in bounding_boxes.items():
            if min_i >= i and min_j >= j and max_i < i+h and max_j < j+w:
                ans[i][j] -= 1
    for row in ans:
        print(' '.join(map(str, row)))

if __name__ == '__main__':
    main()
```

# Atcoder Beginner Contest 285

## A. Edge Checker 2

### Solution 1:  tree structured in a way where the parent is just the floor division by 2

```py
def main():
    a, b = map(int, input().split())
    return 'Yes' if b//2 == a else 'No'

if __name__ == '__main__':
    print(main())
```

## B. Longest Uncommon Prefix

### Solution 1: brute force + O(n^2)

```py
def main():
    n = int(input())
    s = input()
    result = [None]*(n - 1)
    for i in range(1, n):
        l = n - i
        for j in range(n - i):
            if s[j] == s[j + i]:
                l = j
                break
        result[i - 1] = l
    return '\n'.join(map(str, result))

if __name__ == '__main__':
    print(main())
```

## C. abc285_brutmhyhiizp

### Solution 1:

```py

```

## D. Change Usernames

### Solution 1:  detect cycle in directed graph with dfs 

```py
import sys
sys.setrecursionlimit(1000000)
"""
a graph where each node has at most one outgoing edge
"""
def main():
    n = int(input())
    adj_list = {}
    for _ in range(n):
        s, t = input().split()
        adj_list[s] = t
    visited = set()
    in_path = set()
    def detect_cycle(node: str) -> bool:
        visited.add(node)
        in_path.add(node)
        nei = adj_list.get(node, None)
        if nei is not None:
            if nei in in_path:
                return True
            if nei not in visited and detect_cycle(nei):
                return True
        in_path.remove(node)
        return False
    for node in adj_list:
        if node not in visited and detect_cycle(node):
            return "No"
    return "Yes"

if __name__ == '__main__':
    print(main())
```

## 

### Solution 1:

```py

```

## F. Substring of Sorted String

### Solution 1:  segment tree + tle

```py
from typing import List
import os,sys
from io import BytesIO, IOBase

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

class SegmentTree:
    def __init__(self, n: int, neutral: int, initial_arr: List[int]):
        self.neutral = neutral
        self.size = 1
        self.n = n
        while self.size<n:
            self.size*=2
        self.nodes = [self.neutral for _ in range(self.size*2)] 
        self.build(initial_arr)

    def build(self, initial_arr: List[int]) -> None:
        for i, segment_idx in enumerate(range(self.n)):
            segment_idx += self.size - 1
            self.nodes[segment_idx] = initial_arr[i]
            self.ascend(segment_idx)
    
    def calc_op(self, left: int, right: int) -> int:
        return left + right

    def ascend(self, segment_idx: int) -> None:
        while segment_idx > 0:
            segment_idx -= 1
            segment_idx >>= 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.nodes[segment_idx] = self.calc_op(self.nodes[left_segment_idx], self.nodes[right_segment_idx])
        
    def update(self, segment_idx: int, val: int) -> None:
        segment_idx += self.size - 1
        self.nodes[segment_idx] = val
        self.ascend(segment_idx)
            
    def query(self, left: int, right: int) -> int:
        stack = [(0, self.size, 0)]
        result = self.neutral
        while stack:
            # BOUNDS FOR CURRENT INTERVAL and idx for tree
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # COMPLETE OVERLAP
            if segment_left_bound >= left and segment_right_bound <= right:
                result = self.calc_op(result, self.nodes[segment_idx])
                continue
            # PARTIAL OVERLAP
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
        return result
    
    def __repr__(self) -> str:
        return f"nodes array: {self.nodes}"
"""

"""
def main():
    n = int(input())
    s = list(input())
    q = int(input())
    unicode = lambda ch: ord(ch) - ord('a')
    # SEGMENT TREE FOR EACH OF THE 26 CHARACTERS REPRESENTED AS INTEGER [0, 25]
    count_seg_trees = [None]*26
    freq_sorted_s = [s.count(chr(i+ord('a'))) for i in range(26)]
    for i in range(26):
        freq_temp = [0]*n
        for j in range(n):
            if unicode(s[j]) == i:
                freq_temp[j] = 1
        count_seg_trees[i] = SegmentTree(n, 0, freq_temp)
    result = []
    """
    checks if sorted based on the frequency of characters in the substring,
    because when you query the segment tree, each query will return frequency of the smallest character first
    for left index to the frequency of this character, now all of the characters should be in that range, 
    because it is supposed to be sorted, if it is not, that means it is not sorted, the other contributing character to the
    frequencey is somewhere else in the substring
    """
    def is_sorted(freq, left):
        for i in range(26):
            if count_seg_trees[i].query(left, left + freq[i]) != freq[i]: return False
            left += freq[i]
        return True
    for _ in range(q):
        query = input().split()
        if query[0] == '1':
            i, c = int(query[1]), query[2]
            i -= 1
            # UPDATE FREQUENCY OF SORTED STRING
            freq_sorted_s[unicode(s[i])] -= 1
            freq_sorted_s[unicode(c)] += 1
            # UPDATE SEGMENT TREE FOR STRING
            count_seg_trees[unicode(s[i])].update(i, 0)
            count_seg_trees[unicode(c)].update(i, 1)
            # UPDATE STRING FOR CHARACTER
            s[i] = c
        else:
            l, r = map(int, (query[1], query[2]))
            l -= 1
            counts = [0]*26
            min_i, max_i = 26, 0
            for i in range(26):
                counts[i] = count_seg_trees[i].query(l, r)
                if counts[i] > 0:
                    min_i = i if min_i == 26 else min_i
                    max_i = i
            check = True
            for i in range(min_i + 1, max_i):
                check &= freq_sorted_s[i] == counts[i]
            # CHECK IF THE SUBSTRING IS SORTED IN ASCENDING ORDER USING SEGMENT TREE QUERIES AND FREQUENCY ARRAY FOR CHARACTERS
            check &= is_sorted(counts, l)
            if check: result.append('Yes')
            else: result.append('No')
    return '\n'.join(result)

if __name__ == '__main__':
    print(main())
```


### Solution 2: Fenwick Tree for frequency of characters in a range

```cpp## 

### Solution 1:

```py

```

## 

### Solution 1:

```py

```
#include <bits/stdc++.h>
using namespace std;

int neutral = 0;

struct FenwickTree {
    vector<int> nodes;
    
    void init(int n) {
        nodes.assign(n + 1, neutral);
    }

    void update(int idx, int val) {
        while (idx < (int)nodes.size()) {
            nodes[idx] += val;
            idx += (idx & -idx);
        }
    }

    int query(int left, int right) {
        return query(ri## 

### Solution 1:

```py

```

## 

### Solution 1:

```py

```ght) - query(left);
    }

    int query(int idx) {
        int result = neutral;
        while (idx > 0) {
            result += nodes[idx];
            idx -= (idx & -idx);
        }
        return result;
    }

};

vector<FenwickTree> trees(26, FenwickTree());
vector<int> freq(26, 0);

bool is_sorted(vector<int> &range_char_freq, int left, int right) {
    for (int i = 0; i<26; i++) {
        if (left == right) break;
        if (range_char_freq[i] == 0) continue;
        if (trees[i].query(left, left + range_char_freq[i]) != range_char_freq[i]) return false;
        left += range_char_freq[i];
    }
    return true;
}

bool check(int min_i, int max_i, vector<int> &range_char_freq) {
    for (int i = min_i + 1; i < max_i; i++) {
        if (range_char_freq[i] != freq[i]) return false;
    }
    return true;
}

int main() {
    int n, q, idx, left, right, query_type;
    string s;
    char ch;
    cin>>n>>s>>q;

    for (int i = 0; i < 26; i++) {
        trees[i].init(n);
    }

    for (int i = 0; i < n; i++) {
        trees[s[i] - 'a'].update(i + 1, 1);
        freq[s[i] - 'a']++;
    }
    
    while (q--) {
        cin>>query_type;
        if (query_type == 1) {
            cin>>idx>>ch;
            idx--;
            int old_char = s[idx] - 'a';
            int new_char = ch - 'a';
            trees[old_char].update(idx + 1, -1);
            trees[new_char].update(idx + 1, 1);
            s[idx] = ch;
            freq[old_char]--;
            freq[new_char]++;
        } else {
            cin>>left>>right;
            left--;
            bool ans = true;
            int min_i = 0, max_i = 25;
            vector<int> range_char_freq(26, 0);
            for (int i = 0; i < 26; i++) {
                if (freq[i] == 0) continue;
                int curr = trees[i].query(left, right);
                range_char_freq[i] = curr;
            }
            while (range_char_freq[min_i] == 0) min_i++;
            while (range_char_freq[max_i] == 0) max_i--;
            if (!check(min_i, max_i, range_char_freq)) {
                cout << "No" << endl;
                continue;
            }

            if (is_sorted(range_char_freq, left, right)) {
                cout << "Yes" << endl;
            } else {
                cout << "No" << endl;
            }
        }
    }
    return 0;
}## 

### Solution 1:

```py

```

## 

### Solution 1:

```py

```
```

# Atcoder Beginner Contest 287

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

## A. Majority

### Solution 1:  counter

```py
def main():
    n = int(input())
    counts = Counter()
    for _ in range(n):
        s = input()
        counts[s] += 1
    if counts['For'] > counts['Against']:
        return 'Yes'
    else:
        return 'No'

if __name__ == '__main__':
    print(main())
```

## B. Postal Card

### Solution 1:  string + set

```py
def main():
    n, m = map(int, input().split())
    res = 0
    arr = [None]*n
    t_arr = set()
    for i in range(n):
        arr[i] = input()
    for i in range(m):
        t_arr.add(input())
    for s in arr:
        if s[-3:] in t_arr:
            res += 1
    return res

if __name__ == '__main__':
    print(main())
```

## C. Path Graph?

### Solution 1: cycle detection in undirected graph + dfs

```py
def main():
    n, m = map(int, input().split())
    visited = [False]*n
    adj_list = defaultdict(list)
    for i in range(m):
        u, v = map(int, input().split())
        adj_list[u-1].append(v-1)
        adj_list[v-1].append(u-1)
    def is_cycle(node, parent_node):
        visited[node] = True
        for nei in adj_list[node]:
            if not visited[nei]:
                if is_cycle(nei, node):
                    return True
            elif nei != parent_node:
                return True
        return False
    cycle = is_cycle(0, None)
    return 'Yes' if not cycle and sum(visited) == n else 'No' 

if __name__ == '__main__':
    print(main())
```

## D. Match or Not

### Solution 1:  prefix + suffix for boolean + bit manipulation + greedy

```py
def main():
    s = input()
    t = input()
    nt = len(t)
    ns = len(s)
    suffix_bool = [True]*(nt + 1)
    is_equal = lambda x, y: x == y or x == '?' or y == '?'
    for i in reversed(range(len(t))):
        suffix_bool[i] = (is_equal(t[i], s[ns - (nt - i)])) & suffix_bool[i + 1]
    result = [None]*(nt + 1)
    result[0] = 'Yes' if suffix_bool[0] else 'No'
    prefix_bool = True
    for i in range(nt):
        prefix_bool &= is_equal(t[i], s[i])
        res = suffix_bool[i + 1] & prefix_bool
        result[i + 1] = 'Yes' if res else 'No'
    return '\n'.join(result)

if __name__ == '__main__':
    print(main())
```

## E. Karuta

### Solution 1: max prefix in a trie data structure + add, remove, query operations for trie data structure

```py
class TrieNode(defaultdict):
    def __init__(self):
        super().__init__(TrieNode)
        self.prefix_count = 0 # how many words have this prefix

    def __repr__(self) -> str:
        return f'is_word: {self.is_word} prefix_count: {self.prefix_count}, children: {self.keys()}'

def main():
    n = int(input())
    words = [input() for _ in range(n)]
    root = TrieNode()
    for word in words:
        cur = root
        for ch in word:
            cur = cur[ch]
            cur.prefix_count += 1
    result = [0]*n
    for i, word in enumerate(words):
        # REMOVE CURRENT WORD SO DON'T MATCH WITH ITSELF
        cur = root
        for ch in word:
            cur = cur[ch]
            cur.prefix_count -= 1
        # FIND MAX PREFIX
        cur = root
        j = 0
        while j < len(word):
            ch = word[j]
            cur = cur[ch]
            if cur.prefix_count == 0:
                break
            j += 1
        result[i] = j
        # ADD CURRENT WORD BACK
        cur = root
        for ch in word:
            cur = cur[ch]
            cur.prefix_count += 1

    return '\n'.join(map(str, result))

if __name__ == '__main__':
    print(main())
```

### Solution 2:  sort the array of strings and look to left and right neighbor to find max prefix + query the max prefix length for each query + offline query + O(n) time

```py
def main():
    n = int(input())
    arr = sorted([(input(), i) for i in range(n)])
    ans = [0]*n
    for i in range(n):
        max_prefix = 0
        s, idx = arr[i]
        if i > 0:
            max_possible_prefix = min(len(s), len(arr[i-1][0]))
            for j in range(max_possible_prefix):
                if s[j] == arr[i-1][0][j]:
                    max_prefix = j+1
                else:
                    break
        if i < n-1:
            max_possible_prefix = min(len(s), len(arr[i+1][0]))
            for j in range(max_possible_prefix):
                if s[j] == arr[i+1][0][j]:
                    max_prefix = max(max_prefix, j+1)
                else:
                    break
        ans[idx] = max_prefix
    return '\n'.join(map(str, ans))

if __name__ == '__main__':
    print(main())
```

## F. Components

### Solution 1:  tree dp + queue + arbtirary root of tree + convolution for merging results from each child + O(n^2) possibly

Why does convolution work? 

Because if you have two arrays that represent number of ways for components for two children node, you can merge them this way

suppose you have two arrays
[1, 2, 3]
[2, 4]
now index of these arrays correspond to the number of components.

So consider this fact,
if you take x, y components
x + y is the number of components if you merge these two together
and the number of ways is arr1[x] * arr2[y] now.  

so
(0, 0) 0 components
(0, 1) 1 components
(1, 0) 1 components
(1, 1) 2 components
(2, 0) 2 components
(2, 1) 3 components

so the product of these ways should be summed together for x components to get total number of ways to get x components

```py
import math
from collections import deque
from itertools import zip_longest, product

mod = 998_244_353

def convolution(arr1: List[int], arr2: List[int]) -> List[int]:
    """
    Convolution of two sequences of numbers modulo mod
    """
    n = len(arr1)
    m = len(arr2)
    res = [0] * (n + m - 1)
    for i, j in product(range(n), range(m)):
        res[i + j] += arr1[i] * arr2[j]
        res[i + j] %= mod
    return res

def add_sequence(arr1: List[int], arr2: List[int]) -> List[int]:
    """
    Adds two sequences of numbers modulo mod
    """
    return [(a + b) % mod for a, b in zip_longest(arr1, arr2, fillvalue = 0)]

def merge(arrs: List[List[int]]) -> List[int]:
    """
    Merges a list of sequences of numbers modulo mod
    """
    res = [1]
    if len(arrs) == 0: return res # base case
    queue = deque(arrs)
    while len(queue) > 1:
        arr1 = queue.popleft()
        arr2 = queue.popleft()
        queue.append(convolution(arr1, arr2))
    return queue[0]

def main():
    n = int(input())
    adj_list = [[] for _ in range(n)]
    for _ in range(n - 1):
        u, v = map(int, input().split())
        u -= 1
        v -= 1
        adj_list[u].append(v)
        adj_list[v].append(u)
    children = [[] for _ in range(n)]
    dist = [math.inf]*n
    dist[0] = 0 # root tree at node 0
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for child in adj_list[node]:
            if dist[child] == math.inf:
                dist[child] = dist[node] + 1
                children[node].append(child)
                queue.append(child)
    # sorted in decreasing distance from root node 0, descending order based on distance
    vertex = sorted(range(n), key=lambda x: dist[x], reverse=True)
    # memo[i][0][j] corresponds to the number of ways to form j connected components with the subtree rooted at i when skipping node i
    # memo[i][1][j] corresponds to the number of ways to form j connected components with the subtree rooted at i when not skipping node i
    memo = [[[], []] for _ in range(n)] 
    for v in vertex:
        # merge all the children's values for number of ways to form up j connected components
        # shen skipping node v you can just add the number of ways 
        memo[v][0] = merge([add_sequence(memo[child][0], memo[child][1]) for child in children[v]])
        """
        example to understand this logic, given simple example

        n1
        |
        n2

        n1 has child node n2

        suppose
        component_ways_for_keeping_node_n2 =  [0, 10, 5, 5]
        component_ways_for_skipping_node_n2 = [1, 10, 5, 1]
                                   components  0, 1,  2, 3

        then what should be the transition state to find these for node n1, given we know for node n2?
        component_ways_for_keeping_node_n1 
        =
        [0, 10, 5, 5]
        [0, 1, 10, 5, 1]
        = [0, 11, 15, 10, 1]
        basically, if you skipped n2, and you are keeping n1, then you are incrementing number of components by 1, so you shift the entire array to the right by 1.
        So now what was for 1 component, is now for 2 components and added for when you keep n2

        component_ways_for_skipping_node_n1
        =
        [0, 10, 5, 5]
        [1, 10, 5, 1]
        = [1, 11, 10, 6]
        """ 
        memo[v][1] = [0] + merge([add_sequence(memo[child][0], memo[child][1][1:]) for child in children[v]])
    ans = [0]*(n + 1)
    # add number of ways for when including node and not including node 0
    for take in [0, 1]:
        # j corresponds to number of components
        # ways corresponds to number of ways to form j components
        for j, ways in enumerate(memo[0][take]):
            ans[j] += ways
            ans[j] %= mod
    print(*ans[1:], sep = '\n')

if __name__ == '__main__':
    main()
```

# Atcoder Beginner Contest 289

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

## A - flip

### Solution 1: str + xor + bit operations

```py
def main():
    s = input()
    return ''.join(map(str, [x^1 for x in map(int, s)]))
 
if __name__ == '__main__':
    print(main())
```

## B - V

### Solution 1:  dfs for connected components + undirected graph

```py
def main():
    n, m = map(int, input().split())
    arr = list(map(int, input().split()))
    adj_list = [[] for _ in range(n + 1)]
    visited = [0]*(n + 1)
    def bfs(node: int, adj_list: List[int]) -> List[int]:
        stack = [node]
        visited[node] = 1
        cc = []
        while stack:
            cur_node = stack.pop()
            cc.append(cur_node)
            for neighbor in adj_list[cur_node]:
                if visited[neighbor] == 0:
                    visited[neighbor] = 1
                    stack.append(neighbor)
        return cc
    for u in arr:
        v = u + 1
        adj_list[u].append(v)
        adj_list[v].append(u)
    result = []
    for node in range(1, n + 1):
        if visited[node]: continue
        connected_component = bfs(node, adj_list)
        result.extend(sorted(connected_component, reverse = True))
    return ' '.join(map(str, result))
 
if __name__ == '__main__':
    print(main())
```

## C - Coverage

### Solution 1:  bitmask + set operations + union of sets

```py
def main():
    n, m = map(int, input().split())
    s = [None]*m
    for i in range(m):
        _ = int(input())
        s[i] = set(map(int, input().split()))
    res = 0
    for mask in range(1, 1<<m):
        cur_set = set()
        for i in range(m):
            if (mask>>i)&1:
                cur_set |= s[i]
        res += len(cur_set) == n
    return res
 
if __name__ == '__main__':
    print(main())
```

## D - Step Up Robot

### Solution 1:  stack + iterative dfs + visited array

```py
def main():
    n = int(input())
    moves = list(map(int, input().split()))
    m = int(input())
    traps = set(map(int, input().split()))
    x = int(input())
    stack = [0]
    visited = [0]*(x + 1)
    visited[0] = 1
    while stack:
        node = stack.pop()
        for move in moves:
            new_node = node + move
            if new_node == x: return 'Yes'
            if new_node < x and not visited[new_node] and new_node not in traps:
                visited[new_node] = 1
                stack.append(new_node)
    return 'No'
 
if __name__ == '__main__':
    print(main())
```

## E - Swap Places

### Solution 1:  memoization for states + bfs + deque + shortest path

```py
import math
from collections import deque
 
def main():
    n, m = map(int, input().split())
    colors = [0] + list(map(int, input().split()))
    adj_list = [[] for _ in range(n + 1)]
    for _ in range(m):
        u, v = map(int, input().split())
        adj_list[u].append(v)
        adj_list[v].append(u)
    dist = [[math.inf]*(n + 1) for _ in range(n + 1)]
    dist[1][n] = 0
    queue = deque([(1, n, 0)])
    while queue:
        u, v, d = queue.popleft()
        if (u, v) == (n, 1): return d
        for nei_u in adj_list[u]:
            for nei_v in adj_list[v]:
                if colors[nei_u] == colors[nei_v]: continue
                if d + 1 < dist[nei_u][nei_v]:
                    dist[nei_u][nei_v] = d + 1
                    queue.append((nei_u, nei_v, d + 1))
    return -1
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        print(main())
```

## F - Teleporter Takahashi

### Solution 1:  a trivial way to change the current x, y based on two moves in the rectangle leads to solution + the cases that will guarantee that it cannot reach the answer + x, y needs to always have the same parity as tx, ty + you need x_min != x_max if you are going to be able to change that x to a target value.  + all moves will always be less than 1 million, think about it worst case is probably 400,000 moves to get to a target point.

```py
def main():
    sx, sy = map(int, input().split())
    tx, ty = map(int, input().split())
    x_min, x_max, y_min, y_max = map(int, input().split())
    # (rx, ry) is point in the the rectangle
    def move(rx: int, ry: int) -> Tuple[int, int]:
        path.append(f'{rx} {ry}')
        return 2*rx - x, 2*ry - y
    for i in range(2):
        x, y = sx, sy
        path = ['Yes']
        if i:
            x, y = move(x_min, y_min)
        if x%2 != tx%2 or y%2 != ty%2 or (x_min == x_max and x != tx) or (y_min == y_max and y != ty):
            continue
        while (x < tx): 
            x, y = move(x_min, y_min)
            x, y = move(x_min + 1, y_min)
        while (x > tx):
            x, y = move(x_min + 1, y_min)
            x, y = move(x_min, y_min)
        while (y < ty):
            x, y = move(x_min, y_min)
            x, y = move(x_min, y_min + 1)
        while (y > ty):
            x, y = move(x_min, y_min + 1)
            x, y = move(x_min, y_min)
        return '\n'.join(path)
    return 'No'

if __name__ == '__main__':
    print(main())
```

# Atcoder Beginner Contest 291

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

## A - camel Case 

### Solution 1:  loop

```py
def main():
    s = input()
    for i, ch in enumerate(s, start = 1):
        if ch == ch.upper():
            return i
    return -1
 
if __name__ == '__main__':
    print(main())
```

## B - Trimmed Mean 

### Solution 1:  sort + sum

```py
def main():
    n = int(input())
    arr = sorted(list(map(int, input().split())))
    mean = sum(arr[n:4*n]) / (3*n)
    return mean
 
if __name__ == '__main__':
    print(main())
```

## C - LRUD Instructions 2 

### Solution 1:  set + position in plane

```py
def main():
    n = int(input())
    s = input()
    vis = set([(0, 0)])
    x = y = 0
    for ch in s:
        if ch == 'L':
            x -= 1
        elif ch == 'R':
            x += 1
        elif ch == 'U':
            y += 1
        else: # 'D'
            y -= 1
        if (x, y) in vis: return 'Yes'
        vis.add((x, y))
    return 'No'
 
if __name__ == '__main__':
    print(main())
```

## D - Flip Cards 

### Solution 1:  iterative dp + store the number of ways to satisfy condition with the current card face up, front or back card + keep a count for both

```py
def main():
    n = int(input())
    front, back = [0]*n, [0]*n
    for i in range(n):
        front_card, back_card = map(int, input().split())
        front[i], back[i] = front_card, back_card
    mod = 998_244_353
    front_count, back_count = [0]*(n), [0]*n
    front_count[0] = back_count[0] = 1
    for i in range(1, n):
        if front[i] != front[i - 1]:
            front_count[i] += front_count[i - 1]
        if front[i] != back[i - 1]:
            front_count[i] += back_count[i - 1]
        if back[i] != front[i - 1]:
            back_count[i] += front_count[i - 1]
        if back[i] != back[i - 1]:
            back_count[i] += back_count[i - 1]
        front_count[i] %= mod
        back_count[i] %= mod
    return (front_count[-1] + back_count[-1])%mod
 
if __name__ == '__main__':
    print(main())
```

## E - Find Permutation 

### Solution 1:  topological sort + if at any point there are multiple neighbors that have indegree = 0 that would mean it does not have a unique solution + if can't reach the end

```py
from collections import deque
 
def main():
    n, m = map(int, input().split())
    adj_list = [[] for _ in range(n + 1)]
    indegrees = [0]*(n + 1)
    for _ in range(m):
        u, v = map(int, input().split())
        indegrees[v] += 1
        adj_list[u].append(v)
    queue = deque()
    for i in range(1, n + 1):
        if indegrees[i] == 0:
            queue.append(i)
    indices = []
    while queue:
        if len(queue) > 1: return 'No'
        idx = queue.popleft()
        indices.append(idx)
        for nei in adj_list[idx]:
            indegrees[nei] -= 1
            if indegrees[nei] == 0:
                queue.append(nei)
    if len(indices) != n: return 'No'
    res = [0]*(n + 1)
    for i, idx in enumerate(indices, start = 1):
        res[idx] = i
    return f"Yes\n{' '.join(map(str, res[1:]))}"
 
if __name__ == '__main__':
    print(main())
```

## F - Teleporter and Closed off 

### Solution 1:  shortest path with dp + store the shortest distance from 1 to ith node and from n to ith node + do with dp + O(nm) time + for k there are these transitions possible k - m < i < k < j < k + m + if can teleport from i -> j while skipping k then take the distance from 1 and n to get the distance to n by skipping kth node + O(nm^2) time

```py
import math
 
def main():
    n, m = map(int, input().split())
    teleporters = [''] + [input() for _ in range(n)]
    min_teleports = [math.inf]*n
    dist_from_start, dist_from_end = [math.inf]*(n + 1), [math.inf]*(n + 1)
    dist_from_start[1] = dist_from_end[n] = 0    for i in range(2, n + 1):
        for j in range(max(1, i - m), i):
            if teleporters[j][i - j - 1] == '1':
                dist_from_start[i] = min(dist_from_start[i], dist_from_start[j] + 1)
    for i in range(n - 1, 0, -1):
        for j in range(i + 1, min(n + 1, i + m + 1)):
            if teleporters[i][j - i - 1] == '1':
                dist_from_end[i] = min(dist_from_end[i], dist_from_end[j] + 1)
    for k in range(2, n):
        for i in range(max(1, k - m), k):
            for j in range(k + 1, min(n + 1, i + m + 1)):
                if teleporters[i][j - i - 1] == '1':
                    min_teleports[k] = min(min_teleports[k], dist_from_start[i] + dist_from_end[j] + 1)
    min_teleports = [t if t < math.inf else -1 for t in min_teleports]
    return ' '.join(map(str, min_teleports[2:]))
 
if __name__ == '__main__':
    print(main())
```

## G - OR Sum 

### Solution 1:

```py

```



# Atcoder Beginner Contest 292

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

```py

```

## 

### Solution 1:

```py

```



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

# Atcoder Beginner Contest 294

## What is used at the top of each submission

```py
import os,sys
from io import BytesIO, IOBase
sys.setrecursionlimit(10**6)
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

## A - Swap Odd and Even 

### Solution 1: 

```py

```

## B - Call the ID Number

### Solution 1:  

```py

```

## C - Make Takahashi Happy 

### Solution 1:  

```py

```

## D - Tying Rope 

### Solution 1:  

```py

```

## F - Sugar Water 2 

### Solution 1: 

```py

```

## G - Distance Queries on a Tree

### Solution 1:  flatten tree with euler tour + path queries on tree + binary lift for finding lca + fenwick tree for range updates and range queries + O(logn + logm)

Everything is 0 indexed, that is for the edges and nodes, not for the euler tour counter.

The idea here is that you put the weight of each edge onto the child node for the edge. Each edge has a construct of parent and child node, if you root the tree with some arbitrary node such as node labeled at 0, then every other node except the root node will have the weight corresponding to the edge that connects it to it's parent node. With this it becomes the same type of question as finding the distance between two nodes in a tree with updates, so you can use euler tour with fenwick tree. Also need method to find lca quickly in order to find distance to lca and compute path distance between two nodes.  

```py
import math

"""
This is an euler tour for weights on edges on the tree

it starts the counter at 1, so it is 1-indexed flattened tree so it works well with fenwick tree.
Fenwick tree requires 1-indexed arrays. 
"""
class EulerTourPathQueries:
    def __init__(self, num_nodes, adj_list):
        num_edges = num_nodes - 1
        self.edge_to_child_node = [0]*num_edges
        self.num_edges = num_edges
        self.adj_list = adj_list
        self.root_node = 0 # root of the tree
        self.enter_counter, self.exit_counter = [0]*num_nodes, [0]*num_nodes
        self.counter = 1
        self.euler_tour(self.root_node, -1)

    def euler_tour(self, node: int, parent_node: int):
        self.enter_counter[node] = self.counter
        self.counter += 1
        for child_node, _, edge_index in self.adj_list[node]:
            if child_node != parent_node:
                self.edge_to_child_node[edge_index] = child_node
                self.euler_tour(child_node, node)
        self.counter += 1
        self.exit_counter[node] = self.counter

    def __repr__(self):
        return f"enter_counter: {self.enter_counter}, exit_counter: {self.exit_counter}"

class FenwickTree:
    def __init__(self, N):
        self.sums = [0 for _ in range(N+1)]

    def update(self, i, delta):
        while i < len(self.sums):
            self.sums[i] += delta
            i += i & (-i)

    def query(self, i):
        res = 0
        while i > 0:
            res += self.sums[i]
            i -= i & (-i)
        return res

    def __repr__(self):
        return f"array: {self.sums}"
    

class BinaryLift:
    """
    This binary lift function works on any undirected graph that is composed of
    an adjacency list defined by graph
    """
    def __init__(self, node_count: int, graph: List[List[int]]):
        self.size = node_count
        self.graph = graph # pass in an adjacency list to represent the graph
        self.depth = [0]*node_count
        self.parents = [-1]*node_count
        self.visited = [False]*node_count
        # ITERATE THROUGH EACH POSSIBLE TREE
        for node in range(node_count):
            if self.visited[node]: continue
            self.visited[node] = True
            self.get_parent_depth(node)
        self.maxAncestor = 18 # set it so that only up to 2^18th ancestor can exist for this example
        self.jump = [[-1]*self.maxAncestor for _ in range(self.size)]
        self.build_sparse_table()
        
    def build_sparse_table(self) -> None:
        """
        builds the jump sparse arrays for computing the 2^jth ancestor of ith node in any given query
        """
        for j in range(self.maxAncestor):
            for i in range(self.size):
                if j == 0:
                    self.jump[i][j] = self.parents[i]
                elif self.jump[i][j-1] != -1:
                    prev_ancestor = self.jump[i][j-1]
                    self.jump[i][j] = self.jump[prev_ancestor][j-1]
                    
    def get_parent_depth(self, node: int, parent_node: int = -1, depth: int = 0) -> None:
        """
        Fills out the depth array for each node and the parent array for each node
        """
        self.parents[node] = parent_node
        self.depth[node] = depth
        for nei_node, _, _ in self.graph[node]:
            if self.visited[nei_node]: continue
            self.visited[nei_node] = True
            self.get_parent_depth(nei_node, node, depth+1)

    def distance(self, p: int, q: int) -> int:
        """
        Computes the distance between two nodes
        """
        lca = self.find_lca(p, q)
        return self.depth[p] + self.depth[q] - 2*self.depth[lca]

    def find_lca(self, p: int, q: int) -> int:
        # ASSUME NODE P IS DEEPER THAN NODE Q   
        if self.depth[p] < self.depth[q]:
            p, q = q, p
        # PUT ON SAME DEPTH BY FINDING THE KTH ANCESTOR
        k = self.depth[p] - self.depth[q]
        p = self.kthAncestor(p, k)
        if p == q: return p
        for j in range(self.maxAncestor)[::-1]:
            if self.jump[p][j] != self.jump[q][j]:
                p, q = self.jump[p][j], self.jump[q][j] # jump to 2^jth ancestor nodes
        return self.jump[p][0]
    
    def kthAncestor(self, node: int, k: int) -> int:
        while node != -1 and k>0:
            i = int(math.log2(k))
            node = self.jump[node][i]
            k-=(1<<i)
        return node

def main():
    n = int(input())
    adj_list = [[] for _ in range(n)]
    values = [0]*n
    for i in range(n - 1):
        u, v, w = map(int, input().split())
        u -= 1
        v -= 1
        adj_list[u].append((v, w, i))
        adj_list[v].append((u, w, i))
    euler_tour = EulerTourPathQueries(n, adj_list)
    stack = [0]
    vis = [0]*n
    vis[0] = 1
    fenwick = FenwickTree(2*n + 2)
    while stack:
        node = stack.pop()
        for nei, wei, _ in adj_list[node]:
            if vis[nei]: continue
            enter_counter, exit_counter = euler_tour.enter_counter[nei], euler_tour.exit_counter[nei]
            values[nei] = wei
            fenwick.update(enter_counter, wei)
            fenwick.update(exit_counter, -wei)
            vis[nei] = 1
            stack.append(nei)
    binary_lifting = BinaryLift(n, adj_list)
    q = int(input())
    for _ in range(q):
        t, u, v = map(int, input().split())
        if t == 1:
            node = euler_tour.edge_to_child_node[u - 1]
            enter_counter, exit_counter = euler_tour.enter_counter[node], euler_tour.exit_counter[node]
            delta = v - values[node]
            values[node] = v
            fenwick.update(enter_counter, delta)
            fenwick.update(exit_counter, -delta)
        else:
            u -= 1
            v -= 1
            lca_uv = binary_lifting.find_lca(u, v)
            dist_u, dist_v, dist_lca = map(fenwick.query, (euler_tour.enter_counter[u], euler_tour.enter_counter[v], euler_tour.enter_counter[lca_uv]))
            print(dist_u + dist_v - 2*dist_lca)

if __name__ == '__main__':
    main()
```

# Atcoder Beginner Contest 295

## What is used at the top of each submission

```py
import os,sys
from io import BytesIO, IOBase
sys.setrecursionlimit(10**6)
from typing import *
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

## A - Probably English 

### Solution 1: set

```py
def main():
    n = int(input())
    words = map(str, input().split())
    lookup = set(['and', 'not', 'that', 'the', 'you'])
    if any(word in lookup for word in words):
        print('Yes')
    else:
        print('No')

if __name__ == '__main__':
    main()
```

## B - Bombs 

### Solution 1:  brute force bfs from each bomb

```py
from itertools import product
 
def main():
    R, C = map(int, input().split())
    grid = [list(input()) for _ in range(R)]
    empty, wall = '.', '#'
    manhattan_distance = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])
    for r, c in product(range(R), range(C)):
        if grid[r][c] in '.#': continue
        area = int(grid[r][c])
        grid[r][c] = empty
        vis = set()
        vis.add((r, c))
        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            for nr, nc in [(cr + 1, cc), (cr - 1, cc), (cr, cc + 1), (cr, cc - 1)]:
                if 0 <= nr < R and 0 <= nc < C and (nr, nc) not in vis and manhattan_distance((r, c), (nr, nc)) <= area:
                    if grid[nr][nc] == wall: grid[nr][nc] = empty
                    vis.add((nr, nc))
                    stack.append((nr, nc))
    result = '\n'.join(''.join(row) for row in grid)
    print(result)
 
if __name__ == '__main__':
    main()
```

## C - Socks

### Solution 1:  sum

```py
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    counts = Counter(arr)
    print(sum([val//2 for val in counts.values()]))

if __name__ == '__main__':
    main()
```

## D - Three Days Ago 

### Solution 1: prefix + bitmask

Just keep track of a bitmask for the digits for when it has even and odds and sum them up. 

```py
from collections import Counter

def main():
    arr = map(int, list(input()))
    mask_counts = Counter({0: 1})
    prefix_mask = res = 0
    for num in arr:
        prefix_mask ^= (1 << num)
        res += mask_counts[prefix_mask]
        mask_counts[prefix_mask] += 1
    print(res)

if __name__ == '__main__':
    main()
```

## E - Kth Number

### Solution 1:  statistics + binomial distribution + probability 

I wrote down my interpretation of the solution in notes app.

![part 1](images/kth_number/kth_number-1.png)
![part 2](images/kth_number/kth_number-2.png)
![part 3](images/kth_number/kth_number-3.png)

```py
from collections import Counter

def mod_inverse(num, mod):
    return pow(num, mod - 2, mod)

def main():
    # integers between 1 and m
    n, m, k = map(int, input().split())
    freq = Counter(map(int, input().split()))
    mod = 998244353
    fact = [1]*(n + 1)
    for i in range(1, n + 1):
        fact[i] = (fact[i - 1] * i) % mod
    inv_fact = [1]*(n + 1)
    inv_fact[-1] = mod_inverse(fact[-1], mod)
    for i in range(n - 1, -1, -1):
        inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % mod
    def nCr(n, r):
        if n < r: return 0
        return (fact[n] * inv_fact[r] * inv_fact[n - r]) % mod
    def pmf_binomial_distribution(n, m, p):
        return (nCr(n, m) * pow(p, m, mod) * pow(1 - p, n - m, mod))%mod
    cnt_zeros = freq[0]
    # total prob = sum from i = 1 to m of prob(A_k >= i)
    count_less_than_i = total_prob = 0
    for i in range(1, m + 1):
        needed_zeros_to_replace = max(0, k - count_less_than_i)
        if needed_zeros_to_replace == 0: continue
        # p is the probability of success for replacing a 0 with integer less than i
        p = ((i - 1) * mod_inverse(m, mod)) % mod
        prob_less_than_i = 0
        for j in range(needed_zeros_to_replace, cnt_zeros + 1):
            prob_less_than_i = (prob_less_than_i + pmf_binomial_distribution(cnt_zeros, j, p)%mod)%mod
        prob_greater_than_equal_to_i = 1 - prob_less_than_i
        total_prob = (total_prob + prob_greater_than_equal_to_i)%mod
        count_less_than_i += freq[i]
    print(total_prob)

if __name__ == '__main__':
    main()
```

## F - substr = S

### Solution 1:   

```py

```

# Atcoder Beginner Contest 296

## What is used at the top of each submission

```py
import os,sys
from io import BytesIO, IOBase
sys.setrecursionlimit(10**6)
from typing import *
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

## D - M<=ab

### Solution 1:

```py
import math
import bisect

def main():
    n, m = map(int, input().split())
    if m <= n: return m
    res = math.inf
    f1 = min(n, math.ceil(m/n))
    f2 = bisect.bisect_left(range(n), math.ceil(m/f1))
    if f1*f2 >= m: 
        res = min(res, f1*f2)
    f1 = min(n, m//n)
    f2 = bisect.bisect_left(range(n), math.ceil(m/f1))
    if f1*f2 >= m:
        res = min(res, f1*f2)
    if res >= m and res != math.inf: return res
    return -1

if __name__ == '__main__':
    print(main())
```

## E - Transition Game

### Solution 1:

```py
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    adj_list = [-1]*(n + 1)
    for i in range(n):
        u, v = i + 1, arr[i]
        adj_list[u] = v
    on_stack = [0]*(n + 1)
    disc = [0]*(n + 1)
    time = 0
    res = 0
    def dfs(node):
        nonlocal res, time
        stack = []
        while not disc[node]:
            time += 1
            disc[node] = time
            on_stack[node] = 1
            stack.append(node)
            node = adj_list[node]
        if on_stack[node]:
            while stack:
                res += 1
                x = stack.pop()
                on_stack[x] = 0
                if x == node: break
        while stack:
            x = stack.pop()
            on_stack[x] = 0

    for i in range(1, n + 1):
        if disc[i]: continue
        dfs(i)
    return res

if __name__ == '__main__':
    print(main())
```

## G - Polygon and Points

### Solution 1:  binary search + online query + outer product + sort by angle from polygon vertex + fan triangulation of convex polygon

This is a description of the algorithm, while the images below will give a better idea of what is going on.
We can use this lemma below and is why this works.
lemma 1: fan triangulation works for any convex polygon
1. Use fan triangulation to triangulate the polygon.  Use the first vertex as the initial vertex.
1. binary search for which triangle the point belongs to based on fact that each triangle is sorted by angle with respect to initial vertex in polygon.
1. Check if it is in a triangle and on boundary of the polygon edge
1. check if it is on boundary of polygon edge that is adjacent to the non-triangular region outside of the polygon.
1. Check if it is inside one of the triangles of the polygon from the fan triangulation method.
1. all other cases it would be outside of the polygon

![image](images/points_in_convex_polygon_1.PNG)
![image](images/points_in_convex_polygon_2.PNG)
![image](images/points_in_convex_polygon_3.PNG)
![image](images/points_in_convex_polygon_4.PNG)

```py
outer_product = lambda v1, v2: v1[0]*v2[1] - v1[1]*v2[0]

def binary_search(point, polygon):
    n = len(polygon)
    left, right = 0, n
    v = (point[0] - polygon[0][0], point[1] - polygon[0][1])
    while left < right:
        mid = (left + right) >> 1
        v1 = (polygon[mid][0] - polygon[0][0], polygon[mid][1] - polygon[0][1])
        outer_prod = outer_product(v1, v)
        if outer_prod >= 0:
            left = mid + 1
        else:
            right = mid
    return left

def main():
    n = int(input())
    polygon = []
    for _ in range(n):
        polygon.append(tuple(map(int, input().split())))
    q = int(input())
    points = []
    for _ in range(q):
        points.append(tuple(map(int, input().split())))
    res = []
    for p in points:
        v = (p[0] - polygon[0][0], p[1] - polygon[0][1])
        i = binary_search(p, polygon) - 1
        p0, p1, p2 = polygon[0], polygon[i], polygon[(i + 1)%n]
        v1, v2 = (p2[0] - p1[0], p2[1] - p1[1]), (p[0] - p1[0], p[1] - p1[1])
        v3 = (p1[0] - p0[0], p1[1] - p0[1])
        edge_outer_prod = outer_product(v1, v2)
        boundary_outer_prod = outer_product(v3, v)
        # boundary cases
        if 0 < i < n - 1:
            if edge_outer_prod == 0:
                res.append('ON')
                continue
        if i == 1 or i == n - 1:
            if boundary_outer_prod == 0 and min(p0[0], p1[0]) <= p[0] <= max(p0[0], p1[0]) and min(p0[1], p1[1]) <= p[1] <= max(p0[1], p1[1]):
                res.append('ON')
                continue
        # check if inside triangle and therefore polygon
        if 0 < i < n - 1 and edge_outer_prod > 0:
            res.append('IN')
        else:
            res.append('OUT')
    return '\n'.join(res)

if __name__ == '__main__':
    print(main())

"""
example that is same as image above
5
3 4
6 2
8 4
6 7
4 7
15
2 6
2 2
6 3
4 5
8 2
8 7
6 8
1 4
5 7
6 7
4 7
6 2
9 0
5 10
3 4

OUT
OUT
IN
IN
OUT
OUT
OUT
OUT
ON
ON
ON
ON
OUT
OUT
ON
"""
```

# Atcoder Beginner Contest 297

## What is used at the top of each submission

```py
import os,sys
from io import BytesIO, IOBase
sys.setrecursionlimit(10**6)
from typing import *
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

## A - Double Click 

### Solution 1:  loop

```py
def main():
    n, d = map(int, input().split())
    arr = list(map(int, input().split()))
    for i in range(1, n):
        if arr[i] - arr[i - 1] <= d: return arr[i]
    return -1
 
if __name__ == '__main__':
    print(main())
```

## B - chess960 

### Solution 1: string

```py
def main():
    s = input()
    b_locations = []
    r_locations = []
    k_loc = None
    for i in range(len(s)):
        if s[i] == "B": b_locations.append(i)
        elif s[i] == "R": r_locations.append(i)
        elif s[i] == 'K': k_loc = i
    return "Yes" if b_locations[0]%2 != b_locations[1]%2 and r_locations[0] < k_loc < r_locations[1] else "No"
 
if __name__ == '__main__':
    print(main())
```

## C - PC on the Table 

### Solution 1:  matrix loop

```py
def main():
    h, w = map(int, input().split())
    mat = [list(input()) for _ in range(h)]
    for i in range(h):
        j = 0
        while j + 1 < w:
            if mat[i][j] == mat[i][j + 1] == 'T':
                mat[i][j] = 'P'
                mat[i][j + 1] = 'C'
                j += 1
            j += 1
    for row in mat:
        print(''.join(row))
 
if __name__ == '__main__':
    main()
```

## D - Count Subtractions 

### Solution 1:  math

Observe that the difference a - b is always equal to the b (smaller element) and will be the same for some multiple times until a becomes equal to or smaller than b. 

```py
def main():
    a, b = map(int, input().split())
    res = 0
    while a != b:
        if a < b:
            a, b = b, a
        m = a // b
        if m > 1: m -= 1
        a -= m*b
        res += m
    print(res)

if __name__ == '__main__':
    main()
```

## E - Kth Takoyaki Set 

### Solution 1:  pointer pointing to smallest value for each coin + add coin to cost that is smallest

```py
import math
    
def main():
    n, k = map(int, input().split())
    coins = list(set(map(int, input().split())))
    pointers = [0]*len(coins)
    costs = [0]*(k + 1)
    for r in range(1, k + 1):
        cost = math.inf
        for i in range(len(coins)):
            j = pointers[i]
            ncost = costs[j] + coins[i]
            if ncost < cost:
                cost = ncost
        costs[r] = cost
        for i in range(len(coins)):
            while costs[pointers[i]] + coins[i] == cost:
                pointers[i] += 1
    print(costs[-1])
 
if __name__ == '__main__':
    main()
```

## F - Minimum Bounding Box 2 

### Solution 1:

```py

```

## G - Constrained Nim 2 

### Solution 1:  sprague grund theorem + nim game + impartial games

Just look at sprague grundy numbers and find a pattern that reduces the required operations/iterations

What you find is that you can represent the nim value for each pile as the p%(L+R)//L and this kind of looks like this for 3, 14
nim values will be 
0: 0,1,2
1: 3,4,5
2: 6,7,8
3: 9,10,11
4: 12,13,14
5: 15, 16

Which kind of makes sense if you think about the 1 nim value, if you have 3, 4, 5 only 1 possible move can happen, you take 3 stones and no more moves are possible after that so in a sense the 3,4,5 represent a single stone in the classical nim game.  And 0 makes sense as well it represents 0 stones, nobody can take stones.
But for 2, why should this represent 2 stones, it's optional you could take all 8 stones but if you take minimum you can take from it at most 2 times.  It is a bit difficult to prove this but if you take an exmaple of piles = [4, 7], so they are 1 and 2, you get 1^2 = 3, and you can check that the first player can win no matter what happens. And you can check with 1^1 or 2^2 that first player can't win. 

```py
import operator
from functools import reduce

def main():
    N, L, R = map(int, input().split())
    piles = list(map(int, input().split()))
    xor_sum = reduce(operator.xor, [p%(L + R)//L for p in piles])
    return "First" if xor_sum > 0 else "Second"

if __name__ == '__main__':
    print(main())
```

This one belows help find the pattern, cause it is finding winning and losing states with brute force algorithm.  So it is deadly slow but that is how you can start finding the solution. 

```py
def main(N, L, R, piles):
    total = sum(piles)
    def grundy(idx, remaining, num_piles):
        # winning state for player
        if num_piles == 1 and L <= remaining <= R: return 1 
        # losing state for player
        if num_piles == 1 and remaining < L: return 0
        grundy_numbers = set()
        for i in range(N):
            for take in range(L, min(R, piles[i]) + 1):
                piles[i] -= take
                new_num_piles = num_piles - (1 if piles[i] < L else 0)
                grundy_numbers.add(grundy(idx + 1, remaining - take, new_num_piles))
                piles[i] += take
        res = next(dropwhile(lambda i: i in grundy_numbers, range(100_000)))
        return res
    num_piles = sum((1 for p in piles if p >= L))
    grundy_number = grundy(0, total, num_piles)
    return "First" if grundy_number > 0 else "Second"
```

# Atcoder Beginner Contest 298

## What is used at the top of each submission

```py
import os,sys
from io import BytesIO, IOBase
sys.setrecursionlimit(10**6)
from typing import *
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

## E - Unfair Sugoroku

### Solution 1:

```py

```

##

### Solution 1:

```py

```

# Atcoder Beginner Contest 299

## What is used at the top of each submission

```py
import os,sys
from io import BytesIO, IOBase
sys.setrecursionlimit(10**6)
from typing import *
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

## A - Treasure Chest

### Solution 1:  loop + string index and rindex

```py
def main():
    n = int(input())
    s = input()
    first, last = s.index('|'), s.rindex('|')
    for i in range(n):
        if s[i] == '*' and first < i < last:
            return 'in'
    return 'out'
 
if __name__ == '__main__':
    print(main())
```

## B - 	Trick Taking

### Solution 1:  max

custom max based on if correct color and maximized on rank but return the player or index

```py
def main():
    n, t = map(int, input().split())
    colors = list(map(int, input().split()))
    ranks = list(map(int, input().split()))
    if t not in colors:
        t = colors[0]
    if t in colors:
        return max(range(n), key = lambda i: ranks[i] if colors[i] == t else 0) + 1
 
if __name__ == '__main__':
    print(main())
```

## C - 	Dango

### Solution 1:  sliding window

The tricky part to this one is that anytime you see a - character you need to move the left = right + 1, cause you only want to count the o characters.  Also another thing to be careful of is when -oooo, which should be level 4 dango.  This means you need that if left > 0 condition to check for when there appeard a - before the o characters. even in this case it is important oooo-oooooo, cause the best one is after that last - character.

```py
def main():
    n = int(input())
    s = input()
    left = res = 0
    for right in range(n):
        if s[right] == '-':
            res = max(res, right - left)
            left = right + 1
    if left > 0:
        res = max(res, n - left)
    res = res if res > 0 else -1
    print(res)
 
if __name__ == '__main__':
    main()
```

## D - Find by Query

### Solution 1:  binary search

Can binary search because guaranteed that s1 = 0 and sn = 1, so really just need to find the rightmost 0 in a sense that precedes a 1. kind of local maximum idea, cause the truth is if you are looking for last T 

0010110000011
TTFTFFTTTTTFF
so you might say this doesn't work for binary search, but it is fine, if it get's stuck in a local region that region will still contain some form of 
T...TF...F regardless

```py
def main():
    n = int(input())
    left, right = 1, n - 1
    while left < right:
        mid = (left + right + 1) >> 1
        print(f"? {mid}", flush = True)
        resp = int(input())
        if resp == 0:
            left = mid
        else:
            right = mid - 1
    print(f"! {left}", flush = True)
 
if __name__ == '__main__':
    main()
```

## E - Nearest Black Vertex

### Solution 1:  max heap + multisource bfs + memoization 

1. max heap to paint all nodes white that would not satisfy the minimum distance to a node painted black.
1. multisource bfs to find that the minimum distance to every black node is still equal to minimum distance

set all nodes to be painted black, than set up a max heap based on the distance required from the current node that all nodes need to be painted white. Then paint all then nodes white. By using the max heap and storing the max_dist, it prevents it from recomputing on nodes cause it will have already painted on neighbor nodes farther than distance from that node.  the max heap makes certain you use the larger distances first. 

After painting all the nodes that need to be white, you just need to check that there is at least one remaining painted black node, and perform a multisource bfs from each black node and record the minimum distance to every node in the graph. If the minimum distance is greater than the required minimum distance for node then you need to return False, there is no solution.  What this means is there was no valid way to have a node painted black at the minimum distance and satisfy the distance requirements of all the nodes. 

To understand this further the best thing to do is draw out a simple undirected graph and have a few nodes black.  Although some reason I just got this one and didn't need to do a crazy proof. All I did was think about the previous two steps above.  

One concern I had was that I can just iterate from every node that has a minimun distance to a node painted black. Because there would be so much recomputation, if I visite a node 2 times that doesn't make sense, I made observation that if ai visit a node a second time but the remaining distance to the nearest node painted black is less than the previous time, there is no reason to revisit, cause the previous visit will reach farther and all the nodes that are needed to be painted white.  I also realize the best way to guarantee that it uses the larger distances first is to use a max heap.  That will avoid recomputation.  Cause if a node has distance = 5, and another one distance = 2, I will explore from the distance = 5 first, until it becomes smaller than or equal to distance = 2, and on the same playing field.  Therefore technically the first time I visit a node it is guaranteed to also be the largest distance.  This means I probably don't need the max_dist array but just to store visited. 

Now that I've painted all the nodes white that need to be white, cause if they were not the minimum distance to a node painted black would be too small.  Now you just need to check that the minimum distances are correct and in the course of doing this you didn't make it that from a node the minimum distance to a node painted black is now greater than what is required.  Which could happen because of the requirements of another node prevents it.  BFS implemented to do this because can store minimum number of edges traversed as I explore out from all the nodes painted black. 

```py
from collections import deque
import heapq
import math

def main():
    n, m = map(int, input().split())
    adj_list = [[] for _ in range(n)]
    for _ in range(m):
        u, v = map(int, input().split())
        adj_list[u-1].append(v-1)
        adj_list[v-1].append(u-1)
    k = int(input())
    max_heap = []
    max_dist = [0]*n
    min_dist = [math.inf]*n
    for _ in range(k):
        p, d = map(int, input().split())
        p -= 1
        min_dist[p] = d
        if d > 0:
            heapq.heappush(max_heap, (-d, p))
            max_dist[p] = d
    result = [1]*n
    while max_heap:
        dist, node = heapq.heappop(max_heap)
        result[node] = 0
        dist = -dist
        if dist < max_dist[node]: continue
        for nei in adj_list[node]:
            if dist - 1 <= max_dist[nei]: continue
            max_dist[nei] = dist - 1
            heapq.heappush(max_heap, (-(dist-1), nei))
    def bfs():
        vis = [0]*n
        queue = deque()
        for i in range(n):
            if result[i] == 1:
                queue.append(i)
                vis[i] = 1
        dist = 0
        while queue:
            for _ in range(len(queue)):
                node = queue.popleft()
                if dist > min_dist[node]: return False
                for nei in adj_list[node]:
                    if vis[nei]: continue
                    vis[nei] = 1
                    queue.append(nei)
            dist += 1
        return True
    if sum(result) == 0 or not bfs():
        print('No')
        return
    print('Yes')
    print(''.join(map(str, result)))

if __name__ == '__main__':
    main()
```

##  F - Square Subsequence

### Solution 1:

```py

```

## G - Minimum Permutation

### Solution 1:

```py

```

# Atcoder Beginner Contest 300

## What is used at the top of each submission

```py
import os,sys
from io import BytesIO, IOBase
sys.setrecursionlimit(10**6)
from typing import *
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

## A - N-choice question

### Solution 1:  loop

```py
def main():
    n, a, b = map(int, input().split())
    c = list(map(int, input().split()))
    for i, v in enumerate(c, start = 1):
        if a + b == v: return i
    return n
 
if __name__ == '__main__':
    print(main())
```

## B - Same Map in the RPG World 

### Solution 1:  modulus + all + cartesian product loop

```py
from itertools import product
 
def main():
    h, w = map(int, input().split())
    arr1 = [list(input()) for _ in range(h)]
    arr2 = [list(input()) for _ in range(h)]
    for r, c in product(range(h), range(w)):
        if all(arr1[(r + i)%h][(c + j)%w] == arr2[i][j] for i, j in product(range(h), range(w))):
            return 'Yes'
    return 'No'
 
if __name__ == '__main__':
    print(main())
```

## C - Cross 

### Solution 1:  dfs 

dfs from each source of cross, and check if the continually larger cross is valid.  anything invalid with a size = 1 is said to be a cross of size 0, which is basically not a cross. 

```py
def main():
    h, w = map(int, input().split())
    grid = [list(input()) for _ in range(h)]
    cross = '#'
    n = min(h, w)
    res = [0]*(n + 1)
    stack = [(r, c, 0) for r, c in product(range(h), range(w)) if grid[r][c] == cross]
    in_bounds = lambda r, c: 0 <= r < h and 0 <= c < w
    while stack:
        r, c, size = stack.pop()
        flag = False
        for nr, nc in ((r + size, c + size), (r - size, c + size), (r + size, c + size), (r - size, c - size)):
            if not in_bounds(nr, nc) or grid[nr][nc] != cross:
                flag = True
                continue
        if flag:
            res[size - 1] += 1
        else:
            stack.append((r, c, size + 1))
    return ' '.join(map(str, res[1:]))
                    
if __name__ == '__main__':
    print(main())
```

## D - AABCC 

### Solution 1:  prime sieve + math + number theory

The integers needed are less than expected because these are two the power, easiest way is two loops through the squared terms because these will terminate quickly.  And move just the middle pointer to the just small enough prime, if not possible then impossible with this c and any larger c.

```py
import math
 
def main():
    n = int(input())
    def prime_sieve(lim):
        sieve,primes = [[] for _ in range(lim)], []
        for integer in range(2,lim):
            if not len(sieve[integer]):
                primes.append(integer)
                for possibly_divisible_integer in range(integer,lim,integer):
                    current_integer = possibly_divisible_integer
                    while not current_integer%integer:
                        sieve[possibly_divisible_integer].append(integer)
                        current_integer //= integer
        return primes
    threshold = math.ceil(math.sqrt(n/12))
    primes = prime_sieve(threshold + 1)
    m = len(primes)
    res = 0
    for a in range(m):
        b = a + 1
        for c in range(a + 2, m):
            constant = primes[a]**2*primes[c]**2
            while constant*primes[b] > n and b > a + 1:
                b -= 1
            if constant*primes[b] > n: break
            while constant*primes[b + 1] <= n and b + 1 < c:
                b += 1
            res += b - a
    return res
 
                    
if __name__ == '__main__':
    print(main())
```

## E - Dice Product 3 

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

# Atcoder Beginner Contest 301

## What is used at the top of each submission

```py
import os,sys
from io import BytesIO, IOBase
sys.setrecursionlimit(10**6)
from typing import *
# only use pypyjit when needed, it usese more memory, but speeds up recursion in pypy
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')
# sys.stdout = open('output.txt', 'w')

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
                    
if __name__ == '__main__':
    print(main())
    # main()
    # sys.stdout.close()
```

##

### Solution 1:  

```py

```

##

### Solution 1:  

```py

```

## C - AtCoder Cards

### Solution 1:  frequency arrays + counters + anagram

```py
from itertools import product
import math
import heapq
from collections import Counter
import string
 
def main():
    s = input()
    t = input()
    freq_s, freq_t = Counter(s), Counter(t)
    free_s, free_t = s.count('@'), t.count('@')
    chars = 'atcoder'
    for ch in string.ascii_lowercase:
        if freq_s[ch] != freq_t[ch] and ch not in chars: return 'No'
        if freq_s[ch] < freq_t[ch]:
            delta = freq_t[ch] - freq_s[ch]
            if free_s < delta: return 'No'
            free_s -= delta
        elif freq_s[ch] > freq_t[ch]:
            delta = freq_s[ch] - freq_t[ch]
            if free_t < delta: return 'No'
            free_t -= delta
    return 'Yes'
 
                    
if __name__ == '__main__':
    print(main())
```

## D - Bitmask 

### Solution 1:  bit manipulation + greedy

```py
def main():
    s = list(reversed(input()))
    n = int(input())
    res = 0
    for i in range(len(s)):
        if s[i] == '1':
            res |= (1 << i)
    if res > n: return -1
    for i in reversed(range(len(s))):
        if s[i] == '?' and (res | (1 << i)) <= n:
            res |= (1 << i)
    return res
if __name__ == '__main__':
    print(main())
```

## E - Pac-Takahashi 

### Solution 1:  traveling salesman problem + dp bitmask + (n^2 * 2^n) time

Given a list of cities and the distances between each pair of cities, what is the shortest possible route that visits each city exactly once and returns to the origin city?

minimize number of moves for the dp[i][mask], where i is the current node you are visiting and mask contains the set of all nodes that have been visited, which in this case will tell how many candies have been collected. 

Find the shortest distance between all pairs of vertices with bfs with the adjacency matrix

loop through each mask or set of visited vertices, then loop through the src and dst vertex and consider this. 

```py
from itertools import product
import math
from collections import deque

def main():
    H, W, T = map(int, input().split())
    A = [input() for _ in range(H)]
    start, goal, empty, wall, candy = ['S', 'G', '.', '#', 'o']
    in_bounds = lambda r, c: 0 <= r < H and 0 <= c < W
    neighborhood = lambda r, c: [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
    # CONSTRUCT THE VERTICES
    start_pos = goal_pos = None
    nodes = []
    index = {}
    for i, j in product(range(H), range(W)):
        if A[i][j] == start:
            start_pos = (i, j)
        elif A[i][j] == goal:
            goal_pos = (i, j)
        elif A[i][j] == candy:
            nodes.append((i, j))
    nodes.extend([goal_pos, start_pos])
    for i, (r, c) in enumerate(nodes):
        index[(r, c)] = i
    V = len(nodes)
    adj_matrix = [[math.inf]*V for _ in range(V)] # complete graph so don't need adjacency list
    # CONSTRUCT THE EDGES WITH BFS
    for i, (r, c) in enumerate(nodes):
        queue = deque([(r, c)])
        dist = 0
        vis = set([(r, c)])
        while queue:
            dist += 1
            for _ in range(len(queue)):
                row, col = queue.popleft()
                for nr, nc in neighborhood(row, col):
                    if not in_bounds(nr, nc) or A[nr][nc] == wall or (nr, nc) in vis:
                        continue
                    if (nr, nc) in index:
                        adj_matrix[i][index[(nr, nc)]] = dist
                    vis.add((nr, nc))
                    queue.append((nr, nc))
    # TRAVELING SALESMAN PROBLEM
    dp = [[math.inf]*(1 << (V - 1)) for _ in range(V)]
    dp[V - 1][0] = 0 # start at node 0 with no candy
    for mask in range(1 << (V - 1)):
        for i in range(V):
            if dp[i][mask] == math.inf: continue
            for j in range(V - 1):
                nmask = mask | (1 << j)
                dp[j][nmask] = min(dp[j][nmask], dp[i][mask] + adj_matrix[i][j])
    res = -1
    for mask in range(1 << (V - 1)):
        if dp[V - 2][mask] > T: continue
        res = max(res, bin(mask).count('1') - 1)
    return res
                    
if __name__ == '__main__':
    print(main())

```

##

### Solution 1:  

```py

```
 
## G - Worst Picture 

### Solution 1:  3 dimensional space geometry + computational geometry + line intersection in 3 dimensional space

This problem can be solved in O(n^3) time

summary of steps

1.  find lines from pairs of points
1.  add lines to `lines_of_interest` list if it is not parallel to the yz plane



A line is parallel to a plane if the direction vector of the line is orthogonal to the normal vector of the plane. So for the case when the line has a direction vector that is orthogonal to the normal vector of the yz plane.  Any line that would be parallal to the yz plane is not created because it will never intersect region x < 0.  In other words, you can check if a line would be like this by just looking at if the x1 = x2 of the points that the line goes through.  If they are equal, then the line is parallel to the yz plane and you can skip adding it to lines of interest.

good test case cause it include many points that can be colinear, see image of what it looks like in 3d space.  This image includes the line to the point of intersection in the x < 0  region where 4 lines an intersect and only 4 points are visible. 

```txt
11
1 1 1
1 1 -1
1 -1 1
1 -1 -1
3 2 2
3 2 -2
3 -2 2
3 -2 -2
5 3 3
7 4 4
9 5 5
```

![visualization of points](images/worst_picture1.png)



```py

import math
from itertools import product
from collections import defaultdict

def cross(u, v):
    """
    Returns the cross product of two 3D vectors u and v.
    """
    x = u[1] * v[2] - u[2] * v[1]
    y = u[2] * v[0] - u[0] * v[2]
    z = u[0] * v[1] - u[1] * v[0]
    return x, y, z

def norm(u):
    """
    Returns the norm of a 3D vector u.
    """
    return math.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)

def intersect(line1, line2):
    """
    Returns point if line segments line1 and line2 intersect in x < 0 region.
    returns None if the lines line1 and line2 do not intersect in x < 0 region or are parallel.
    """
    # Compute direction vectors and a point on each line
    v1 = (line1[1][0] - line1[0][0], line1[1][1] - line1[0][1], line1[1][2] - line1[0][2])
    # v1 = line1[1] - line1[0]
    p1 = line1[0]
    v2 = (line2[1][0] - line2[0][0], line2[1][1] - line2[0][1], line2[1][2] - line2[0][2])
    # v2 = line2[1] - line2[0]
    p2 = line2[0]

    # Compute normal vector to plane containing both lines
    n = cross(v1, v2)

    # Check if lines are parallel
    if norm(n) < 1e-6:
        return None

    # Compute intersection point
    p2p1 = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
    p1p2 = (p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])
    t1 = dot_product(cross(p2p1, v2), n) / dot_product(cross(v1, v2), n)
    t2 = dot_product(cross(p1p2, v1), n) / dot_product(cross(v2, v1), n)


    # point = p1 + t1 * v1
    # point2 = p2 + t2 * v2
    point1 = (p1[0] + t1*v1[0], p1[1] + t1*v1[1], p1[2] + t1*v1[2])
    point2 = (p2[0] + t2*v2[0], p2[1] + t2*v2[1], p2[2] + t2*v2[2])

    # check for skew lines
    if any(abs(v1 - v2) > 1e-6 for v1, v2 in zip(point1, point2)): return None

    # Check if intersection point is in x < 0 region
    if point1[0] < 0:
        return point1

    return None

def dot_product(u, v):
    """
    Returns the dot product of two 3D vectors u and v.
    """
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]

def parallel(line1, line2):
    """
    Returns True if line1 and line2 are parallel.
    """
    # Compute direction vectors of lines
    v1 = (line1[1][0] - line1[0][0], line1[1][1] - line1[0][1], line1[1][2] - line1[0][2])
    v2 = (line2[1][0] - line2[0][0], line2[1][1] - line2[0][1], line2[1][2] - line2[0][2])
    # v1 = line1[1] - line1[0]
    # v2 = line2[1] - line2[0]

    # Compute cross product of direction vectors
    norm_vec = cross(v1, v2)

    # Check if cross product is zero
    if norm(norm_vec) < 1e-6:
        return True

    return False

def main():
    n = int(input())
    points = [None] * n
    for i in range(n):
        x, y, z = map(int, input().split())
        points[i] = (x, y, z)
    # form all the lines that have delta_x != 0, else the line is in the yz plane, and will never cross into the x < 0 region.
    lines_of_interest = []
    blocked = 0
    counts = []
    for i, j in product(range(n), repeat = 2):
        p1, p2 = points[i], points[j]
        if p1[0] >= p2[0]: continue # must be not equal on x, and p2.x > p1.x
        line = [p1, p2]
        lines_of_interest.append(line)
        cnt = 1
        for k in range(n):
            p3 = points[k]
            if p3[0] <= p1[0] or p3[0] <= p2[0]: continue # requiring that p3.x > p2.x > p1.x
            p2p1 = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
            p3p1 = (p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2])
            normal_vector = cross(p2p1, p3p1)
            if norm(normal_vector) < 1e-6: cnt += 1
        counts.append(cnt)
        blocked = max(blocked, cnt)
    intersections = defaultdict(set)
    # find all the line intersections
    for i, j in product(range(len(lines_of_interest)), repeat = 2):
        line1, line2 = lines_of_interest[i], lines_of_interest[j]
        # point is None means in the region x < 0
        point = intersect(line1, line2)
        if point is None: continue
        good_i = good_j = True
        cur_lines = intersections[tuple(point)]
        not_good = []
        for k in cur_lines:
            line3 = lines_of_interest[k]
            if parallel(line1, line3):
                if counts[i] < counts[k]:
                    good_i = False
                else:
                    not_good.append(k)
            elif parallel(line2, line3):
                if counts[j] < counts[k]:
                    good_j = False
                else:
                    not_good.append(k)
        for k in not_good:
            cur_lines.discard(k)
        if good_i:
            cur_lines.add(i)
        if good_j:
            cur_lines.add(j)
    for intersecting_lines in intersections.values():
        blocked = max(blocked, sum(counts[i] for i in intersecting_lines))
    return n - blocked

if __name__ == '__main__':
    print(main())
```

# Atcoder Beginner Contest 302

## What is used at the top of each submission

```py
import os,sys
from io import BytesIO, IOBase
sys.setrecursionlimit(10**6)
from typing import *
# only use pypyjit when needed, it usese more memory, but speeds up recursion in pypy
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')
# sys.stdout = open('output.txt', 'w')

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
                    
if __name__ == '__main__':
    print(main())
    # main()
    # sys.stdout.close()
```

## A - Attack

### Solution 1:  math

```py
import math
 
def main():
    a, b = map(int, input().split())
    attacks = a // b
    return attacks if a - attacks * b <= 0 else attacks + 1
```

## B - Find snuke

### Solution 1:  bfs

```py
from collections import deque
from itertools import product
 
def solve():
    h, w = map(int, input().split())
    grid = [input() for _ in range(h)]
    target = 'snuke'
    in_bounds = lambda r, c: 0 <= r < h and 0 <= c < w
    main_diag_neighborhood = lambda r, c: [(r + 1, c + 1), (r - 1, c - 1)]
    minor_diag_neighborhood = lambda r, c: [(r + 1, c - 1), (r - 1, c + 1)]
    hor_neighborhood = lambda r, c: [(r, c + 1), (r, c - 1)]
    vert_neighborhood = lambda r, c: [(r + 1, c), (r - 1, c)]
    def bfs(neighborhood):
        queue = deque([(r, c, [(r, c)]) for r, c in product(range(h), range(w)) if grid[r][c] == 's'])
        while queue:
            r, c, points = queue.popleft()
            if len(points) == len(target):
                break
            for nr, nc in neighborhood(r, c):
                if not in_bounds(nr, nc) or grid[nr][nc] != target[len(points)]: continue
                queue.append((nr, nc, points + [(nr, nc)]))
        return points
    main_diag = bfs(main_diag_neighborhood)
    if len(main_diag) == len(target):
        return main_diag
    minor_diag = bfs(minor_diag_neighborhood)
    if len(minor_diag) == len(target):
        return minor_diag
    hor = bfs(hor_neighborhood)
    if len(hor) == len(target):
        return hor
    vert = bfs(vert_neighborhood)
    if len(vert) == len(target):
        return vert
    
def main():
    points = solve()
    for u, v in points:
        print(u + 1, v + 1)
 
if __name__ == '__main__':
    main()
```

## C - Almost Equal

### Solution 1:  bitmask + hamming distance

find if can reach the end_mask meaning visited all nodes given condition can only traverse an edge when edge_weight = 1.  where edge weight is the count of different characters between two strings.

```py
from collections import deque
from itertools import product
 
def solve():
    n, m = map(int, input().split())
    strs = [input() for _ in range(n)]
    dist = [[0]*n for _ in range(n)]
    for i, j in product(range(n), repeat = 2):
        dist[i][j] = sum(1 for k in range(m) if strs[i][k] != strs[j][k])
    vis = [[0]*(1 << n) for _ in range(n)]
    stack = []
    for i in range(n):
        vis[i][1 << i] = 1
        stack.append((i, 1 << i))
    end_mask = (1 << n) - 1
    while stack:
        i, mask = stack.pop()
        if mask == end_mask: return "Yes"
        for j in range(n):
            if dist[i][j] != 1 or ((mask >> j) & 1): continue
            nmask = mask | (1 << j)
            if vis[j][nmask]: continue
            vis[j][nmask] = 1
            stack.append((j, nmask)) 
    return "No"
 
def main():
    print(solve())
 
if __name__ == '__main__':
    main()
```

## D - Impartial Gift

### Solution 1:  sort + two pointers

```py
 def main():
    n, m, d = map(int, input().split())
    A = sorted(list(map(int, input().split())))
    B = sorted(list(map(int, input().split())))
    j = 0
    res = -1
    for i in range(n):
        while j + 1 < m and B[j + 1] <= A[i] + d:
            j += 1
        if abs(A[i] - B[j]) <= d:
            res = max(res, A[i] + B[j])
    print(res)
 
if __name__ == '__main__':
    main()
```

## E - Isolation

### Solution 1:  undirected graph + set adjacency list for removals + degrees

```py
def main():
    n, q = map(int, input().split())
    degrees = [0] * (n + 1)
    adj_list = [set() for _ in range(n + 1)]
    cnt = n
    res = [None] * q
    for i in range(q):
        query = list(map(int, input().split()))
        if query[0] == 1:
            u, v = query[1:]
            cnt -= degrees[u] == 0
            cnt -= degrees[v] == 0
            degrees[u] += 1
            degrees[v] += 1
            adj_list[u].add(v)
            adj_list[v].add(u)
        else:
            u = query[1]
            cnt += degrees[u] > 0
            degrees[u] = 0
            for v in adj_list[u]:
                degrees[v] -= 1
                cnt += degrees[v] == 0
                adj_list[v].discard(u)
            adj_list[u].clear()
        res[i] = cnt
    return '\n'.join(map(str, res))
 
if __name__ == '__main__':
    print(main())
```

## F - Merge Set

### Solution 1:  undirected graph + elements are nodes + sets are nodes + bfs + shortest path from source to destination node

Each edge connects a set to an element

![image](images/merge_sets.PNG)

```py
from collections import deque
 
def main():
    n, m = map(int, input().split())
    adj_list = [[] for _ in range(n + m)]
    for i in range(n):
        _ = input()
        arr = list(map(int, input().split()))
        for num in arr:
            adj_list[num - 1].append(m + i)
            adj_list[m + i].append(num - 1)
    queue = deque([0])
    vis = [0] * (n + m)
    vis[0] = 1
    steps = 0
    while queue:
        for _ in range(len(queue)):
            node = queue.popleft()
            if node == m - 1: return steps // 2 - 1
            for nei in adj_list[node]:
                if vis[nei]: continue
                vis[nei] = 1
                queue.append(nei)
        steps += 1
    return -1
 
if __name__ == '__main__':
    print(main())
```

## G - Sort from 1 to 4

### Solution 1

```py
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    res = 0
    def inversion_count(left, right):
        if right-left <= 1: return 0
        mid = (left+right)>>1
        res = inversion_count(left, mid) + inversion_count(mid, right) + merge(left, right, mid)
        return res

    def merge(left, right, mid):
        i, j = left, mid
        temp = []
        inv_count = 0
        while i < mid and j < right:
            if arr[i] <= arr[j]:
                temp.append(arr[i])
                i += 1
            else:
                temp.append(arr[j])
                print('mid', mid, 'i', i, 'j', j)
                inv_count += (mid - i)
                j += 1
        while i < mid:
            temp.append(arr[i])
            i += 1
        while j < right:
            temp.append(arr[j])
            j += 1
        for i in range(left, right):
            arr[i] = temp[i-left]
        return inv_count

    res = inversion_count(0, n)
    return res

if __name__ == '__main__':
    print(main())
```


# Atcoder Beginner Contest 303

## What is used at the top of each submission

```py
import os,sys
from io import BytesIO, IOBase
sys.setrecursionlimit(10**6)
from typing import *
# only use pypyjit when needed, it usese more memory, but speeds up recursion in pypy
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')
# sys.stdout = open('output.txt', 'w')

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
                    
if __name__ == '__main__':
    print(main())
    # main()
    # sys.stdout.close()
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

## G - Bags Game 

### Solution 1:  dynammic programming + prefix sum + O(n^3) time

This will time out obviously, need a solution that can solve in O(n^2) 

```py
from itertools import accumulate

def main():
    N, A, B, C, D = map(int, input().split())
    arr = list(map(int, input().split()))
    psum = [0] + list(accumulate(arr))
    def psum_query(left, right):
        return psum[right + 1] - psum[left]
    dp = [[0]*(N+1) for _ in range(N + 1)]
    for i in range(N): # base cases
        dp[i][0] = 0
        dp[i][1] = arr[i]
    for j in range(2, N + 1):
        for i in range(N):
            r = i + j - 1
            if r >= N: break # out of bounds, right boundary is i + j - 1
            # TAKE LEFTMOST ELEMENT
            dp[i][j] = arr[i] - dp[i + 1][j - 1]
            # TAKE RIGHTMOST ELEMENT
            dp[i][j] = max(dp[i][j], arr[r] - dp[i][j - 1])
            # TAKING B ELEMENTS AT COST A
            take = min(B, j)
            # take k elements from left side and take - k elements from right side
            for k in range(take + 1):
                dp[i][j] = max(dp[i][j], psum_query(i, r) - psum_query(i + k, r - (take - k)) - dp[i + k][j - take] - A)
            # TAKING D ELEMENTS AT COST C
            take = min(D, j)
            for k in range(take + 1):
                dp[i][j] = max(dp[i][j], psum_query(i, r) - psum_query(i + k, r - (take - k)) - dp[i + k][j - take] - C)
    return dp[0][N]

if __name__ == '__main__':
    print(main())
```

### Solution 2:  dynamic programming + 

```py

```

# Atcoder Beginner Contest 304

## What is used at the top of each submission

```py
import os,sys
from io import BytesIO, IOBase
sys.setrecursionlimit(10**6)
from typing import *
# only use pypyjit when needed, it usese more memory, but speeds up recursion in pypy
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')
# sys.stdout = open('output.txt', 'w')

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
                    
if __name__ == '__main__':
    print(main())
    # main()
    # sys.stdout.close()
```

## A - First Player 

### Solution 1: 

```py
def main():
    n = int(input())
    names = [None] * n
    ages = [None] * n
    for i in range(n):
        name, age = input().split()
        names[i] = name
        ages[i] = int(age)
    index = min(range(n), key=lambda i: ages[i])
    for i in range(n):
        name = names[(index + i) % n]
        print(name)
 
if __name__ == '__main__':
    main()
```

## B - Subscribers 

### Solution 1: 

```py
def main():
    n = list(input())
    for i in range(3, len(n)):
        n[i] = '0'
    return ''.join(n)
 
if __name__ == '__main__':
    print(main())
```

## C - Virus 

### Solution 1:  directed graph + dfs

```py
def main():
    n, d = map(int, input().split())
    people = [tuple(map(int, input().split())) for _ in range(n)]
    euclidean_dist = lambda p1, p2: (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
    adj_list = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i):
            if euclidean_dist(people[i], people[j]) <= d**2:
                adj_list[i].append(j)
                adj_list[j].append(i) 
    stack = [0]
    vis = [0] * n
    vis[0] = 1
    while stack:
        node = stack.pop()
        for nei in adj_list[node]:
            if vis[nei]: continue
            vis[nei] = 1
            stack.append(nei)
    for i in range(n):
        if vis[i]:
            print("Yes")
        else:
            print("No")
 
if __name__ == '__main__':
    main()
```

## D - A Piece of Cake 

### Solution 1:  binary search + grid + counter

```py
import bisect
from collections import Counter
 
def main():
    R, C = map(int, input().split())
    n = int(input())
    strawberries = [tuple(map(int, input().split())) for _ in range(n)]
    A = int(input())
    arr = list(map(int, input().split()))
    B = int(input())
    brr = list(map(int, input().split()))
    counts = Counter()
    for x, y in strawberries:
        vertical = bisect.bisect_left(arr, x)
        horizontal = bisect.bisect_left(brr, y)
        counts[(vertical, horizontal)] += 1
    m = min(counts.values()) if len(counts) == (A + 1) * (B + 1) else 0
    M = max(counts.values())
    print(m, M)
 
if __name__ == '__main__':
    main()
```

## E - Good Graph 

### Solution 1:  condense graph to connected components + union find

```py
class UnionFind:
    def __init__(self, n: int):
        self.size = [1]*n
        self.parent = list(range(n))
    
    def find(self,i: int) -> int:
        while i != self.parent[i]:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i
 
    """
    returns true if the nodes were not union prior. 
    """
    def union(self,i: int,j: int) -> bool:
        i, j = self.find(i), self.find(j)
        if i!=j:
            if self.size[i] < self.size[j]:
                i,j=j,i
            self.parent[j] = i
            self.size[i] += self.size[j]
            return True
        return False
    
    def __repr__(self) -> str:
        return f'parents: {[(i, parent) for i, parent in enumerate(self.parent)]}, sizes: {self.size}'
 
def main():
    n, m = map(int, input().split())
    dsu = UnionFind(n + 1)
    for _ in range(m):
        u, v = map(int, input().split())
        dsu.union(u, v)
    k = int(input())
    seen = set()
    for _ in range(k):
        u, v = map(lambda x: dsu.find(x), map(int, input().split()))
        seen.add((min(u, v), max(u, v)))
    q = int(input())
    for _ in range(q):
        c1, c2 = map(lambda x: dsu.find(x), map(int, input().split()))
        if (min(c1, c2), max(c1, c2)) in seen:
            print("No")
        else:
            print("Yes")
 
if __name__ == '__main__':
    main()
```

## F - Shift Table 

### Solution 1:  count ways for factors + inclusion exclusion in dynammic programming to remove duplicates

This is a tricky problem for me, I still have trouble understanding how the dp part can remove duplicates.  An example of a duplicates is something like this, suppose you have n = 12
M = 3, and M = 6.  And we know that for 3 that these are the elements that are fixed represented by 1 and the others means that Aoki has option to work or not work on that day.  The fixed ones means Aoki must work on that day else, not all days will be worked by either Takahashi or Aoki. 

So suppose input is 
12
####.####.##
then we know that for M = 3, [1,1,0] that is Aoki must work on first and second day, and then this will be repeated through out.  Now we are going to count the number of possiblities here which there is 2. 

Then for M = 6, [0,0,0,1,1,0] there are two days that Aoki must work and the rest there are two possiblities.  We can see from above that there is a duplicate state we will be counting, which is this one, 
[1,1,0,1,1,0], Thus we need to conclucde that any pattern in M = 3 will be repeated in M = 6, so take the patterns in 3 and subtract them from those in M = 6.  That way those repeated patterns will not count in M = 6 and the remaining ones must be patterns that were not in M = 3. 

Remember M = 6 counted patterns that are same as M = 3 and those that are not, so we want to exclude those repeated and include those that are not.  This is the idea of inclusion exclusion.

```py
def main():
    n = int(input())
    s = input()
    mod = 998244353
    factors = []
    for i in range(1, n):
        # i is factor if n is divisible by i
        if n % i == 0: factors.append(i)
    m = len(factors)
    dp = [[0] * m for _ in range(m)]
    for i in range(m):
        dp[i][i] = 1 # i is divisible by i
        for j in range(i):
            if factors[i] % factors[j] == 0: dp[i][j] = 1 # factor_i is divisible by factor_j
    # count the ways
    counts = [0] * m
    for i in range(m):
        # finds position that must be fixed, that is takahashi doesn't work that day so that '.'
        fixed = [0] * factors[i]
        for j in range(n):
            if s[j] == '.': fixed[j % factors[i]] = 1
        unset = factors[i] - sum(fixed)
        counts[i] = pow(2, unset, mod)
    # dynamic programming to remove the duplicates
    for i in range(m):
        for j in range(i):
            if dp[i][j]:
                counts[i] -= counts[j]
    print(counts)
    return sum(counts) % mod

if __name__ == '__main__':
    print(main())
```

## Ex - Constrained Topological Sort 

### Solution 1:  directed graph + backwart and forward topological sort + indegrees for forward top sort and outdegrees for backward top sort + min heaps

Use the available from topological sort with the smallest right values first.  But you need to find the minimum allows right values with the backwards topological sort, because the min right for an element depends on the elements reached from it, because you need to have p be that so you can reach the other ones within the right constraint.  But also you store ones that are not ready because the left constraint has not been met yet. 

```py
from collections import deque
from heapq import heappush, heappop

def main():
    n, m = map(int, input().split())
    indegrees = [0] * (n + 1)
    outdegrees = [0] * (n + 1)
    adj_list = [[] for _ in range(n + 1)]
    rev_adj_list = [[] for _ in range(n + 1)]
    for _ in range(m):
        u, v = map(int, input().split())
        adj_list[u].append(v)
        rev_adj_list[v].append(u)
        indegrees[v] += 1
        outdegrees[u] += 1
    lranges, rranges = [0] * (n + 1), [0] * (n + 1)
    for i in range(1, n + 1):
        left, right = map(int, input().split())
        lranges[i], rranges[i] = left, right
    # reverse topological sort
    queue = deque()
    for i in range(1, n + 1):
        if outdegrees[i] == 0:
            queue.append(i)
    while queue:
        node = queue.popleft()
        for nei in rev_adj_list[node]:
            outdegrees[nei] -= 1
            rranges[nei] = min(rranges[nei], rranges[node] - 1)
            if outdegrees[nei] == 0:
                queue.append(nei)
    # topological sort
    res = [0] * (n + 1)
    p = 1
    ready, not_ready = [], []
    for i in range(1, n + 1):
        if indegrees[i] == 0:
            if lranges[i] > p: heappush(not_ready, (lranges[i], i))
            else: heappush(ready, (rranges[i], i))
    while ready:
        right, node = heappop(ready)
        if p > right: 
            print("No")
            return
        res[node] = p
        p += 1
        while not_ready and not_ready[0][0] <= p:
            _, n1 = heappop(not_ready)
            heappush(ready, (rranges[n1], n1))
        for nei in adj_list[node]:
            indegrees[nei] -= 1
            if indegrees[nei] == 0:
                if lranges[nei] > p: heappush(not_ready, (lranges[nei], nei))
                else: heappush(ready, (rranges[nei], nei))
    if p <= n: 
        print("No")
        return
    print("Yes")
    print(*res[1:])

if __name__ == '__main__':
    main()
```

# Atcoder Beginner Contest 305

## What is used at the top of each submission

```py
import os,sys
from io import BytesIO, IOBase
sys.setrecursionlimit(10**6)
from typing import *
# only use pypyjit when needed, it usese more memory, but speeds up recursion in pypy
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')
# sys.stdout = open('output.txt', 'w')

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
                    
if __name__ == '__main__':
    print(main())
    # main()
    # sys.stdout.close()
```

## A - Water Station 

### Solution 1:  math

```py
def main():
    n = int(input())
    quotient = n // 5
    n1, n2 = quotient * 5, (quotient + 1) * 5
    if abs(n - n1) < abs(n - n2):
        print(n1)
    else:
        print(n2)

if __name__ == '__main__':
    main()
```

## B - ABCDEFG 

### Solution 1:  successor graph + search

```py
def main():
    p, q = map(int, input().split())
    adj_list = {"A": ("B", 3), "B": ("C", 1), "C": ("D", 4), "D": ("E", 1), "E": ("F", 5), "F": ("G", 9)}
    p, q = min(p, q), max(p, q)
    node = p
    res = 0
    while node != q:
        node, cost = adj_list[node]
        res += cost
    print(res)

if __name__ == '__main__':
    main()
```

## C - Snuke the Cookie Picker 

### Solution 1:  matrix + count adjacent cells with cookies, if greater than or equal to 2 than it is cookie taken

```py
def main():
    h, w = map(int, input().split())
    grid = [list(input()) for _ in range(h)]
    in_bounds = lambda r, c: 0 <= r < h and 0 <= c < w
    neighborhood = lambda r, c: [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]
    for r, c in product(range(h), range(w)):
        if grid[r][c] == '#': continue
        cnt = 0
        for nr, nc in neighborhood(r, c):
            if not in_bounds(nr, nc): continue
            cnt += grid[nr][nc] == '#'
        if cnt > 1: return print(f"{r + 1} {c + 1}")

if __name__ == '__main__':
    main()
```

## D - Sleep Log 

### Solution 1:  binary search for prefix and suffix endpoints + count middle with prefix sum

```py
import bisect

def main():
    n = int(input())
    arr = list(map(int, input().split()))
    q = int(input())
    psum = [0] * (n + 1)
    for i in range(2, n, 2):
        psum[i] = psum[i - 1] + (arr[i] - arr[i - 1] if i % 2 == 0 else 0)
    for _ in range(q):
        left, right = map(int, input().split())
        i, j = bisect.bisect_left(arr, left), bisect.bisect_left(arr, right)
        cur = 0
        if i % 2 == 0:
            cur += arr[i] - left
        if j % 2 == 0:
            cur += right - arr[j - 1]
        mid = psum[j - 1] - psum[i]
        cur += mid
        print(cur)

if __name__ == '__main__':
    main()
```

## E - Art Gallery on Graph 

### Solution 1:  undirected graph + max heap + visited array based on maximum remaining stamina

```py
import heapq

def main():
    n, m, k = map(int, input().split())
    adj_list = [[] for _ in range(n + 1)]
    for _ in range(m):
        u, v = map(int, input().split())
        adj_list[u].append(v)
        adj_list[v].append(u)
    max_heap = []
    vis = [-1] * (n + 1)
    for _ in range(k):
        p, h = map(int, input().split())
        max_heap.append((-h, -p))
        vis[p] = h
    heapq.heapify(max_heap)
    while max_heap:
        h, p = map(abs, heapq.heappop(max_heap))
        if vis[p] > h: continue
        for nei in adj_list[p]:
            nh = h - 1
            if vis[nei] >= nh: continue
            vis[nei] = nh
            heapq.heappush(max_heap, (-nh, -nei))
    print(sum(1 for v in vis if v >= 0))
    print(' '.join(map(str, [i for i in range(1, n + 1) if vis[i] >= 0])))

if __name__ == '__main__':
    main()
```

## F - Dungeon Explore 

### Solution 1:  dfs on hidden graph + takes at most 2*N visits + visit each node at most 2 times + stack + visited array + O(n^2)

```py
def main():
    n, m = map(int, input().split())
    vis = [0] * (n + 1)
    v = 1
    vis[1] = 1
    stack = [1]
    while True:
        inp = input()
        if inp == 'OK' or inp == '-1': break
        vertices = list(map(int, inp.split()))
        next_ = 0
        for u in reversed(vertices[1:]):
            if vis[u] == 0:
                next_ = u
                break
        if next_:
            v = next_
            stack.append(v)
            print(v, flush = True)
        else:
            stack.pop()
            v = stack[-1]
            print(v, flush = True)
        vis[v] = 1

if __name__ == '__main__':
    main()
```

## G - Banned Substrings 

### Solution 1: 

```py

```

# Atcoder Beginner Contest 306

## What is used at the top of each submission

```py
import os,sys
from io import BytesIO, IOBase
sys.setrecursionlimit(10**6)
from typing import *
# only use pypyjit when needed, it usese more memory, but speeds up recursion in pypy
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')
# sys.stdout = open('output.txt', 'w')

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
                    
if __name__ == '__main__':
    print(main())
    # main()
    # sys.stdout.close()
```

## A - Echo 

### Solution 1: 

```py
def main():
    n = int(input())
    s = input()
    res = []
    for ch in s:
        res.extend([ch] * 2)
    res = ''.join(res)
    print(res)

if __name__ == '__main__':
    main()
```

## B - Base 2 

### Solution 1: 

```py
def main():
    arr = list(map(int, input().split()))
    res = 0
    n = len(arr)
    for i in range(n):
        res += (arr[i] << i)
    print(res)

if __name__ == '__main__':
    main()
```

## C - Centers 

### Solution 1: 

```py
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    indices = [[] for _ in range(n + 1)]
    for i, x in enumerate(arr, start = 1):
        indices[x].append(i)
    res = sorted(range(1, n + 1), key = lambda x: indices[x][1])
    print(' '.join(map(str, res)))

if __name__ == '__main__':
    main()
```

## D - Poisonous Full-Course 

### Solution 1: 

```py
import math

def main():
    n = int(input())
    anti, pois = 0, 1
    arr = [None] * n
    for i in range(n):
        x, y = map(int, input().split())
        arr[i] = (x, y)
    dp = [[-math.inf] * 2 for _ in range(n + 1)]
    dp[0][0] = 0
    for i in range(n):
        x, y = arr[i]
        # skipping the course
        dp[i + 1][0] = dp[i][0]
        dp[i + 1][1] = dp[i][1]
        if x == anti:
            dp[i + 1][0] = max(dp[i + 1][0], dp[i][1] + y, dp[i][0] + y)
        else:
            dp[i + 1][1] = max(dp[i + 1][1], dp[i][0] + y)
    print(max(dp[-1]))

if __name__ == '__main__':
    main()
```

## E - Best Performances 

### Solution 1: 

```py
import heapq

def main():
    n, k, q = map(int, input().split())
    queries = [None] * q
    for i in range(q):
        x, y = map(int, input().split())
        x -= 1
        queries[i] = (x, y)
    arr = [0] * n # current value
    queried = [0] * n # number of times queried
    activity_level = [0] * n # if in k largest elements or not
    min_heap, max_heap = [], [(0, i, 0) for i in range(n)] # (value, index, query id)
    heapq.heapify(max_heap)
    sum_ = num_active = 0
    cnt = 0
    for i, y in queries:
        cnt += 1
        delta = y - arr[i]
        arr[i] = y
        queried[i] += 1
        if num_active < k:
            num_active += (activity_level[i] == 0)
            activity_level[i] = 1
            sum_ += delta
            heapq.heappush(min_heap, (arr[i], i, queried[i]))
        else:
            if activity_level[i] == 0:
                sum_ += arr[i]
            else:
                sum_ += delta
            heapq.heappush(min_heap, (arr[i], i, queried[i]))
            num_active += (activity_level[i] == 0)
            activity_level[i] = 1
            # balance them
            while min_heap and queried[min_heap[0][1]] != min_heap[0][2]:
                heapq.heappop(min_heap)
            while max_heap and queried[max_heap[0][1]] != max_heap[0][2]:
                heapq.heappop(max_heap)
            # swap
            if min_heap and max_heap and abs(max_heap[0][0]) > min_heap[0][0]:
                v1, i1, q1 = heapq.heappop(min_heap)
                v2, i2, q2 = heapq.heappop(max_heap)
                v2 = abs(v2)
                sum_ += v2 - v1
                activity_level[i1] = 0
                activity_level[i2] = 1
                heapq.heappush(max_heap, (-v1, i1, q1))
                heapq.heappush(min_heap, (v2, i2, q2))
            # balance them
            while min_heap and queried[min_heap[0][1]] != min_heap[0][2]:
                heapq.heappop(min_heap)
            while max_heap and queried[max_heap[0][1]] != max_heap[0][2]:
                heapq.heappop(max_heap)
            # remove one from active
            if num_active > k:
                v, i, q = heapq.heappop(min_heap)
                activity_level[i] = 0
                num_active -= 1
                sum_ -= v
                heapq.heappush(max_heap, (-v, i, q))
        print(sum_)

if __name__ == '__main__':
    main()
```

## F - Merge Sets 

### Solution 1: 

```py

```

## G - Return to 1 

### Solution 1: 

```py

```

# Atcoder Beginner Contest 307

## What is used at the top of each submission

```py
import os,sys
from io import BytesIO, IOBase
sys.setrecursionlimit(10**6)
from typing import *
# only use pypyjit when needed, it usese more memory, but speeds up recursion in pypy
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')
# sys.stdout = open('output.txt', 'w')

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
                    
if __name__ == '__main__':
    print(main())
    # main()
    # sys.stdout.close()
```

## A - Weekly Records 

### Solution 1: 

```py
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    for i in range(0, 7 * n, 7):
        res = sum(arr[i:i+7])
        print(res)

if __name__ == '__main__':
    main()
```

## B - racecar 

### Solution 1: 

```py
def main():
    n = int(input())
    words = [input() for _ in range(n)]
    is_palindrome = lambda s1, s2: s1 + s2 == (s1 + s2)[::-1]
    for i in range(n):
        for j in range(i + 1, n):
            if is_palindrome(words[i], words[j]) or is_palindrome(words[j], words[i]):
                return print("Yes")
    print("No")

if __name__ == '__main__':
    main()
```

## C - Ideal Sheet 

### Solution 1:  set + matrix + union of sets + normalize everything to first quadrant and cutout at that spot

The main reason I struggled with this problem is that I didn't realize that X must contain all the black squares from A and B. 

```py
from itertools import product

# i corresponds to the row, and also the position on the y axis
# j corresponds to the column, and also the position on the x axis
def convert(H, W, grid):
    s = set()
    for i, j in product(range(H), range(W)):
        if grid[i][j] == '#':
            s.add((i, j))
    return s

"""
convert everything to the first quadrant, and with minimim black squares on x = 0 and y = 0
"""
def normalize(s):
    min_x, min_y = min(x for y, x in s), min(y for y, x in s)
    return set((y - min_y, x - min_x) for y, x in s)

def main():
    n = int(input())
    HA, WA = map(int, input().split())
    A = normalize(convert(HA, WA, [input() for _ in range(HA)]))
    HB, WB = map(int, input().split())
    B = normalize(convert(HB, WB, [input() for _ in range(HB)]))
    HX, WX = map(int, input().split())
    X = normalize(convert(HX, WX, [input() for _ in range(HX)]))
    res = False
    for dx, dy in product(range(-HX, HX + 1), range(-WX, WX + 1)):
        res |= normalize(A.union(set((y + dy, x + dx) for y, x in B))) == X
    print('Yes' if res else 'No')

if __name__ == '__main__':
    main()
```

## D - Mismatched Parentheses 

### Solution 1:  2 stacks + result string stack and parentheses stack

```py
def main():
    n = int(input())
    s = input()
    result = []
    stack = []
    for ch in s:
        result.append(ch)
        if ch == '(':
            stack.append(ch)
        elif ch == ')' and stack:
            stack.pop()
            while result[-1] != '(':
                result.pop()
            result.pop()
    print(''.join(result))

if __name__ == '__main__':
    main()
```

## E - Distinct Adjacent 

### Solution 1:  dynamic programming + combinatorics + O(n) time + O(1) space

This is a tough one to solve, but you can derive a dynamic programming method.  Since the starting node is important, consider the starting node, you need to do this m times, but because of symmetry and each one is the same you just need to compute it for 1 tree, and multiply result by m at the end. 

So for one tree, you want to track the count of the number of nodes that contain the starting integer and the number of nodes that do not contain the starting integer. 

so if starting node is x1, then others is set of m - 1 other integers that are not x1 or in other words
suppose entire set is S, then it will be the others = S - {x1} from set difference.
then you will have a count for x1, and a count for others

There is a pattern to how these counts change for each level in the tree, it is rather simple though
so let x represent count of starting node that is fixed, and y represent count of the other nodes. 

initially it will be set x = 0, y = 1, because the first level you cannot have the adjacent tree. 

Then for each transition state. it will be that all the m - 1 integers that have count = y.  these will contribute to the fixed integer at the next level so x = (m - 1) * y
Then for the non fixed integers, they will all appear based on the sum of all integers - the count of y, so y = x + (m - 2) * y, which can be represented like this as well. 

Then add everything together as shown in image to get the final result and multiply m.

![images](images/distinct_adjacent_1.png)
![images](images/distinct_adjacent_2.png)

```py
def main():
    n, m = map(int, input().split())
    mod = 998244353
    x, y = 1, 0
    for _ in range(n - 2):
        nx = ((m - 1) * y) % mod
        ny = (x + (m - 2) * y) % mod 
        x, y = nx, ny
    res = (m * (x * (m - 1) + y * (m - 2) * (m - 1))) % mod
    print(res)

if __name__ == '__main__':
    main()
```

## F - Virus 2 

### Solution 1:  dijkstra + min heap

```py
from heapq import heappush, heappop

def main():
    n, m = map(int, input().split())
    adj_list = [[] for _ in range(n + 1)]
    for _ in range(m):
        u, v, w = map(int, input().split())
        adj_list[u].append((v, w))
        adj_list[v].append((u, w))
    K = int(input())
    start_nodes = map(int, input().split())
    D = int(input())
    dist = [0] + list(map(int, input().split()))
    res = [-1] * (n + 1)
    min_heap = []
    for node in start_nodes:
        res[node] = 0
        for nei, wei in adj_list[node]:
            heappush(min_heap, (wei, nei))
    def dfs(node, rem_dist):
        neighbors = []
        rem_heap = [(rem_dist, node)]
        while rem_heap:
            rem_dist, node = heappop(rem_heap)
            for nei, wei in adj_list[node]:
                if res[nei] != -1: continue
                if wei <= rem_dist:
                    res[nei] = day
                    heappush(rem_heap, (rem_dist - wei, nei))
                else:
                    neighbors.append((wei, nei))
        return neighbors
    for day in range(1, D + 1):
        tomorrow = []
        while min_heap and min_heap[0][0] <= dist[day]:
            wei, node = heappop(min_heap)
            if res[node] != -1: continue
            res[node] = day
            tomorrow.extend(dfs(node, dist[day] - wei))
        for wei, node in tomorrow:
            heappush(min_heap, (wei, node))
    print('\n'.join(map(str, res[1:])))

if __name__ == '__main__':
    main()
```

## G - Approximate Equalization 

### Solution 1:  dynamic programming + prefix sum + O(n^2) time

dp(i, j) = minimum number of operations to make the first j of the first i elements equal to v0 and the rest equal to v1. 

prefix sum can be used to calculate what the last value of the array must have been set to from previous elements. 

then can transition that last_val to v0 or v1, and update the number of operations required for that. 

```py
import math
from itertools import accumulate

def main():
    n = int(input())
    arr = list(map(int, input().split()))
    psum = [0] + list(accumulate(arr))
    sum_ = sum(arr)
    v0 = sum_ // n
    v1 = v0 + 1
    v0_terms = n - sum_ % n
    dp = [[math.inf] * (n + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    # j v0 terms
    for i in range(n):
        for j in range(i + 1):
            prev_psum = j * v0 + (i - j) * v1
            last_val = psum[i + 1] - prev_psum
            # add v0 term
            dp[i + 1][j + 1] = min(dp[i + 1][j + 1], dp[i][j] + abs(v0 - last_val))
            # add v1 term
            dp[i + 1][j] = min(dp[i + 1][j], dp[i][j] + abs(v1 - last_val))
    print(dp[-1][v0_terms])

if __name__ == '__main__':
    main()
```

# Atcoder Beginner Contest 308

## What is used at the top of each submission

```py
import os,sys
from io import BytesIO, IOBase
sys.setrecursionlimit(10**6)
from typing import *
# only use pypyjit when needed, it usese more memory, but speeds up recursion in pypy
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')
# sys.stdout = open('output.txt', 'w')

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
                    
if __name__ == '__main__':
    print(main())
    # main()
    # sys.stdout.close()
```

## A - New Scheme 

### Solution 1: all conditions

```py
def main():
    arr = list(map(int, input().split()))
    if all(x % 25 == 0 for x in arr) and all(100 <= x <= 675 for x in arr) and all(y >= x for x, y in zip(arr, arr[1:])):
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    main()
```

## B - Default Price 

### Solution 1: hash table

```py
def main():
    n, m = map(int, input().split())
    colors = input().split()
    D = input().split()
    P = list(map(int, input().split()))
    costs = {d: p for d, p in zip(D, P[1:])}
    res = sum(costs[col] if col in costs else P[0] for col in colors)
    print(res)

if __name__ == '__main__':
    main()
```

## C - Standings 

### Solution 1: sorting fractions + custom comparator + fraction class + math

Let us first design a function that compares the success rates of two people. Here, it is discouraged to compare h / (h + t)
using a floating-point number type, due to potential computational errors. In fact, some test cases are prepared to hack the solutions that compares values using std::double.

```py
# comparator for fractions
class Fraction:
    def __init__(self, num, denom):
        self.num, self.denom = num, denom
    
    def __lt__(self, other):
        return self.num * other.denom < other.num * self.denom

def main():
    n = int(input())
    heads, tails = [None] * n, [None] * n
    succ = [0] * n
    for i in range(n):
        h, t = map(int, input().split())
        heads[i], tails[i] = h, t
        succ[i] = Fraction(h, h + t)
    res = sorted(range(1, n + 1), key = lambda i: (succ[i - 1], -i), reverse = True)
    print(*res)

if __name__ == '__main__':
    main()
```

## D - Snuke Maze 

### Solution 1:  bfs + deque + grid + modulus

```py
from collections import deque

def main():
    H, W = map(int, input().split())
    S = [input() for _ in range(H)]
    vis = [[0] * W for _ in range(H)]
    neighborhood = lambda r, c: [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
    in_bounds = lambda r, c: 0 <= r < H and 0 <= c < W
    target = "snuke"
    index = 0
    queue = deque()
    if S[0][0] == target[index]:
        vis[0][0] = 1
        index += 1
        queue.append((0, 0))
    while queue:
        for _ in range(len(queue)):
            r, c = queue.popleft()
            if (r, c) == (H - 1, W - 1): return print("Yes")
            for nr, nc in neighborhood(r, c):
                if not in_bounds(nr, nc) or vis[nr][nc] or S[nr][nc] != target[index]: continue
                vis[nr][nc] = 1
                queue.append((nr, nc))
        index += 1
        index %= len(target)
    print("No")

if __name__ == '__main__':
    main()
```

## E - MEX 

### Solution 1:  dynamic programming + count

```py
from itertools import dropwhile

def main():
    n = int(input())
    arr = list(map(int, input().split()))
    s = input()
    mex = lambda x, y, z: next(dropwhile(lambda i: i in (x, y, z), range(4)))
    m = [0] * 3
    me = [[0] * 3 for _ in range(3)]
    res = 0
    for i in range(n):
        if s[i] == 'M':
            m[arr[i]] += 1
        elif s[i] == 'E':
            for j in range(3):
                r, c = min(j, arr[i]), max(j, arr[i])
                me[r][c] += m[j]
        else:
            for j in range(3):
                for k in range(j, 3):
                    res += mex(j, k, arr[i]) * me[j][k]
    print(res)

if __name__ == '__main__':
    main()
```

## F - Vouchers 

### Solution 1: max heap + sort + offline query

vouchers are added into the max heap when they become available for the current price and always take the best voucher cause it gives the most discount

```py
from heapq import heappush, heappop

def main():
    n, m = map(int, input().split())
    P = list(map(int, input().split()))
    L = list(map(int, input().split()))
    D = list(map(int, input().split()))
    queries = sorted([(l, d) for l, d in zip(L, D)])
    max_heap = []
    res = i = 0
    for p in sorted(P):
        res += p
        while i < m and queries[i][0] <= p:
            heappush(max_heap, -queries[i][1])
            i += 1
        if max_heap:
            res += heappop(max_heap)
    print(res)

if __name__ == '__main__':
    main()
```

## G - Minimum Xor Pair Query 



### Solution 1:  min heap + bit manipulation + sorted list + online query

For this problem can use the observation that the sorted integers, the minimum xor will come from the adjacent elements.  

proof in the image
![images](images/minimum_xor_pair.png)

Then the next step is to note we need to store the elements in a sorted list for the queries.  And also track the xor values in min heap to get the minimum xor.  Then there are some cases for when adding and deleting integer from array.  To keep the min heap updated to get minimum value. 

Case 1: sandwiched
delete y from x ^ y ^ z.  
In this case you must delete from min heap the xor values x ^ y and x ^ z. And you must add the xor value x ^ z.
add y to x ^ z
In this case you must add xor values x ^ y and x ^ z.  And you must delete the xor value x ^ z.
And also in both situations add or delete y from the sortedlist

Case 2: not sandwiched
This is must easier, just have to add x ^ y or delete x ^ y from the min heap.

```py
from collections import defaultdict
import heapq

from bisect import bisect_left as lower_bound
from bisect import bisect_right as upper_bound

class FenwickTree:
    def __init__(self, x):
        bit = self.bit = list(x)
        size = self.size = len(bit)
        for i in range(size):
            j = i | (i + 1)
            if j < size:
                bit[j] += bit[i]

    def update(self, idx, x):
        """updates bit[idx] += x"""
        while idx < self.size:
            self.bit[idx] += x
            idx |= idx + 1

    def __call__(self, end):
        """calc sum(bit[:end])"""
        x = 0
        while end:
            x += self.bit[end - 1]
            end &= end - 1
        return x

    def find_kth(self, k):
        """Find largest idx such that sum(bit[:idx]) <= k"""
        idx = -1
        for d in reversed(range(self.size.bit_length())):
            right_idx = idx + (1 << d)
            if right_idx < self.size and self.bit[right_idx] <= k:
                idx = right_idx
                k -= self.bit[idx]
        return idx + 1, k

class SortedList:
    block_size = 700

    def __init__(self, iterable=()):
        self.macro = []
        self.micros = [[]]
        self.micro_size = [0]
        self.fenwick = FenwickTree([0])
        self.size = 0
        for item in iterable:
            self.insert(item)

    def insert(self, x):
        i = lower_bound(self.macro, x)
        j = upper_bound(self.micros[i], x)
        self.micros[i].insert(j, x)
        self.size += 1
        self.micro_size[i] += 1
        self.fenwick.update(i, 1)
        if len(self.micros[i]) >= self.block_size:
            self.micros[i:i + 1] = self.micros[i][:self.block_size >> 1], self.micros[i][self.block_size >> 1:]
            self.micro_size[i:i + 1] = self.block_size >> 1, self.block_size >> 1
            self.fenwick = FenwickTree(self.micro_size)
            self.macro.insert(i, self.micros[i + 1][0])

    # requires index, so pop(i)
    def pop(self, k=-1):
        i, j = self._find_kth(k)
        self.size -= 1
        self.micro_size[i] -= 1
        self.fenwick.update(i, -1)
        return self.micros[i].pop(j)

    def __getitem__(self, k):
        i, j = self._find_kth(k)
        return self.micros[i][j]

    def count(self, x):
        return self.upper_bound(x) - self.lower_bound(x)

    def __contains__(self, x):
        return self.count(x) > 0

    def lower_bound(self, x):
        i = lower_bound(self.macro, x)
        return self.fenwick(i) + lower_bound(self.micros[i], x)

    def upper_bound(self, x):
        i = upper_bound(self.macro, x)
        return self.fenwick(i) + upper_bound(self.micros[i], x)

    def _find_kth(self, k):
        return self.fenwick.find_kth(k + self.size if k < 0 else k)

    def __len__(self):
        return self.size

    def __iter__(self):
        return (x for micro in self.micros for x in micro)

    def __repr__(self):
        return str(list(self))
    
class Heap:
    def __init__(self):
        self.heap = []
        self.deleted = defaultdict(int)
 
    def push(self, val):
        heapq.heappush(self.heap, val)
 
    def clean(self):
        while len(self.heap) > 0 and self.heap[0] in self.deleted:
            self.deleted[self.heap[0]] -= 1
            if self.deleted[self.heap[0]] == 0:
                del self.deleted[self.heap[0]]
            heapq.heappop(self.heap)
 
    def __len__(self):
        self.clean()
        return len(self.heap)
    
    def min(self):
        self.clean()
        return self.heap[0]
    
    def __repr__(self):
        return str(self.deleted)
    
    def delete(self, val):
        self.deleted[val] += 1
 
    def pop(self):
        self.clean()
        return heapq.heappop(self.heap)

def main():
    q = int(input())
    order = SortedList()
    min_heap = Heap()
    for _ in range(q):
        query = list(map(int, input().split()))
        if query[0] == 1:
            x = query[1]
            order.insert(x)
            i = order.lower_bound(x)
            cnt = xor = 0
            for j in [i - 1, i + 1]:
                if 0 <= j < len(order):
                    min_heap.push(order[i] ^ order[j])
                    cnt += 1
                    xor ^= order[j]
            if cnt == 2:
                min_heap.delete(xor)
        elif query[0] == 2:
            x = query[1]
            i = order.lower_bound(x)
            cnt = xor = 0
            for j in [i - 1, i + 1]:
                if 0 <= j < len(order):
                    cnt += 1
                    min_heap.delete(order[i] ^ order[j])
                    xor ^= order[j]
            order.pop(i)
            if cnt == 2:
                min_heap.push(xor)
        else:
            print(min_heap.min())

if __name__ == '__main__':
    main()
```

# Atcoder Beginner Contest 309

## What is used at the top of each submission

```py
import os,sys
from io import BytesIO, IOBase
sys.setrecursionlimit(10**6)
from typing import *
# only use pypyjit when needed, it usese more memory, but speeds up recursion in pypy
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')
# sys.stdout = open('output.txt', 'w')

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
                    
if __name__ == '__main__':
    print(main())
    # main()
    # sys.stdout.close()
```

## A - Nine

### Solution 1:  math + modulus

since a < b it turns out it only is going to be yes if a is indivisible by 3 and b == a + 1

```py
def main():
    a, b = map(int, input().split())
    res = "Yes" if a % 3 and b == a + 1 else "No"
    print(res)

if __name__ == '__main__':
    main()
```

## B - Rotate

### Solution 1:  deque + flatten outside border



```py
from collections import deque

def main():
    n = int(input())
    grid = [list(map(int, input().split())) for _ in range(n)]
    outside = deque()
    for c in range(n):
        outside.append(grid[0][c])
    for r in range(1, n):
        outside.append(grid[r][n-1])
    for c in range(n-2, -1, -1):
        outside.append(grid[n-1][c])
    for r in range(n-2, 0, -1):
        outside.append(grid[r][0])
    for c in range(1, n):
        grid[0][c] = outside.popleft()
    for r in range(1, n):
        grid[r][n-1] = outside.popleft()
    for c in range(n-2, -1, -1):
        grid[n-1][c] = outside.popleft()
    for r in range(n-2, -1, -1):
        grid[r][0] = outside.popleft()
    res = '\n'.join([' '.join(map(str, row)) for row in grid])
    print(res)

if __name__ == '__main__':
    main()
```

## C - Medicine

### Solution 1:  sort + two pointers

```py
def main():
    n, k = map(int, input().split())
    meds = [None] * n
    cur = 0
    for i in range(n):
        a, b = map(int, input().split())
        meds[i] = (a, b)
        cur += b
    meds.sort()
    i = 0
    day = 1
    while cur > k:
        day = meds[i][0]
        while i < n and meds[i][0] == day:
            cur -= meds[i][1]
            i += 1
    print(day)

if __name__ == '__main__':
    main()
```

## D - Add One Edge

### Solution 1: undirected graph + bfs + deque

```py
from collections import deque

def main():
    n1, n2, m = map(int, input().split())
    adj_list = [[] for _ in range(n1 + n2 + 1)]
    for _ in range(m):
        u, v = map(int, input().split())
        adj_list[u].append(v)
        adj_list[v].append(u)
    vis = [0] * (n1 + n2 + 1)
    def bfs(src):
        queue = deque([(src, 0)])
        vis[src] = 1
        while queue:
            node, dist = queue.popleft()
            for nei in adj_list[node]:
                if vis[nei]: continue
                vis[nei] = 1
                queue.append((nei, dist + 1))
        return dist
    res = bfs(1) + bfs(n1 + n2) + 1
    print(res)

if __name__ == '__main__':
    main()
```

## E - Family and Insurance

### Solution 1:  directed graph + tree + bfs

The trick is just keep the track of the remaining descendants that can be covered by the current insurance

```py
from collections import deque, Counter

def main():
    n, m  = map(int, input().split())
    parent = [0] * 2 + list(map(int, input().split()))
    queries = Counter()
    for i in range(m):
        x, y = map(int, input().split())
        queries[x] = max(queries[x], y + 1)
    adj_list = [[] for _ in range(n + 1)]
    for i in range(2, n + 1):
        adj_list[parent[i]].append(i)
    queue = deque([(1, queries[1])])
    vis = [0] * (n + 1)
    vis[1] = 1
    res = 0
    while queue:
        node, depth = queue.popleft()
        if depth > 0: res += 1
        for nei in adj_list[node]:
            if vis[nei]: continue
            vis[nei] = 1
            queue.append((nei, max(depth - 1, queries[nei])))
    print(res)

if __name__ == '__main__':
    main()
```

## F - Box in Box

### Solution 1:  sort + offline queries + minimum segment tree + coordinate compression

This is good segment tree problem you can sort It turns out that you can use any permutation of h, w, d with some rotation of a rectangular prism.  So then what you can do is consider the problem of sorting by h in ascending order, so that for j > i it is true that hj >= hi.  So smaller ones processed first, But since they can be equal you will want to sort wi in descending order.  Then what can be done is created a segment tree that will hold the minimum di for each wi.  Need to use coordinate compression on wi and segment tree.  Then can query a range from [0, wi - 1] with segment tree to get minimum di and if di < dj, it is already guaranteed wi < wj and that hi < hj.  so that is the solution.

```py
import math

class SegmentTree:
    def __init__(self, n: int, neutral: int, func):
        self.func = func
        self.neutral = neutral
        self.size = 1
        self.n = n
        while self.size<n:
            self.size*=2
        self.nodes = [neutral for _ in range(self.size*2)]

    def ascend(self, segment_idx: int) -> None:
        while segment_idx > 0:
            segment_idx -= 1
            segment_idx >>= 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.nodes[segment_idx] = self.func(self.nodes[left_segment_idx], self.nodes[right_segment_idx])
        
    def update(self, segment_idx: int, val: int) -> None:
        segment_idx += self.size - 1
        self.nodes[segment_idx] = self.func(self.nodes[segment_idx], val)
        self.ascend(segment_idx)
            
    def query(self, left: int, right: int) -> int:
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
    
    def __repr__(self) -> str:
        return f"nodes array: {self.nodes}, next array: {self.nodes}"

def main():
    n = int(input())
    boxes = [None] * n
    widths = set()
    compressed = {}
    for i in range(n):
        triplet = sorted(map(int, input().split()))
        boxes[i] = triplet
        widths.add(triplet[1])
    for i, w in enumerate(sorted(widths)):
        compressed[w] = i
    min_seg_tree = SegmentTree(len(widths), math.inf, min)
    # box[i] = (hi, wi, di)
    # sort by ascending order of h, and descending order in w
    boxes.sort(key = lambda box: (box[0], -box[1]))
    for _, w, d in boxes:
        left, right = 0, compressed[w]
        min_d = min_seg_tree.query(left, right)
        if min_d < d: return print("Yes")
        min_seg_tree.update(right, d)
    print("No")

if __name__ == '__main__':
    main()
```

## G - Ban Permutation

### Solution 1:  bitmask window + dynamic programming + inclusion exclusion principle + factorial

Instead of solving the problem for |Pi - i| >= X , try solving the problem with dynamic programming for |Pi - i| < X 
And then use inclusion exclusion principle to get the answer for |Pi - i| >= X.  

Inclusion Exclusion principle is needed for this because thereare overlapping, and you want to take the union of all the sets.  It is similar to derangement problem and I used that one to help understand why this works. 

![images](images/ban_permutation_1.png)
![images](images/ban_permutation_2.png)
![images](images/ban_permutation_3.png)
![images](images/ban_permutation_4.png)

```py
def factorials(n, mod):
    fact = [1]*(n + 1)
    for i in range(1, n + 1):
        fact[i] = (fact[i - 1] * i) % mod
    return fact

def main():
    N, X = map(int, input().split())
    mod = 998244353
    fact = factorials(N, mod)
    window_size = 2 * X - 1
    dp = [[0] * (1 << window_size)]
    dp[0][0] = 1
    for i in range(N):
        ndp = [[0] * (1 << window_size) for _ in range(i + 2)]
        for taken in range(i + 1):
            for mask in range(1 << window_size):
                if dp[taken][mask] == 0: continue
                for disp in range(-X + 1, X):
                    if i + disp < 0 or i + disp >= N: continue
                    window_index = disp + X - 1
                    if mask & (1 << window_index): continue
                    new_mask = (mask | (1 << window_index)) >> 1
                    # taking valid i where |P_i - i| < X
                    ndp[taken + 1][new_mask] += dp[taken][mask]
                # not taking an integer, but upgrading the mask for the window
                new_mask = mask >> 1
                ndp[taken][new_mask] += dp[taken][mask]
        dp = ndp
    res = 0
    for taken in range(N + 1):
        cur = 0
        for mask in range(1 << window_size):
            cur = (cur + dp[taken][mask]) % mod
        if taken & 1:
            res = (res - cur * fact[N - taken]) % mod
        else:
            res = (res + cur * fact[N - taken]) % mod
    print(res)
    
if __name__ == '__main__':
    main()
```

# Atcoder Beginner Contest 310

## What is used at the top of each submission

```py
import os,sys
from io import BytesIO, IOBase
sys.setrecursionlimit(10**6)
from typing import *
# only use pypyjit when needed, it usese more memory, but speeds up recursion in pypy
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')
# sys.stdout = open('output.txt', 'w')

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
                    
if __name__ == '__main__':
    print(main())
    # main()
    # sys.stdout.close()
```

## 

### Solution 1: 

```py
def main():
    n, p, q = map(int, input().split())
    d = list(map(int, input().split()))
    res = min(p, q + min(d))
    print(res)

if __name__ == '__main__':
    main()
```

## 

### Solution 1: 

```py
def main():
    n, m = map(int, input().split())
    prices = [None] * n
    functions = [set() for _ in range(n)]
    for i in range(n):
        arr = list(map(int, input().split()))
        prices[i] = arr[0]
        for j in range(2, len(arr)):
            functions[i].add(arr[j])
    indices = sorted(range(n), key = lambda i: prices[i])
    for idx in range(n):
        for jdx in range(idx + 1, n):
            i, j = indices[jdx], indices[idx]
            common = functions[i] & functions[j]
            if len(common) < len(functions[i]): continue
            if prices[i] == prices[j] and len(functions[j]) == len(common): continue
            return print('Yes')
    print('No')

if __name__ == '__main__':
    main()
```

## 

### Solution 1: 

```py
def main():
    n = int(input())
    vis = set()
    res = 0
    for _ in range(n):
        s = input()
        res += s not in vis
        vis.add(s)
        vis.add(s[::-1])
    print(res)

if __name__ == '__main__':
    main()
```

## 

### Solution 1: 

```py
from functools import lru_cache

def main():
    N, T, M = map(int, input().split())
    incompatible_masks = set()
    end_mask = (1 << N) - 1
    for _ in range(M):
        a, b = map(int, input().split())
        mask = (1 << a) | (1 << b)
        incompatible_masks.add(mask)    
    @lru_cache(None)
    def dfs(i, mask):
        if i == T: return mask == end_mask
        if mask == end_mask: return 0
        cnt = 0
        for team_mask in range(1, 1 << N):
            if team_mask & mask: continue
            cnt += dfs(i + 1, mask | team_mask)
        if mask == 7:
            print('i', i, 'mask', mask, 'cnt', cnt)
        return cnt
    res = dfs(0, 0)
    print(res)

if __name__ == '__main__':
    main()
```

## 

### Solution 1: 

```py
def main():
    n = int(input())
    arr = list(map(int, list(input())))
    ones = zeros = res = 0
    for num in arr:
        ones, zeros = zeros + (1 if num else ones), ones if num else 1
        res += ones
    return res

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

# Atcoder Beginner Contest 311

## What is used at the top of each submission

```py
import os,sys
from io import BytesIO, IOBase
sys.setrecursionlimit(10**6)
from typing import *
# only use pypyjit when needed, it usese more memory, but speeds up recursion in pypy
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')
# sys.stdout = open('output.txt', 'w')

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
                    
if __name__ == '__main__':
    print(main())
    # main()
    # sys.stdout.close()
```

## A - First ABC 

### Solution 1:  set

```py
def main():
    n = int(input())
    s = input()
    seen = set()
    for i, ch in enumerate(s):
        seen.add(ch)
        if len(seen) == 3: return print(i + 1)
    print(n)

if __name__ == '__main__':
    main()
```

## B - Vacation Together 

### Solution 1:  sliding window + any

```py
def main():
    n, d = map(int, input().split())
    arr = [list(input()) for _ in range(n)]
    res = window = 0
    for i in range(d):
        if any(person[i] == 'x' for person in arr):
            window = 0
        else:
            window += 1
        res = max(res, window)
    print(res)

if __name__ == '__main__':
    main()
```

## C - Find it! 

### Solution 1:  dfs + detect cycle in directed graph + backtrack to recreate directed cycle

```py
def main():
    n = int(input())
    adj_list = [[] for _ in range(n + 1)]
    parent = [0] + list(map(int, input().split()))
    for i in range(1, n + 1):
        adj_list[i].append(parent[i])
    visited = [0] * (n + 1)
    in_path = [0] * (n + 1)
    path = []
    def detect_cycle(node) -> bool:
        path.append(node)
        visited[node] = 1
        in_path[node] = 1
        for nei in adj_list[node]:
            if in_path[nei]: return nei
            if visited[nei]: continue
            res = detect_cycle(nei)
            if res: return res
        in_path[node] = 0
        path.pop()
        return 0
    for i in range(1, n + 1):
        if visited[i]: continue
        node = detect_cycle(i)
        cur = node
        cycle = [node]
        if cur:
            while path[-1] != node:
                cur = path.pop()
                cycle.append(cur)
            print(len(cycle))
            print(*cycle[::-1])
            return
    print(-1)

if __name__ == '__main__':
    main()
```

## D - Grid Ice Floor 

### Solution 1:  bfs + modified neighborhood

```py
from collections import deque

def main():
    R, C = map(int, input().split())
    grid = [list(input()) for _ in range(R)]
    vis = set()
    vis2 = set([(1, 1)])
    queue = deque([(1, 1)])
    def neighborhood(r, c):
        for dr, dc in ((-1, 0), (0, -1), (1, 0), (0, 1)):
            nr, nc = r, c
            while grid[nr][nc] == '.':
                vis.add((nr, nc))
                nr += dr
                nc += dc
            yield nr - dr, nc - dc
    while queue:
        r, c = queue.popleft()
        for nr, nc in neighborhood(r, c):
            if (nr, nc) in vis2: continue
            vis2.add((nr, nc))
            queue.append((nr, nc))
    print(len(vis))

if __name__ == '__main__':
    main()
```

## E - Defect-free Squares 

### Solution 1:  dynamic programming + size of squares

```py
from itertools import product

def main():
    R, C, N = map(int, input().split())
    dp = [[1] * C for _ in range(R)]
    for _ in range(N):
        r, c = map(int, input().split())
        dp[r - 1][c - 1] = 0
    for r, c in product(range(R), range(C)):
        if r == 0 or c == 0: continue
        if dp[r][c] == 0: continue
        dp[r][c] = min(dp[r - 1][c], dp[r][c - 1], dp[r - 1][c - 1]) + 1
    res = sum(sum(row) for row in dp)
    print(res)

if __name__ == '__main__':
    main()
```

## F - Yet Another Grid Task 

### Solution 1: 

```py

```

## G - One More Grid Task 

### Solution 1: 

```py

```

# Atcoder Beginner Contest 312

## What is used at the top of each submission

```py
import os,sys
from io import BytesIO, IOBase
sys.setrecursionlimit(10**6)
from typing import *
# only use pypyjit when needed, it usese more memory, but speeds up recursion in pypy
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')
# sys.stdout = open('output.txt', 'w')

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
                    
if __name__ == '__main__':
    print(main())
    # main()
    # sys.stdout.close()
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
from itertools import product

def main():
    n = int(input())
    m = 101
    voxels = [[[-1] * m for _ in range(m)] for _ in range(m)]
    for i in range(n):
        x1, y1, z1, x2, y2, z2 = map(int, input().split())
        for x, y, z in product(range(x1, x2), range(y1, y2), range(z1, z2)):
            voxels[x][y][z] = i
    res = [set() for _ in range(n)]
    for x, y, z in product(range(m), repeat = 3):
        if voxels[x][y][z] == -1: continue
        i = voxels[x][y][z]
        n1, n2, n3 = voxels[x + 1][y][z], voxels[x][y + 1][z], voxels[x][y][z + 1]
        if n1 != -1 and n1 != i: 
            res[i].add(n1)
            res[n1].add(i)
        if n2 != -1 and n2 != i:
            res[i].add(n2)
            res[n2].add(i)
        if n3 != -1 and n3 != i:
            res[i].add(n3)
            res[n3].add(i)
    print('\n'.join(map(str, map(len, res))))

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

# Atcoder Beginner Contest 314

## A - 3.14

### Solution 1:  string slice

```py
def main():
    pi = "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"
    n = int(input())
    print(pi[:n + 2])
 
if __name__ == '__main__':
    main()
```

## B - Roulette

### Solution 1:  set 

```py
def main():
    n = int(input())
    bets = [set() for _ in range(n)]
    for i in range(n):
        c = int(input())
        b = map(int, input().split())
        bets[i].update(b)
    x = int(input())
    min_bets = 40
    betters = []
    for i in range(n):
        if x in bets[i] and len(bets[i]) < min_bets:
            min_bets = len(bets[i])
            betters = [i + 1]
        elif x in bets[i] and len(bets[i]) == min_bets: betters.append(i + 1)
    print(len(betters))
    print(*betters)
 
if __name__ == '__main__':
    main()
```

## C - Rotate Colored Subsequence

### Solution 1:  simulation + equivalence class

For each equivalence class add all the indices in that equivalence class and the nloop over the equivalence classes and rotate within them.  Cause you can completely identify one from the index of the characters in that class or color

```py
def main():
    n, m = map(int, input().split())
    s = input()
    colors = list(map(int, input().split()))
    res = [None] * n
    indices = [[] for _ in range(m + 1)]
    for i in range(n):
        indices[colors[i]].append(i)
    for i in range(1, m + 1):
        for j in range(len(indices[i])):
            res[indices[i][j]] = s[indices[i][j - 1]]
    print(''.join(res))
 
if __name__ == '__main__':
    main()
```

## D - LOWER

### Solution 1:  greedy + trick + set to upper or lower case and only perform necessary queries

```py
def main():
    n = int(input())
    s = input()
    q = int(input())
    lower = upper = False
    start = -1
    queries = []
    res = list(s)
    for i in range(q):
        t, x, c = input().split()
        t = int(t)
        x = int(x) - 1
        if t == 1: 
            res[x] = c
            queries.append((i, x, c))
        else:
            lower = True if t == 2 else False
            upper = False if t == 2 else True
            start = i
    if lower:
        res = list(map(lambda x: x.lower(), res))
    if upper:
        res = list(map(lambda x: x.upper(), res))
    for i, x, c in queries:
        if i <= start: continue
        res[x] = c
    print("".join(res))
    
if __name__ == '__main__':
    main()
```

## E - Roulettes

### Solution 1:  expected value + expected amount + contribution

```py

```

## F - A Certain Game

### Solution 1:  disjoint set union + directed rooted tree + arborescence + expected value + inverse modular

The expected value is either win or lose so the value is 1 or 0, so it turns out it will just be the sum of probabilities for a player and each time his team wins.

creating a directed graph that is also like a directed rooted tree, or a out-tree, also called an arborescence

![image](images/a_certain_game.png)

```py
class UnionFind:
    def __init__(self, n: int):
        self.size = [1]*n
        self.parent = list(range(n))
    
    def find(self,i: int) -> int:
        while i != self.parent[i]:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    """
    returns true if the nodes were not union prior. 
    """
    def union(self,i: int,j: int) -> bool:
        i, j = self.find(i), self.find(j)
        if i!=j:
            if self.size[i] < self.size[j]:
                i,j=j,i
            self.parent[j] = i
            self.size[i] += self.size[j]
            return True
        return False
    
    def __repr__(self) -> str:
        return f'parents: {[(i, parent) for i, parent in enumerate(self.parent)]}, sizes: {self.size}'
    
def mod_inverse(num, mod):
    return pow(num, mod - 2, mod)

from collections import defaultdict

def main():
    mod = 998244353
    n = int(input())
    dsu = UnionFind(n + 1)
    res = [0] * (n + 1)
    root = n + 1
    adj_list = defaultdict(list)
    nodes = {i: i for i in range(1, n + 1)}
    for _ in range(n - 1):
        p, q = map(int, input().split())
        root += 1
        sz_p, sz_q = dsu.size[dsu.find(p)], dsu.size[dsu.find(q)]
        sz = sz_p + sz_q
        inv = mod_inverse(sz, mod)
        adj_list[root].append((nodes[dsu.find(p)], sz_p * inv))
        adj_list[root].append((nodes[dsu.find(q)], sz_q * inv))
        dsu.union(p, q)
        nodes[dsu.find(p)] = root
    # dfs that computes the sum of probabilities going down each root
    stk = [(root, 0)]
    while stk:
        node, pr = stk.pop()
        if node <= n:
            res[node] = pr
        for child, wei in adj_list[node]:
            stk.append((child, (pr + wei) % mod))
    print(*res[1:])

if __name__ == '__main__':
    main()
```

## G - Amulets

### Solution 1: 

```py

```



# Atcoder Beginner Contest 315

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

### Solution 1:  dynamic programming

Observe that there will be at most 2^20 penalty, cause anything worse than that results in a penalty that is greater than the cost of just including any checkpoint.

Suppose all checkpoints are the farthest distance from each other which is 20,000.  The two numbers somewhat converge when you suppose you skip 20 checkpoints with worse case which is going to give you 400,000, and 2^20 penalty is always going to be worse than actually including all the checkpoints, cause 2^20 = 1,048,576.

Solve this equation $20,000 * x = 2^{x}$

As you can see by the graph of the equation the point at which $2^{x}$ becomes larger than the cost of including shortcuts is just before x = 19.  So picking x = 20 should work.  Once the line is in the positive y values it means the penalty of skipping the checkpoints is incurring more cost then just not skipping the checkpoints.

You can think of this function as the cost of skipping checkpoints.  x = number of checkpoints skipped
$f(x) = 2^{x} - 20,000 * x$, when $f(x) < 0$ that means reducing the total cost.  When $f(x) > 0$ that means it is increasing the cost.


![iamges](images/shortcuts_solution_graph.png)

![images](images/shortcuts.png)

```py

```

## 

### Solution 1: 

```py

```



# Atcoder Beginner Contest 317

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



# Atcoder Beginner Contest 318

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



# Atcoder Beginner Contest 319

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



# Atcoder Beginner Contest 320

## A - Leyland Number

### Solution 1:  math

```py
def main():
    a, b = map(int, input().split())
    print(a**b + b**a)

if __name__ == '__main__':
    main()
```

## B - Longest Palindrome

### Solution 1:  is palindrome + two pointers

```py
def main():
    s = input()
    n = len(s)
    res = 1
    def is_palindrome(part):
        left, right = 0, len(part) - 1
        while left < right and part[left] == part[right]:
            left += 1
            right -= 1
        return left >= right
    for i in range(n):
        for j in range(i + 1, n + 1):
            cur = s[i : j]
            if is_palindrome(cur):
                res = max(res, len(cur))
    print(res)

if __name__ == '__main__':
    main()
```

## C - Slot Strategy 2 (Easy)

### Solution 1:  brute force

```py
import math
from itertools import product

def main():
    m = int(input())
    s1, s2, s3 = list(map(int, input())), list(map(int, input())), list(map(int, input()))
    res = math.inf
    for dig in range(10):
        vals = [[] for _ in range(3)]
        for i in range(3 * m):
            if s1[i % m] == dig and len(vals[0]) < 3:
                vals[0].append(i)
            if s2[i % m] == dig and len(vals[1]) < 3:
                vals[1].append(i)
            if s3[i % m] == dig and len(vals[2]) < 3:
                vals[2].append(i)
        if not len(vals[0]) == len(vals[1]) == len(vals[2]) == 3: continue
        for i, j, k in product(range(3), repeat = 3):
            nums = set([vals[0][i], vals[1][j], vals[2][k]])
            if len(nums) < 3: continue
            res = min(res, max(nums))
    print(res if res != math.inf else -1)

if __name__ == '__main__':
    main()
```

## D - Relative Position

### Solution 1:  weighted undirected graph + dfs

```py
def main():
    n, m = map(int, input().split())
    pos = [None] * n
    pos[0] = (0, 0)
    adj_list = [[] for _ in range(n)]
    for _ in range(m):
        u, v, x, y = map(int, input().split())
        u -= 1
        v -= 1
        adj_list[u].append((v, x, y))
        adj_list[v].append((u, -x, -y))
    stack = [0]
    while stack:
        u = stack.pop()
        x, y = pos[u]
        for v, dx, dy in adj_list[u]:
            if pos[v] is not None: continue
            nx, ny = x + dx, y + dy
            pos[v] = (nx, ny)
            stack.append(v)
    for i in range(n):
        if pos[i] is None:
            print("undecidable")
        else:
            print(*pos[i])
        
if __name__ == '__main__':
    main()
```

## E - Somen Nagashi

### Solution 1:  heap + greedy

```py
from heapq import heappush, heappop, heapify

def main():
    n, m = map(int, input().split())
    people = list(range(n))
    heapify(people)
    res = [0] * n
    free = []
    for _ in range(m):
        t, w, s = map(int, input().split())
        while free and free[0][0] <= t:
            _, u = heappop(free)
            heappush(people, u)
        if not people: continue
        u = heappop(people)
        res[u] += w
        heappush(free, (t + s, u))
    for i in range(n):
        print(res[i])

if __name__ == '__main__':
    main()
```

## F - Fuel Round Trip

### Solution 1:  dynamic programming

```py
import math

def main():
    N, H = map(int, input().split())
    pos = list(map(int, input().split()))
    stations = [None] * (N - 1)
    for i in range(N - 1):
        p, f = map(int, input().split())
        stations[i] = (p, f)
    dp = [[math.inf] * (H + 1) for _ in range(H + 1)] # j, k
    if pos[0] > H: return print(-1)
    for k in range(pos[0], H + 1):
        dp[H - pos[0]][k] = 0
    for i in range(N - 1):
        ndp = [[math.inf] * (H + 1) for _ in range(H + 1)]
        d = pos[i + 1] - pos[i]
        for j in range(H + 1):
            for k in range(H + 1):
                if dp[j][k] == math.inf: continue
                p, f = stations[i]
                # refuel none
                if k + d <= H and j - d >= 0:
                    nj, nk = j - d, k + d
                    ndp[nj][nk] = min(ndp[nj][nk], dp[j][k])
                # refuel j
                if k + d <= H:
                    nj, nk = min(j + f, H) - d, k + d
                    if nj >= 0:
                        ndp[nj][nk] = min(ndp[nj][nk], dp[j][k] + p)
                # refuel k
                if j - d >= 0 and k >= f and k - f + d <= H:
                    nj, nk = j - d, k - f + d
                    ndp[nj][nk] = min(ndp[nj][nk], dp[j][k] + p)
                    if k == H:
                        for w in range(nk + 1, H + 1):
                            ndp[nj][w] = min(ndp[nj][w], dp[j][k] + p)
        dp = ndp
    res = min(dp[i][i] for i in range(H + 1))
    print(res if res != math.inf else -1)

if __name__ == '__main__':
    main()
```

## G - Slot Strategy 2 (Hard)

### Solution 1:  graph matching + bipartite graph matching + graph theory

```py

```



# Atcoder Beginner Contest 321

## A - 321-like Checker

### Solution 1: 

```py
def main():
    n = list(map(int, input()))
    res = "Yes" if all(n[i] < n[i - 1] for i in range(1, len(n))) else 'No'
    print(res)

if __name__ == '__main__':
    main()
```

## B - Cutoff

### Solution 1: 

```py
def main():
    n, k = map(int, input().split())
    arr = sorted(map(int, input().split()))
    sum_ = sum(arr[1:-1])
    needed = k - sum_
    if needed <= arr[0]: return print(0)
    if needed > arr[-1]: return print(-1)
    print(needed)

if __name__ == '__main__':
    main()
```

## C - 321-like Searcher

### Solution 1: 

```py
def main():
    k = int(input())
    arr = []
    for mask in range(1, 1 << 10):
        val = 0
        for i in reversed(range(10)):
            if (mask >> i) & 1:
                val = val * 10 + i
        arr.append(val)
    arr.sort()
    print(arr[k])

if __name__ == '__main__':
    main()
```

## D - Set Menu

### Solution 1: 

```py
import bisect

def main():
    n, m, p = map(int, input().split())
    A = list(map(int, input().split()))
    B = sorted(map(int, input().split()))
    psum = [0] * (m + 1)
    for i in range(m):
        psum[i + 1] = psum[i] + B[i]
    res = 0
    for a in A:
        i = bisect.bisect_right(B, p - a)
        res += i * a + psum[i] + (m - i) * p
    print(res)

if __name__ == '__main__':
    main()
```

## E - Complete Binary Tree

### Solution 1: 

```py
def main():
    t = int(input())
    for _ in range(t):
        n, x, k = map(int, input().split())
        print("n, x, k", n, x, k)
        res = 0
        # first tree
        u = x
        rem = k
        while (u << 1) <= n and rem > 0:
            u <<= 1
            rem -= 1
        if rem == 0:
            res += min(n, u + pow(2, k) - 1) - u + 1
        prev_even = x % 2 == 0
        k -= 1
        if k == 0: 
            res += 1
        k -= 1
        x >>= 1
        while x > 0 and k >= 0:
            u = 2 * x + prev_even
            print("start u", u)
            rem = k
            while (u << 1) <= n and rem > 0:
                u <<= 1
                rem -= 1
            print("u", u, "rem", rem)
            if rem == 0:
                res += min(n, u + pow(2, k) - 1) - u + 1
            prev_even = x % 2 == 0
            print("res", res, "k", k, "x", x)
            x >>= 1
            k -= 1

        print(res)
        

if __name__ == '__main__':
    main()
```

## F - #(subset sum = K) with Add and Erase

### Solution 1:  knapsack + dynamic programming + remove element from knapsack + reverse knapsack

```py
def main():
    mod = 998244353
    q, k = map(int, input().split())
    dp = [0] * (k + 1)
    dp[0] = 1
    for _ in range(q):
        t, x = input().split()
        x = int(x)
        if t == "+":
            for i in range(k, x - 1, -1):
                dp[i] += dp[i - x] % mod
                dp[i] %= mod
        else:
            for i in range(x, k + 1):
                dp[i] -= dp[i - x] % mod
                dp[i] %= mod
        print(dp[-1])

if __name__ == '__main__':
    main()
```

## G - Electric Circuit

### Solution 1: 

```py

```



# Atcoder Beginner Contest 322

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



# Atcoder Beginner Contest 324

## B - 3-smooth Numbers 

### Solution 1: 

```py
def main():
    N = int(input())
    for x in range(64):
        a = pow(2, x)
        if a > N: continue
        for y in range(40):
            b = pow(3, y)
            if b > N: continue
            if a * b > N: continue
            if a * b == N: return print("Yes")
    print("No")

if __name__ == '__main__':
    main()
```

## C - Error Correction 

### Solution 1: 

```py
def main():
    N, T = input().split()
    N = int(N)
    works = [0] * N
    for i in range(N):
        S = input()
        if len(S) == len(T): # changed 0 or 1 character in string
            hamming_dist = sum(1 for x, y in zip(S, T) if x != y)
            works[i] = int(hamming_dist <= 1)
        elif len(S) == len(T) + 1: # inserted 1 character
            j = 0
            for ch in S:
                if j < len(T) and ch == T[j]:
                    j += 1
            works[i] = int(j == len(T))
        elif len(S) == len(T) - 1: # deleted 1 character
            j = 0
            for ch in T:
                if j < len(S) and ch == S[j]:
                    j += 1
            works[i] = int(j == len(S))
    K = sum(works)
    print(K)
    print(" ".join(map(str, [i + 1 for i in range(N) if works[i]])))

if __name__ == '__main__':
    main()
```

## D - Square Permutation 

### Solution 1: 

precompute all the squares in less than 10^7 operations
The frequency of the digits in the squares should be the same as that in S if it is a permutation
However you can ignore 0s. 

```py
def main():
    N = int(input())
    S = sorted(input())
    M = round(10 ** (N / 2))
    res = 0
    for i in range(M + 1):
        res += S == sorted(str(i * i).zfill(N))
    print(res)

if __name__ == '__main__':
    main()
```

## E - Joint Two Strings 

### Solution 1: binary search, two pointers

```py
def main():
    N, T = input().split()
    N = int(N)
    arr = [None] * N
    pindex = [0] * N
    for i in range(N):
        arr[i] = input()
        j = 0
        for ch in arr[i]:
            if j == len(T): break
            if ch == T[j]: j += 1
        pindex[i] = j
    pindex.sort(reverse = True)
    def bsearch(target):
        left, right = 0, N
        while left < right:
            mid = (left + right) >> 1
            if pindex[mid] >= target:
                left = mid + 1
            else:
                right = mid
        return left
    res = 0
    for i in range(N):
        j = 0
        for ch in reversed(arr[i]):
            if j == len(T): break
            if ch == T[len(T) - j - 1]: j += 1
        k = bsearch(len(T) - j)
        res += k
    print(res)

if __name__ == '__main__':
    main()
```

## F - Beautiful Path 

### Solution 1: 

```py

```

## G - Generate Arrays 

### Solution 1: 

```py

```



# Atcoder Beginner Contest 326

## C - Peak 

### Solution 1: 

```py
from collections import deque

def main():
    N, M = map(int, input().split())
    arr = sorted(map(int, input().split()))
    queue = deque()
    res = 0
    i = 0
    while i < N:
        cur = arr[i]
        while i < N and arr[i] == cur:
            i += 1
            queue.append(cur)
        while queue[0] <= queue[-1] - M:
            queue.popleft()
        res = max(res, len(queue))
    print(res)

if __name__ == '__main__':
    main()
```

## D - ABC Puzzle 

### Solution 1:  

```py

```

## E - Revenge of "The Salary of AtCoder Inc." 

### Solution 1:  probability, uniform, expectation value, cumulative sum

probability of finding ith index after x = j where j is 0 <= j < i, but the summation of all those probabilitys multiplied by 1/3.  including if x = 0 which is p_0 = 1

![image](images/salary_at_atcoder_expectation_value_plot.png)

```py
mod = 998244353

def mod_inverse(v):
    return pow(v, mod - 2, mod)

def main():
    N = int(input())
    arr = list(map(int, input().split()))
    res = 0
    psum = 1
    for num in arr:
        cur = (psum * mod_inverse(N)) % mod
        res = (res + num * cur) % mod
        psum = (psum + cur) % mod
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



# Atcoder Beginner Contest 327

## D - Good Tuple Problem 

### Solution 1: 2 coloring, bipartite graph, iterative dfs with stack, undirected graph, even length cycles

if a graph has odd length cycle it cannot be bipartite

```py
def main():
    N, M = map(int, input().split())
    adj = [[] for _ in range(N)]
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    for u, v in zip(A, B):
        u -= 1
        v -= 1
        adj[u].append(v)
        adj[v].append(u)
    color = [-1] * N
    for i in range(N):
        if color[i] != -1: continue
        stack = [i]
        color[i] = 0
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if color[v] == -1:
                    color[v] = color[u] ^ 1
                    stack.append(v)
                elif color[v] == color[u]:
                    return print("No")
    print("Yes")

if __name__ == '__main__':
    main()
```

## E - Maximize Rating

### Solution 1: dynamic programming, math

```py

```

## F - Apples

### Solution 1: segment tree, lazy segment tree, line sweep

lattice points in 2D space.  Use segment tree because it is asking for range addition updates and range maximum queries or can just use lazy segment tree.

```py
import math 

class SegmentTree:
    def __init__(self, n: int, neutral: int, initial: int):
        self.mod = int(1e9) + 7
        self.neutral = neutral
        self.size = 1
        self.n = n
        self.initial_val = initial
        while self.size<n:
            self.size*=2
        self.operations = [initial for _ in range(self.size*2)]
        self.values = [neutral for _ in range(self.size*2)]
        self.build()
 
    def build(self):
        for segment_idx in range(self.n):
            segment_idx += self.size - 1
            self.values[segment_idx]  = self.initial_val
            self.ascend(segment_idx)
 
    def modify_op(self, x: int, y: int) -> int:
        return x + y
    
    def calc_op(self, x: int, y: int) -> int:
        return max(x, y)
 
    def ascend(self, segment_idx: int) -> None:
        while segment_idx > 0:
            segment_idx -= 1
            segment_idx >>= 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.values[segment_idx] = self.calc_op(self.values[left_segment_idx], self.values[right_segment_idx])
            self.values[segment_idx] = self.modify_op(self.values[segment_idx], self.operations[segment_idx])
        
    def update(self, left: int, right: int, val: int) -> None:
        stack = [(0, self.size, 0)]
        segments = []
        while stack:
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # COMPLETE OVERLAP
            if segment_left_bound >= left and segment_right_bound <= right:
                self.operations[segment_idx] = self.modify_op(self.operations[segment_idx], val)
                self.values[segment_idx] = self.modify_op(self.values[segment_idx], val)
                segments.append(segment_idx)
                continue
            # PARTIAL OVERLAP
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
        for segment_idx in segments:
            self.ascend(segment_idx)
            
    def query(self, left: int, right: int) -> int:
        stack = [(0, self.size, 0, self.initial_val)]
        result = self.neutral
        while stack:
            # BOUNDS FOR CURRENT INTERVAL and idx for tree
            segment_left_bound, segment_right_bound, segment_idx, operation_val = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # COMPLETE OVERLAP
            if segment_left_bound >= left and segment_right_bound <= right:
                modified_val = self.modify_op(self.values[segment_idx], operation_val)
                result = self.calc_op(result, modified_val)
                continue
            # PARTIAL OVERLAP
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            operation_val = self.modify_op(operation_val, self.operations[segment_idx])
            stack.extend([(mid_point, segment_right_bound, right_segment_idx, operation_val), (segment_left_bound, mid_point, left_segment_idx, operation_val)])
        return result
    
    def __repr__(self) -> str:
        return f"operations array: {self.operations}, values array: {self.values}"

def main():
    M = 200_001
    N, D, W = map(int, input().split())
    queries = [[] for _ in range(M)]
    # line sweep construction
    for _ in range(N):
        t, x = map(int, input().split())
        queries[max(0, t - D)].append((x, 1))
        queries[t].append((x, -1))
    seg = SegmentTree(M, -math.inf, 0)
    res = 0
    for t in range(M):
        for x, delta in queries[t]:
            seg.update(max(0, x - W), x, delta) # range addition update
        # range max query
        res = max(res, seg.query(0, M))
    print(res)

if __name__ == '__main__':
    main()
```

```py
class LazySegmentTree:
    def __init__(self, n: int, neutral: int, noop: int):
        self.mod = int(1e9) + 7
        self.neutral = neutral
        self.size = 1
        self.noop = noop
        self.n = n 
        while self.size<n:
            self.size*=2
        self.operations = [noop for _ in range(self.size*2)]
        self.values = [neutral for _ in range(self.size*2)]

    def is_leaf_node(self, segment_right_bound, segment_left_bound) -> bool:
        return segment_right_bound - segment_left_bound == 1

    def propagate(self, segment_idx: int, segment_left_bound: int, segment_right_bound: int) -> None:
        if self.is_leaf_node(segment_right_bound, segment_left_bound) or self.operations[segment_idx] == self.noop: return
        left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
        self.operations[left_segment_idx] = self.modify_op(self.operations[left_segment_idx], self.operations[segment_idx])
        self.operations[right_segment_idx] = self.modify_op(self.operations[right_segment_idx], self.operations[segment_idx])
        self.values[left_segment_idx] = self.modify_op(self.values[left_segment_idx], self.operations[segment_idx])
        self.values[right_segment_idx] = self.modify_op(self.values[right_segment_idx], self.operations[segment_idx])
        self.operations[segment_idx] = self.noop
 
    def modify_op(self, x: int, y: int) -> int:
        return x + y
    
    def calc_op(self, x: int, y: int) -> int:
        return max(x, y)
 
    def ascend(self, segment_idx: int) -> None:
        while segment_idx > 0:
            segment_idx -= 1
            segment_idx >>= 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.values[segment_idx] = self.calc_op(self.values[left_segment_idx], self.values[right_segment_idx])
            self.values[segment_idx] = self.modify_op(self.values[segment_idx], self.operations[segment_idx])
        
    def update(self, left: int, right: int, val: int) -> None:
        stack = [(0, self.size, 0)]
        segments = []
        while stack:
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # COMPLETE OVERLAP
            if segment_left_bound >= left and segment_right_bound <= right:
                self.operations[segment_idx] = self.modify_op(self.operations[segment_idx], val)
                self.values[segment_idx] = self.modify_op(self.values[segment_idx], val)
                segments.append(segment_idx)
                continue
            # PARTIAL OVERLAP
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.propagate(segment_idx, segment_left_bound, segment_right_bound)
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
        for segment_idx in segments:
            self.ascend(segment_idx)
            
    def query(self, left: int, right: int) -> int:
        stack = [(0, self.size, 0)]
        result = self.neutral
        while stack:
            # BOUNDS FOR CURRENT INTERVAL and idx for tree
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # COMPLETE OVERLAP
            if segment_left_bound >= left and segment_right_bound <= right:
                result = self.calc_op(result, self.values[segment_idx])
                continue
            # PARTIAL OVERLAP
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.propagate(segment_idx, segment_left_bound, segment_right_bound)
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
        return result
    
    def __repr__(self) -> str:
        return f"operations array: {self.operations}, values array: {self.values}"

def main():
    M = 200_001
    N, D, W = map(int, input().split())
    queries = [[] for _ in range(M)]
    # line sweep construction
    for _ in range(N):
        t, x = map(int, input().split())
        queries[max(0, t - D)].append((x, 1))
        queries[t].append((x, -1))
    seg = LazySegmentTree(M, 0, 0)
    res = 0
    for t in range(M):
        for x, delta in queries[t]:
            seg.update(max(0, x - W), x, delta) # range addition update
        # range max query
        res = max(res, seg.query(0, M))
    print(res)

if __name__ == '__main__':
    main()
```

This is the solution that passes, the other segment trees are apparently too slow. 

```py
# https://qiita.com/ether2420/items/7b67b2b35ad5f441d686
def segfunc(x,y):
    return max(x, y)
class LazySegTree_RAQ:
    def __init__(self,init_val,segfunc,ide_ele):
        n = len(init_val)
        self.segfunc = segfunc
        self.ide_ele = ide_ele
        self.num = 1<<(n-1).bit_length()
        self.tree = [ide_ele]*2*self.num
        self.lazy = [0]*2*self.num
        for i in range(n):
            self.tree[self.num+i] = init_val[i]
        for i in range(self.num-1,0,-1):
            self.tree[i] = self.segfunc(self.tree[2*i], self.tree[2*i+1])
    def gindex(self,l,r):
        l += self.num
        r += self.num
        lm = l>>(l&-l).bit_length()
        rm = r>>(r&-r).bit_length()
        while r>l:
            if l<=lm:
                yield l
            if r<=rm:
                yield r
            r >>= 1
            l >>= 1
        while l:
            yield l
            l >>= 1
    def propagates(self,*ids):
        for i in reversed(ids):
            v = self.lazy[i]
            if v==0:
                continue
            self.lazy[i] = 0
            self.lazy[2*i] += v
            self.lazy[2*i+1] += v
            self.tree[2*i] += v
            self.tree[2*i+1] += v
    def add(self,l,r,x):
        ids = self.gindex(l,r)
        l += self.num
        r += self.num
        while l<r:
            if l&1:
                self.lazy[l] += x
                self.tree[l] += x
                l += 1
            if r&1:
                self.lazy[r-1] += x
                self.tree[r-1] += x
            r >>= 1
            l >>= 1
        for i in ids:
            self.tree[i] = self.segfunc(self.tree[2*i], self.tree[2*i+1]) + self.lazy[i]
    def query(self,l,r):
        self.propagates(*self.gindex(l,r))
        res = self.ide_ele
        l += self.num
        r += self.num
        while l<r:
            if l&1:
                res = self.segfunc(res,self.tree[l])
                l += 1
            if r&1:
                res = self.segfunc(res,self.tree[r-1])
            l >>= 1
            r >>= 1
        return res

def main():
    M = 200_001
    N, D, W = map(int, input().split())
    queries = [[] for _ in range(M)]
    # line sweep construction
    for _ in range(N):
        t, x = map(int, input().split())
        queries[max(0, t - D)].append((x, 1))
        queries[t].append((x, -1))
    seg = LazySegTree_RAQ([0] * M, segfunc, 0)
    res = 0
    for t in range(M):
        for x, delta in queries[t]:
            seg.add(max(0, x - W), x, delta) # range addition update
        # range max query
        res = max(res, seg.query(0, M))
    print(res)

if __name__ == '__main__':
    main()
```

## G - Many Good Tuple Problems

### Solution 1: counting

```py

```


# Atcoder Beginner Contest 329

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



# Atcoder Beginner Contest 330

## E - Mex and Update 

### Solution 1:  frequency table, segment tree, mex queries in segment tree, range sum segment tree, PURQ segment tree

-1 means remove from segment tree
+1 means add to segment tree

![image](../images/segment_tree_mex.png)

```py
from collections import Counter

class SegmentTree:
    def __init__(self, n: int, neutral: int, func):
        self.func = func
        self.neutral = neutral
        self.size = 1
        self.n = n
        while self.size<n:
            self.size*=2
        self.nodes = [neutral for _ in range(self.size*2)]
    def ascend(self, segment_idx: int) -> None:
        while segment_idx > 0:
            segment_idx -= 1
            segment_idx >>= 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.nodes[segment_idx] = self.func(self.nodes[left_segment_idx], self.nodes[right_segment_idx]) 
    def update(self, segment_idx: int, val: int) -> None:
        segment_idx += self.size - 1
        self.nodes[segment_idx] += val
        self.ascend(segment_idx)
    def query_mex(self) -> int:
        segment_left_bound, segment_right_bound, segment_idx = 0, self.size, 0
        while segment_left_bound + 1 < segment_right_bound:
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2 * segment_idx + 1, 2 * segment_idx + 2
            child_segment_len = (segment_right_bound - segment_left_bound) >> 1
            if self.nodes[left_segment_idx] < child_segment_len:
                segment_idx = left_segment_idx
                segment_right_bound = mid_point
            else:
                segment_idx = right_segment_idx
                segment_left_bound = mid_point
        return segment_left_bound
    def __repr__(self) -> str:
        return f"nodes array: {self.nodes}, next array: {self.nodes}"
    
MAXN = 2 * 10**5

def main():
    N, Q = map(int, input().split())
    arr = list(map(int, input().split()))
    freq = Counter()
    summation = lambda x, y: x + y
    seg = SegmentTree(MAXN + 1, 0, summation)  
    for num in arr:
        freq[num] += 1
        if freq[num] == 1 and num <= MAXN:
            seg.update(num, 1)
    for _ in range(Q):
        i, x = map(int, input().split())
        i -= 1
        freq[arr[i]] -= 1
        if freq[arr[i]] == 0 and arr[i] <= MAXN:
            seg.update(arr[i], -1)
        arr[i] = x
        freq[arr[i]] += 1
        if freq[arr[i]] == 1 and arr[i] <= MAXN:
            seg.update(arr[i], 1)
        print(seg.query_mex())

if __name__ == '__main__':
    main()
```

# Atcoder Beginner Contest 331

## D. Tile Pattern

### Solution 1:  2D prefix sum + periodicity

```py
from itertools import product

def main():
    N, Q = map(int, input().split())
    grid = [list(input()) for _ in range(N)]
    psum = [[0] * (N + 1) for _ in range(N + 1)]
    for r, c in product(range(1, N + 1), repeat = 2):
        psum[r][c] = psum[r - 1][c] + psum[r][c - 1] - psum[r - 1][c - 1] + (grid[r - 1][c - 1] == "B")
    def g(r, c):
        r_span, c_span = r // N, c // N
        return (
            psum[N][N] * r_span * c_span
            + psum[N][c % N] * r_span
            + psum[r % N][N] * c_span
            + psum[r % N][c % N]
        )
    def f(r1, c1, r2, c2):
        return g(r2, c2) - g(r1, c2) - g(r2, c1) + g(r1, c1)
    for _ in range(Q):
        r1, c1, r2, c2 = map(int, input().split())
        print(f(r1, c1, r2 + 1, c2 + 1))

if __name__ == '__main__':
    main()
```

## E - Set Meal 

### Solution 1:  hash map, sort, offline query

```py
def main():
    N, M, L = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    sides = sorted(range(M), key = lambda i: B[i], reverse = True)
    bad_combos = [set() for _ in range(N)]
    for _ in range(L):
        c, d = map(int, input().split())
        c -= 1
        d -= 1
        bad_combos[c].add(d)
    ans = 0
    for i in range(N):
        for j in sides:
            if j not in bad_combos[i]:
                ans = max(ans, A[i] + B[j])
                break
    print(ans)
    
if __name__ == '__main__':
    main()
```

## F - Palindrome Query 

### Solution 1:  rolling hash on segment tree

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

if __name__ == '__main__':
    main()
```

```cpp

```

## G - Collect Them All 

### Solution 1:  fast fourier transform, dynamic programming, combinatorics, absorbing markov chains

```py
from collections import deque
from itertools import product

class FFT:
    """
    https://github.com/shakayami/ACL-for-python/blob/master/convolution.py
    """
    def primitive_root_constexpr(self, m):
        if m == 2: return 1
        if m == 167772161: return 3
        if m == 469762049: return 3
        if m == 754974721: return 11
        if m == 998244353: return 3
        divs = [0] * 20
        divs[0] = 2
        x = (m - 1) // 2
        while x % 2 == 0: x //= 2
        i = 3
        cnt = 1
        while i * i <= x:
            if x % i == 0:
                divs[cnt] = i
                cnt += 1 
                while x % i == 0: x //= i
            i += 2
        if x > 1:
            divs[cnt] = x
            cnt += 1
        g = 2
        while 1:
            ok = True
            for i in range(cnt):
                if pow(g, (m - 1) // divs[i], m) == 1: 
                    ok = False
                    break
            if ok: return g
            g += 1
    # bit scan forward, finds the rightmost set bit? maybe? 
    def bsf(self, x):
        res = 0
        while x % 2 == 0:
            res += 1
            x //= 2
        return res
    rank2 = 0
    root = []
    iroot = []
    rate2 = []
    irate2 = []
    rate3 = []
    irate3 = []
    def __init__(self, MOD):
        self.mod = MOD
        self.g = self.primitive_root_constexpr(self.mod)
        self.rank2 = self.bsf(self.mod - 1)
        self.root = [0] * (self.rank2 + 1)
        self.iroot = [0] * (self.rank2 + 1)
        self.rate2 = [0] * self.rank2
        self.irate2 = [0] * self.rank2
        self.rate3 = [0] * (self.rank2 - 1)
        self.irate3 = [0] * (self.rank2 - 1)
        self.root[self.rank2] = pow(self.g, (self.mod - 1) >> self.rank2, self.mod)
        self.iroot[self.rank2] = pow(self.root[self.rank2], self.mod - 2, self.mod)
        for i in range(self.rank2 - 1, -1, -1):
            self.root[i] = (self.root[i + 1] ** 2) % self.mod
            self.iroot[i] = (self.iroot[i + 1] ** 2) % self.mod
        prod = iprod = 1
        for i in range(self.rank2 - 1):
            self.rate2[i] = (self.root[i + 2] * prod) % self.mod
            self.irate2[i] = (self.iroot[i + 2] * iprod) % self.mod
            prod = (prod * self.iroot[i + 2]) % self.mod
            iprod = (iprod * self.root[i + 2]) % self.mod
        prod = iprod = 1
        for i in range(self.rank2 - 2):
            self.rate3[i] = (self.root[i + 3] * prod) % self.mod
            self.irate3[i] = (self.iroot[i + 3] * iprod) % self.mod
            prod = (prod * self.iroot[i + 3]) % self.mod
            iprod = (iprod * self.root[i + 3]) % self.mod
    def butterfly(self, a):
        n = len(a)
        h = (n - 1).bit_length()
        LEN = 0
        while LEN < h:
            if h - LEN == 1:
                p = 1 << (h - LEN - 1)
                rot = 1
                for s in range(1 << LEN):
                    offset = s << (h - LEN)
                    for i in range(p):
                        l = a[i + offset]
                        r = a[i + offset + p] * rot
                        a[i + offset] = (l + r) % self.mod
                        a[i + offset + p] = (l - r) % self.mod
                    rot *= self.rate2[(~s & -~s).bit_length() - 1]
                    rot %= self.mod
                LEN += 1
            else:
                p = 1 << (h - LEN - 2)
                rot = 1
                imag = self.root[2]
                for s in range(1 << LEN):
                    rot2 = (rot * rot) % self.mod
                    rot3 = (rot2 * rot) % self.mod 
                    offset = s << (h - LEN)
                    for i in range(p):
                        a0 = a[i + offset]
                        a1 = a[i + offset + p] * rot
                        a2 = a[i + offset + 2 * p] * rot2
                        a3 = a[i + offset + 3 * p] * rot3
                        a1na3imag = (a1 - a3) % self.mod * imag
                        a[i + offset] = (a0 + a1 + a2 + a3) % self.mod
                        a[i + offset + p] = (a0 + a2 - a1 - a3) % self.mod
                        a[i + offset + 2 * p] = (a0 - a2 + a1na3imag) % self.mod
                        a[i + offset + 3 * p] = (a0 - a2 - a1na3imag) % self.mod
                    rot *= self.rate3[(~s & -~s).bit_length() - 1]
                    rot %= self.mod
                LEN += 2
    def butterfly_inv(self, a):
        n = len(a)
        h = (n - 1).bit_length()
        LEN = h
        while LEN:
            if LEN == 1:
                p = 1 << (h - LEN)
                irot = 1
                for s in range(1 << (LEN - 1)):
                    offset = s << (h - LEN + 1)
                    for i in range(p):
                        l = a[i + offset]
                        r = a[i + offset + p]
                        a[i + offset] = (l + r) % self.mod
                        a[i + offset + p] = (l - r) * irot % self.mod
                    irot *= self.irate2[(~s & -~s).bit_length() - 1]
                    irot %= self.mod
                LEN -= 1
            else:
                p = 1 << (h - LEN)
                irot = 1
                iimag = self.iroot[2]
                for s in range(1 << (LEN - 2)):
                    irot2 = (irot * irot) % self.mod
                    irot3 = (irot * irot2) % self.mod
                    offset = s << (h - LEN + 2)
                    for i in range(p):
                        a0 = a[i + offset]
                        a1 = a[i + offset + p]
                        a2 = a[i + offset + 2 * p]
                        a3 = a[i + offset + 3 * p]
                        a2na3iimag = (a2 - a3) * iimag % self.mod
                        a[i + offset] = (a0 + a1 + a2 + a3) % self.mod
                        a[i + offset + p] = (a0 - a1 + a2na3iimag) * irot % self.mod
                        a[i + offset + 2 * p] = (a0 + a1 - a2 - a3) * irot2 % self.mod
                        a[i + offset + 3 * p] = (a0 - a1 - a2na3iimag) * irot3 % self.mod
                    irot *= self.irate3[(~s & -~s).bit_length() - 1]
                    irot %= self.mod
                LEN -= 2
    def convolution(self, a, b):
        n = len(a)
        m = len(b)
        if not (a) or not (b): return []
        if min(n, m) <= 40: # naive solution
            res = [0] * (n + m - 1)
            for i, j in product(range(n), range(m)):
                res[i + j] += a[i] * b[j]
                res[i + j] %= self.mod
            return res
        z = 1 << (n + m - 2).bit_length()
        a = a + [0] * (z - n)
        b = b + [0] * (z - m)
        self.butterfly(a)
        self.butterfly(b)
        c = [(a[i] * b[i]) % self.mod for i in range(z)]
        self.butterfly_inv(c)
        iz = pow(z, self.mod - 2, self.mod)
        for i in range(n + m - 1):
            c[i] = (c[i] * iz) % self.mod
        return c[: n + m - 1]
    
def mod_inverse(x):
    return pow(x, mod - 2, mod) % mod

mod = 998244353
def main():
    N, M = map(int, input().split())
    C = list(map(int, input().split()))
    fft = FFT(mod)
    queue = deque()
    for c in C:
        X = [0] * (c + 1)
        X[0] = 1
        X[c] = -1
        queue.append(X)
    while len(queue) > 1:
        a = queue.popleft()
        b = queue.popleft()
        queue.append(fft.convolution(a, b))
    ans = 0
    for k in range(N):
        ans += N * mod_inverse(N - k) * queue[0][k] % mod
        ans %= mod
    if M % 2 == 0:
        ans *= -1
        ans %= mod
    print(ans)
if __name__ == '__main__':
    main()
```

# Atcoder Beginner Contest 332

## D - Swapping Puzzle 

### Solution 1:  enumerate permutations, selection sort

```py
from itertools import permutations, product
import math

def main():
    R, C = map(int, input().split())
    A = [list(map(int, input().split())) for _ in range(R)]
    B = [list(map(int, input().split())) for _ in range(R)]
    res = math.inf
    def check(rows, cols):
        for (i, r), (j, c) in product(enumerate(rows), enumerate(cols)):
            if A[i][j] != B[r][c]:
                return False
        return True
    def swaps(N, arr):
        res = 0
        vals = list(range(N))
        for v in vals:
            for i, x in enumerate(arr):
                if x == v: break
            while x < i:
                arr[i], arr[i - 1] = arr[i - 1], arr[i]
                res += 1
                i -= 1
        return res
    for rows, cols in product(permutations(range(R)), permutations(range(C))):
        if check(rows, cols):
            res = min(res, swaps(R, list(rows)) + swaps(C, list(cols)))
    res = res if res != math.inf else -1
    print(res)
    
if __name__ == '__main__':
    main()
```

## E - Lucky bag 

### Solution 1:  dynamic programming with bitmasks, enumerating submasks

```py
import math

def main():
    N, D = map(int, input().split())
    arr = list(map(int, input().split()))
    avg = sum(arr) / D
    dp = [[math.inf] * (1 << N) for _ in range(D + 1)]
    def bag(mask):
        weight = sum(arr[i] for i in range(N) if (mask >> i) & 1)
        return (weight - avg) ** 2
    # base case is for every possible set of items in one bag dp[1][mask]
    for mask in range(1 << N): 
        dp[1][mask] = bag(mask)
    for i in range(2, D + 1): # i bags
        for mask in range(1, 1 << N): # set of items taken in i bags
            submask = mask
            dp[i][mask] = dp[i - 1][mask] + dp[1][0] # take no items into new bag
            while submask > 0:
                submask = (submask - 1) & mask
                dp[i][mask] = min(dp[i][mask], dp[i - 1][submask] + dp[1][mask ^ submask])
    ans = dp[-1][-1] / D
    print(ans)

if __name__ == '__main__':
    main()
```

## F - Random Update Query 

### Solution 1: lazy segment tree

```py
class lazy_segtree():
    def update(self,k):self.d[k]=self.op(self.d[2*k],self.d[2*k+1])
    def all_apply(self,k,f):
        self.d[k]=self.mapping(f,self.d[k])
        if (k<self.size):self.lz[k]=self.composition(f,self.lz[k])
    def push(self,k):
        self.all_apply(2*k,self.lz[k])
        self.all_apply(2*k+1,self.lz[k])
        self.lz[k]=self.identity
    def __init__(self,V,OP,E,MAPPING,COMPOSITION,ID):
        self.n=len(V)
        self.log=(self.n-1).bit_length()
        self.size=1<<self.log
        self.d=[E for i in range(2*self.size)]
        self.lz=[ID for i in range(self.size)]
        self.e=E
        self.op=OP
        self.mapping=MAPPING
        self.composition=COMPOSITION
        self.identity=ID
        for i in range(self.n):self.d[self.size+i]=V[i]
        for i in range(self.size-1,0,-1):self.update(i)
    def set(self,p,x):
        assert 0<=p and p<self.n
        p+=self.size
        for i in range(self.log,0,-1):self.push(p>>i)
        self.d[p]=x
        for i in range(1,self.log+1):self.update(p>>i)
    def get(self,p):
        assert 0<=p and p<self.n
        p+=self.size
        for i in range(self.log,0,-1):self.push(p>>i)
        return self.d[p]
    def prod(self,l,r):
        assert 0<=l and l<=r and r<=self.n
        if l==r:return self.e
        l+=self.size
        r+=self.size
        for i in range(self.log,0,-1):
            if (((l>>i)<<i)!=l):self.push(l>>i)
            if (((r>>i)<<i)!=r):self.push(r>>i)
        sml,smr=self.e,self.e
        while(l<r):
            if l&1:
                sml=self.op(sml,self.d[l])
                l+=1
            if r&1:
                r-=1
                smr=self.op(self.d[r],smr)
            l>>=1
            r>>=1
        return self.op(sml,smr)
    def all_prod(self):return self.d[1]
    def apply_point(self,p,f):
        assert 0<=p and p<self.n
        p+=self.size
        for i in range(self.log,0,-1):self.push(p>>i)
        self.d[p]=self.mapping(f,self.d[p])
        for i in range(1,self.log+1):self.update(p>>i)
    def apply(self,l,r,f):
        assert 0<=l and l<=r and r<=self.n
        if l==r:return
        l+=self.size
        r+=self.size
        for i in range(self.log,0,-1):
            if (((l>>i)<<i)!=l):self.push(l>>i)
            if (((r>>i)<<i)!=r):self.push((r-1)>>i)
        l2,r2=l,r
        while(l<r):
            if (l&1):
                self.all_apply(l,f)
                l+=1
            if (r&1):
                r-=1
                self.all_apply(r,f)
            l>>=1
            r>>=1
        l,r=l2,r2
        for i in range(1,self.log+1):
            if (((l>>i)<<i)!=l):self.update(l>>i)
            if (((r>>i)<<i)!=r):self.update((r-1)>>i)
    def max_right(self,l,g):
        assert 0<=l and l<=self.n
        assert g(self.e)
        if l==self.n:return self.n
        l+=self.size
        for i in range(self.log,0,-1):self.push(l>>i)
        sm=self.e
        while(1):
            while(l%2==0):l>>=1
            if not(g(self.op(sm,self.d[l]))):
                while(l<self.size):
                    self.push(l)
                    l=(2*l)
                    if (g(self.op(sm,self.d[l]))):
                        sm=self.op(sm,self.d[l])
                        l+=1
                return l-self.size
            sm=self.op(sm,self.d[l])
            l+=1
            if (l&-l)==l:break
        return self.n
    def min_left(self,r,g):
        assert (0<=r and r<=self.n)
        assert g(self.e)
        if r==0:return 0
        r+=self.size
        for i in range(self.log,0,-1):self.push((r-1)>>i)
        sm=self.e
        while(1):
            r-=1
            while(r>1 and (r%2)):r>>=1
            if not(g(self.op(self.d[r],sm))):
                while(r<self.size):
                    self.push(r)
                    r=(2*r+1)
                    if g(self.op(self.d[r],sm)):
                        sm=self.op(self.d[r],sm)
                        r-=1
                return r+1-self.size
            sm=self.op(self.d[r],sm)
            if (r&-r)==r:break
        return 0

mod = 998244353

# I don't get this? 
def op(a, b):
    return ((a[0] + b[0]) % mod, (a[1] + b[1]) % mod)
# maps F, S to S
def mapping(f, x):
    return ((f[0] * x[0] + x[1] * f[1]) % mod, x[1])
# composition of F, F to F
def composition(f, g):
    return ((f[0] * g[0]) % mod, (g[1] * f[0] + f[1]) % mod)
# identity element for op (0, 0)
# identity element for mapping (1, 0)

def mod_inverse(v):
    return pow(v, mod - 2, mod)

def main():
    N, M = map(int, input().split())
    arr = list(map(int, input().split()))
    seg = lazy_segtree([(x, 1) for x in arr], op, (0, 0), mapping, composition, (1, 0))
    for _ in range(M):
        left, right, x = map(int, input().split())
        left -= 1
        delta = right - left
        a = (1 - mod_inverse(delta)) % mod
        b = (x * mod_inverse(delta)) % mod
        seg.apply(left, right, (a, b))
    print(*[seg.get(i)[0] for i in range(N)])

if __name__ == '__main__':
    main()
```

## G - Not Too Many Balls 

### Solution 1:  max flow min cut theorem and dynamic programming

```py
from collections import defaultdict

def main():
    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    max_k = N * (N + 1) // 2
    INF = 1 << 60
    dp = [INF] * (max_k + 1)
    dp[0] = 0
    for i, a in enumerate(A, start = 1):
        available_k = i * (i - 1) // 2
        for k in range(available_k, -1, -1):
            dp[k + i] = min(dp[k + i], dp[k] + a)
    over_bs = defaultdict(list)
    for j, b in enumerate(B, start = 1):
        max_k2 = b // j
        over_bs[max_k2].append(j)
    j_sum = M * (M + 1) // 2
    b_sum = 0
    ans = INF
    for k2 in range(max_k + 1):
        ans = min(ans, dp[max_k - k2] + k2 * j_sum + b_sum)
        if k2 in over_bs:
            for j in over_bs[k2]:
                j_sum -= j
                b_sum += B[j - 1]
    print(ans)

if __name__ == '__main__':
    main()
```


# Atcoder Beginner Contest 333

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

### Solution 1:  Stern Brocot tree, farey tree, binary search, gcd to compute irriducible fraction

This is a solution that has worst case time complexity of O(p + q) I believe,  And it happens when you have something like 1/q,  a really large number, it will take q + 1 iterations to get the result for this one. 

Since N can be 10^10 this will time out.

```py
import math

def main():
    r = input()
    N = int(input())
    p, q = int(r[2:]), 10 ** (len(r) - 2)
    # convert to irreducible fraction
    g = math.gcd(p, q)
    while g != 1:
        p //= g
        q //= g
        g = math.gcd(p, q)
    if p <= N and q <= N: return print(p, q)
    # find the closest fraction
    a, b = 0, 1
    c, d = 1, 1
    while a + c <= N and b + d <= N:
        pm, qm = a + c, b + d
        if p * qm < q * pm:
            c, d = pm, qm
        else:
            a, b = pm, qm
    if b * q * (c * q - p * d) >= d * q * (p * b - a * q): print(a, b)
    else: print(c, d)

if __name__ == '__main__':
    main()
```

The logarithmic solution
binary search to find mediants


```py
import math

def main():
    r = input()
    N = int(input())
    p, q = int(r[2:]), 10 ** (len(r) - 2)
    # convert to irreducible fraction
    g = math.gcd(p, q)
    while g != 1:
        p //= g
        q //= g
        g = math.gcd(p, q)
    if p <= N and q <= N: return print(p, q)
    # find the closest fraction
    a, b = 0, 1
    c, d = 1, 1
    while a + c <= N and b + d <= N:
        pm, qm = a + c, b + d
        if p * qm < q * pm:
            left, right = 0, N
            while left < right:
                mid = (left + right + 1) >> 1
                if pm + mid * a > N or qm + mid * b > N: right = mid - 1; continue
                if q * (pm + mid * a) <= p * (qm + mid * b):
                    right = mid - 1
                else:
                    left = mid
            c = pm + left * a
            d = qm + left * b
        else:
            left, right = 0, N
            while left < right:
                mid = (left + right + 1) >> 1
                if pm + mid * c > N or qm + mid * d > N: right = mid - 1; continue
                if q * (pm + mid * c) <= p * (qm + mid * d):
                    left = mid
                else:
                    right = mid - 1
            a = pm + left * c
            b = qm + left * d
    if b * q * (c * q - p * d) >= d * q * (p * b - a * q): print(a, b)
    else: print(c, d)

if __name__ == '__main__':
    main()
```



# Atcoder Beginner Contest 334

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



# Atcoder Beginner Contest 335

## C - Loong Tracking

### Solution 1:  stack

```py
dirs = {
    "U": (0, 1),
    "D": (0, -1),
    "R": (1, 0),
    "L": (-1, 0)
}

def main():
    N, Q = map(int, input().split())
    body = [(i, 0) for i in reversed(range(1, N + 1))]
    for _ in range(Q):
        t, v = input().split()
        if t == "1":
            dr, dc = dirs[v]
            r, c = body[-1]
            body.append((r + dr, c + dc))
        else:
            print(*body[-int(v)])

if __name__ == '__main__':
    main()
```

## D - Loong and Takahashi

### Solution 1:  spiral, grid

```py
def main():
    N = int(input())
    grid = [[0] * N for _ in range(N)]
    grid[N // 2][N // 2] = "T"
    v = 1
    for i in range(N // 2):
        # top row 
        for c in range(N):
            if grid[i][c] != 0: continue
            grid[i][c] = v
            v += 1
        # right column
        for r in range(N):
            if grid[r][N - i - 1] != 0: continue
            grid[r][N - i - 1] = v
            v += 1
        # bottom row
        for c in reversed(range(N)):
            if grid[N - i - 1][c] != 0: continue 
            grid[N - i - 1][c] = v 
            v += 1
        # left column
        for r in reversed(range(N)):
            if grid[r][i] != 0: continue
            grid[r][i] = v 
            v += 1
    for row in grid:
        print(*row)

if __name__ == '__main__':
    main()
```

## E - Non-Decreasing Colorful Path

### Solution 1:   priority queue, undirected graph

```py
from heapq import heappop, heappush

def main():
    N, M = map(int, input().split())
    arr = list(map(int, input().split()))
    adj = [[] for _ in range(N)]
    for _ in range(M):
        u, v = map(int, input().split())
        u -= 1; v -= 1
        adj[u].append(v)
        adj[v].append(u)
    vis = [0] * N
    min_heap = [(arr[0], -1, 0)]
    while min_heap:
        _, score, u = heappop(min_heap)
        score = -score
        if u == N - 1: return print(score)
        if vis[u]: continue
        vis[u] = 1
        for v in adj[u]:
            if arr[v] < arr[u]: continue
            nscore = score + (1 if arr[v] > arr[u] else 0)
            heappush(min_heap, (arr[v], -nscore, v))
    print(0)

if __name__ == '__main__':
    main()
```

## F - Hop Sugoroku

### Solution 1:  DP, square root 

```py
mod = 998244353
def main():
    N = int(input())
    arr = list(map(int, input().split()))
    dp = [0] * N
    deltas = [0] * N
    dp[0] = 1
    for i in range(N):
        x = (dp[i] + deltas[i]) % mod
        for j in range(i + arr[i], N, arr[i]):
            dp[j] = (dp[j] + x) % mod
            if arr[i] == arr[j]:
                deltas[j] = (deltas[j] + x) % mod
                break
    print(sum(dp) % mod)

if __name__ == '__main__':
    main()
```

## G - Discrete Logarithm Problems 

### Solution 1:  Number theory, modular arithmetic, fermat's little theorem, gcd, primitive roots, multiplicative order and additive order

```py
import math

# collect all the prime factors
def prime_factors(num: int) -> List[int]:
    factors = []
    while num % 2 == 0:
        if not factors or (factors and factors[-1] != 2):
            factors.append(2)
        num //= 2
    for i in range(3, math.isqrt(num) + 1, 2):
        while num % i == 0:
            if not factors or (factors and factors[-1] != i):
                factors.append(i)
            num //= i
    if num > 2:
        factors.append(num)
    return factors

def main():
    N, P = map(int, input().split())
    pf = prime_factors(P - 1)
    arr = list(map(int, input().split()))
    mult_orders = [P - 1] * N
    for i in range(N):
        while mult_orders[i] > 1:
            found = False
            for pfactor in pf:
                if pfactor > mult_orders[i] or mult_orders[i] % pfactor: continue
                x = mult_orders[i] // pfactor
                if pow(arr[i], x, P) == 1:
                    mult_orders[i] = x
                    found = True
                    break
            if not found: break
    mult_pairs = Counter()
    for i in range(N):
        mult_pairs[mult_orders[i]] += 1
    mult_pairs = list(mult_pairs.items())
    ans = 0
    for i in range(len(mult_pairs)):
        k1, v1 = mult_pairs[i]
        for j in range(i, len(mult_pairs)):
            k2, v2 = mult_pairs[j]
            if k1 % k2 == 0 or k2 % k1 == 0:
                ans += v1 * v2
    print(ans)

if __name__ == '__main__':
    main()
```


# Atcoder Beginner Contest 336

## C - Even Digits 

### Solution 1:  Convert to base 5

```py
def main():
    N = int(input()) - 1
    if N == 0: return print(N % 5)
    values = [0, 2, 4, 6, 8]
    base_five = []
    while N > 0:
        base_five.append(N % 5)
        N //= 5
    ans = []
    for v in reversed(base_five):
        ans.append(values[v])
    print("".join(map(str, ans)))

if __name__ == '__main__':
    main()
```

## D - Pyramid 

### Solution 1:  dynamic programming 

```py
def main():
    N = int(input())
    arr = list(map(int, input().split()))
    dp = [0] * (N + 1)
    for i in reversed(range(N)):
        dp[i] = min(arr[i], dp[i + 1] + 1)
    plen = ans = 0
    for i in range(N):
        plen = min(arr[i], plen + 1)
        ans = max(ans, min(plen, dp[i]))
    print(ans)
if __name__ == '__main__':
    main()
```

## E - Digit Sum Divisible 

### Solution 1:  digit dp, digit sums

```py
# for each fixed digit sum
# (index, digit sum, remainder modulos fixed digit sum, tight)

def main():
    N = input()
    ans = 0
    for ds in range(1, 9 * 14 + 1):
        dp = Counter({(0, 0, 1): 1})
        for d in map(int, N):
            ndp = Counter()
            for (dig_sum, rem, tight), cnt in dp.items():
                for dig in range(10 if not tight else d + 1):
                    ndig_sum, nrem, ntight = dig_sum + dig, (rem * 10 + dig) % ds, tight and dig == d
                    if ndig_sum > ds: break
                    ndp[(ndig_sum, nrem, ntight)] += cnt
            dp = ndp
        ans += dp[(ds, 0, 0)] + dp[(ds, 0, 1)]
    print(ans)

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



# Atcoder Beginner Contest 338

## C - Leftover Recipes 

### Solution 1:  math, division

```py
import math

def main():
    N = int(input())
    Q = list(map(int, input().split()))
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    cnt_a = ans = 0
    while True:
        cnt_b = math.inf
        for i in range(N):
            if B[i] == 0: continue
            cnt_b = min(cnt_b, (Q[i] // B[i]))
        ans = max(ans, cnt_a + cnt_b)
        cnt_a += 1
        flag = True
        for i in range(N):
            Q[i] -= A[i]
            if Q[i] < 0: 
                flag = False
        if not flag: break
    print(ans)

if __name__ == '__main__':
    main()
```

## D - Island Tour 

### Solution 1:  modified prefix sums, observe that only change when cut separates two sections

```py
def main():
    N, M = map(int, input().split())
    arr = list(map(lambda x: int(x) - 1, input().split()))
    adj = [Counter() for _ in range(N)]
    cur = 0
    for i in range(M - 1):
        u, v = arr[i], arr[i + 1]
        adj[u][v] += 1
        adj[v][u] += 1
        cur += abs(u - v)
    ans = cur
    for r in range(N - 2):
        for u in adj[r]:
            if u > r:
                old = u - r
                new = N - u + r
                cur += (new - old) * adj[r][u]
            elif u < r:
                old = N - r + u
                new = r - u
                cur += (new - old) * adj[r][u]
        ans = min(ans, cur)
    print(ans)

if __name__ == '__main__':
    main()
```

## E - Chords 

### Solution 1:  stack, chords on circle

```py
UNVISITED = -1
def main():
    N = int(input())
    A = [UNVISITED] * (2 * N + 1)
    B = [UNVISITED] * (2 * N + 1)
    for i in range(N):
        u, v = map(int, input().split())
        if u > v: u, v = v, u
        A[u] = i
        B[v] = i
    stk = []
    for i in range(1, 2 * N + 1):
        if A[i] != UNVISITED: stk.append(A[i])
        else:
            if not stk: return print("Yes")
            v = stk.pop()
            if B[i] != v: return print("Yes")
    print("No")

if __name__ == '__main__':
    main()
```

## F - Negative Traveling Salesman

### Solution 1:  traveling salesman problem, dynamic programming, floyd warshall, all pairs shortest path

```py
import math
def main():
    N, M = map(int, input().split())
    dist = [[math.inf] * N for _ in range(N)]
    for i in range(N):
        dist[i][i] = 0
    for _ in range(M):
        u, v, w = map(int, input().split())
        u -= 1; v -= 1
        dist[u][v] = w
    end_mask = (1 << N) - 1
    # floyd warshall, all pairs shortest path
    for i in range(N):
        for j in range(N):
            for k in range(N):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    # dp for tsp
    dp = [[math.inf] * N for _ in range(1 << N)]
    for start in range(N):
        dp[1 << start][start] = 0
    for mask in range(1 << N):
        for u in range(N):
            if dp[mask][u] == math.inf: continue
            for v in range(N):
                if (mask >> v) & 1: continue
                nmask = mask | (1 << v)
                dp[nmask][v] = min(dp[nmask][v], dp[mask][u] + dist[u][v])
    ans = min(dp[end_mask])
    print(ans if ans < math.inf else "No")

if __name__ == '__main__':
    main()
```

## 

### Solution 1: 

```py

```



# Atcoder Beginner Contest 339

## C - Perfect Bus

### Solution 1:  min, sum, find the lowest point under 0

```py
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    mn = cur = 0
    for x in arr:
        cur += x
        mn = min(mn, cur)
    cur = abs(mn)
    for x in arr:
        cur += x
    print(cur)

if __name__ == '__main__':
    main()
```

## D - Synchronized Players

### Solution 1:  bfs

```py
from itertools import product
from collections import deque
import math

def main():
    n = int(input())
    grid = [input() for _ in range(n)]
    pos = []
    for r, c in product(range(n), repeat = 2):
        if grid[r][c] == "P": pos.extend((r, c))
    dq = deque([tuple(pos)])
    dist = [[[[math.inf] * n for _ in range(n)] for _ in range(n)] for _ in range(n)]
    dist[pos[0]][pos[1]][pos[2]][pos[3]] = 0
    in_bounds = lambda r, c: 0 <= r < n and 0 <= c < n
    while dq:
        r1, c1, r2, c2 = dq.popleft()
        if (r1, c1) == (r2, c2): return print(dist[r1][c1][r2][c2])
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr1, nc1, nr2, nc2 = r1 + dr, c1 + dc, r2 + dr, c2 + dc
            if not in_bounds(nr1, nc1) or grid[nr1][nc1] == "#": nr1, nc1 = r1, c1
            if not in_bounds(nr2, nc2) or grid[nr2][nc2] == "#": nr2, nc2 = r2, c2
            if dist[nr1][nc1][nr2][nc2] < math.inf: continue
            dist[nr1][nc1][nr2][nc2] = dist[r1][c1][r2][c2] + 1
            dq.append((nr1, nc1, nr2, nc2))
    print(-1)
if __name__ == '__main__':
    main()
```

## E - Smooth Subsequence

### Solution 1:  segment tree to get range max queries, point updates

```cpp
const int neutral = 0;

struct SegmentTree {
    int size;
    vector<int> nodes;

    void init(int num_nodes) {
        size = 1;
        while (size < num_nodes) size *= 2;
        nodes.resize(size * 2, neutral);
    }

    int func(int x, int y) {
        return max(x, y);
    }

    void ascend(int segment_idx) {
        while (segment_idx > 0) {
            int left_segment_idx = 2 * segment_idx, right_segment_idx = 2 * segment_idx + 1;
            nodes[segment_idx] = func(nodes[left_segment_idx], nodes[right_segment_idx]);
            segment_idx >>= 1;
        }
    }

    void update(int segment_idx, int val) {
        segment_idx += size;
        nodes[segment_idx] = val;
        segment_idx >>= 1;
        ascend(segment_idx);
    }

    int query(int left, int right) {
        left += size, right += size;
        int res = neutral;
        while (left <= right) {
            if (left & 1) {
                res = max(res, nodes[left]);
                left++;
            }
            if (~right & 1) {
                res = max(res, nodes[right]);
                right--;
            }
            left >>= 1, right >>= 1;
        }
        return res;
    }
};

const int MAXN = 5 * 1e5 + 5;
int N, D, dp[MAXN];
vector<int> arr;

signed main() {
    cin >> N >> D;
    arr.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> arr[i];
    }
    SegmentTree seg;
    seg.init(MAXN);
    memset(dp, 0, sizeof(dp));
    int ans = 0;
    for (int i = 0; i < N; i++) {
        int l = max(0LL, arr[i] - D), r = min(MAXN - 1, arr[i] + D);
        int mx = seg.query(l, r);
        ans = max(ans, mx + 1);
        seg.update(arr[i], mx + 1);
    }
    cout << ans << endl;
    return 0;
}
```

## F - Product Equality

### Solution 1:  multiplication of large numbers, modular arithmetic, hash table

I think this solution is wrong some reason, just read through and you will remember.

```py
import random
P = 20
MODS = [random.randint(10**9, 2 * 10**9) for _ in range(P)]

def main():
    n = int(input())
    arr = [[0] * P for _ in range(n)]
    counter = [Counter() for _ in range(P)]
    for i in range(n):
        a = int(input())
        for j in range(P):
            ma = a % MODS[j]
            arr[i][j] = ma
            counter[j][ma] += 1
    ans = 0
    for i in range(n):
        for j in range(i, n):
            val = arr[i][0] * arr[j][0]
            cand = counter[0][val % MODS[0]]
            if all(counter[k][(arr[i][k] * arr[j][k]) % MODS[k]] == cand for k in range(1, P)):
                ans += cand
                if i != j: ans += cand
    print(ans)

if __name__ == '__main__':
    main()
```

## G - Smaller Sum

### Solution 1:  merge sort tree, online queries, cumulative sum of all elements less than or equal to X,

```cpp
const int N = 2e5 + 10;
vector<int> tree[4 * N], psum[4 * N];
int n, arr[N], a, b, c;

struct MergeSortTree {
    void build(int u, int left, int right) {
        if (left == right) {
            tree[u].push_back(arr[left]);
            psum[u].push_back(arr[left]);
            return;
        }
        int mid = (left + right) >> 1;
        int left_segment = u << 1;
        int right_segment = left_segment | 1;
        build(left_segment, left, mid);
        build(right_segment, mid + 1, right);
        merge(tree[left_segment].begin(), tree[left_segment].end(), tree[right_segment].begin(), tree[right_segment].end(), back_inserter(tree[u]));
        int l = 0, r = 0, nl = tree[left_segment].size(), nr = tree[right_segment].size(), cur = 0;
        while (l < nl or r < nr) {
            if (l < nl and r < nr) {
                if (tree[left_segment][l] <= tree[right_segment][r]) {
                    cur += tree[left_segment][l];
                    l += 1;
                } else {
                    cur += tree[right_segment][r];
                    r += 1;
                }
            } else if (l < nl) {
                cur += tree[left_segment][l];
                l += 1;
            } else {
                cur += tree[right_segment][r];
                r += 1;
            }
            psum[u].push_back(cur);
        }
    }
    // not greater than k, so <= k we want
    int query(int u, int left, int right, int i, int j, int k) {
        if (i > right || left > j) return 0; // NO OVERLAP
        if (i <= left && right <= j) { // COMPLETE OVERLAP
            int idx = upper_bound(tree[u].begin(), tree[u].end(), k) - tree[u].begin();
            return idx > 0 ? psum[u][idx - 1] : 0;
        }
        // PARTIAL OVERLAP
        int mid = (left + right) >> 1;
        int left_segment = u << 1;
        int right_segment = left_segment | 1;
        return query(left_segment, left, mid, i, j, k) + query(right_segment, mid + 1, right, i, j, k);
    }
};

signed main() {
    cin >> n;
    for (int i = 0; i < n; i++) {
        cin >> arr[i];
    }
    MergeSortTree mst;
    mst.build(1, 0, n - 1);
    int q, ans = 0;
    cin >> q;
    while (q--) {
        cin >> a >> b >> c;
        int L = a ^ ans, R = b ^ ans, K = c ^ ans;
        ans = mst.query(1, 0, n - 1, L - 1, R - 1, K);
        cout << ans << endl;
    }
    return 0;
}
```



# Atcoder Beginner Contest 340

## C - Divide and Divide

### Solution 1:  dynamic programming

```py
import math

def main():
    N = int(input())
    dp = Counter({N: 1})
    ans = 0
    while dp:
        ndp = Counter()
        for k, v in dp.items():
            ans += v * k
            floor = k // 2
            ceil = (k + 1) // 2
            if floor > 1: ndp[floor] += v
            if ceil > 1: ndp[ceil] += v
        dp = ndp
    print(ans)

if __name__ == '__main__':
    main()
```

## D - Super Takahashi Bros. 

### Solution 1:  dijkstra, directed graph, single source shortest path froms source to destination

```py
import heapq
def dijkstra(adj, src, dst):
    N = len(adj)
    min_heap = [(0, src)]
    vis = [0] * N
    while min_heap:
        cost, u = heapq.heappop(min_heap)
        if u == dst: return cost
        if vis[u]: continue
        vis[u] = 1
        for v, w in adj[u]:
            if vis[v]: continue
            heapq.heappush(min_heap, (cost + w, v))
    return -1
def main():
    N = int(input())
    adj = [[] for _ in range(N)]
    for i in range(N - 1):
        A, B, X = map(int, input().split())
        X -= 1
        adj[i].append((i + 1, A))
        adj[i].append((X, B))
    ans = dijkstra(adj, 0, N - 1)
    print(ans)

if __name__ == '__main__':
    main()
```

## E - Mancala 2

### Solution 1:  range update, point query, addition on segment, lazy propagation

```py
class LazySegmentTree:
    def __init__(self, n: int, neutral: int, noop: int):
        self.neutral = neutral
        self.size = 1
        self.noop = noop
        self.n = n 
        while self.size<n:
            self.size*=2
        self.tree = [neutral for _ in range(self.size*2)]

    def is_leaf_node(self, segment_right_bound, segment_left_bound) -> bool:
        return segment_right_bound - segment_left_bound == 1

    def operation(self, x: int, y: int) -> int:
        return x + y

    def propagate(self, segment_idx: int, segment_left_bound: int, segment_right_bound: int) -> None:
        # do not want to propagate if it is a leaf node
        if self.is_leaf_node(segment_right_bound, segment_left_bound): return
        left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
        self.tree[left_segment_idx] = self.operation(self.tree[left_segment_idx], self.tree[segment_idx])
        self.tree[right_segment_idx] = self.operation(self.tree[right_segment_idx], self.tree[segment_idx])
        self.tree[segment_idx] = self.noop
    
    def update(self, left: int, right: int, val: int) -> None:
        stack = [(0, self.size, 0)]
        while stack:
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # COMPLETE OVERLAP
            if segment_left_bound >= left and segment_right_bound <= right:
                self.tree[segment_idx] = self.operation(self.tree[segment_idx], val)
                continue
            # PARTIAL OVERLAP
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.propagate(segment_idx, segment_left_bound, segment_right_bound)
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])

    def query(self, i: int) -> int:
        stack = [(0, self.size, 0)]
        while stack:
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if i < segment_left_bound or i >= segment_right_bound: continue
            # LEAF NODE
            if self.is_leaf_node(segment_right_bound, segment_left_bound): 
                return self.tree[segment_idx]
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.propagate(segment_idx, segment_left_bound, segment_right_bound)            
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
    
    def __repr__(self) -> str:
        return f"array: {self.tree}"

# [L, R)
def main():
    N, M = map(int, input().split())
    arr = list(map(int, input().split()))
    queries = list(map(int, input().split()))
    seg = LazySegmentTree(N, 0, 0)
    for i in range(N):
        seg.update(i, i + 1, arr[i])
    for i in range(M):
        idx = queries[i]
        balls = seg.query(idx)
        x = balls // N
        seg.update(idx, idx + 1, -balls)
        if x > 0:
            seg.update(0, N, x)
        if balls % N > 0:
            segment = [(idx + 1) % N, (idx + balls % N) % N]
            if segment[1] < segment[0]:
                seg.update(segment[0], N, 1)
                seg.update(0, segment[1] + 1, 1)
            else:
                seg.update(segment[0], segment[1] + 1, 1)
    ans = [seg.query(i) for i in range(N)]
    print(*ans)
```

## F - S = 1

### Solution 1:  linear diophantine equation, extended euclidean algorithm

```py
def extended_euclidean(a, b, x, y):
    if b == 0: return a, 1, 0
    g, x1, y1 = extended_euclidean(b, a % b, x, y)
    return g, y1, x1 - y1 * (a // b)

def main():
    A, B = map(int, input().split())
    C = 2
    # Ax + By = C
    # Bx - Ay = C
    g, x, y = extended_euclidean(B, -A, 0, 0)
    if C % g != 0: return print(-1)
    x *= 2 // g
    y *= 2 // g
    print(x, y)

if __name__ == '__main__':
    main()
```

## G - Leaf Color

### Solution 1:  virtual or aux tree, lca, binary jumping, dfs, dp on tree

```py
from collections import deque, defaultdict
LOG = 18
MOD = 998244353

def main():
    n = int(input())
    colors = list(map(int, input().split()))
    adj = [[] for _ in range(n)]
    virt = [[] for _ in range(n)]
    for _ in range(n - 1):
        u, v = map(int, input().split())
        u -= 1; v -= 1
        adj[u].append(v)
        adj[v].append(u)
    depth = [0] * n
    parent = [-1] * n
    # CONSTRUCT THE PARENT, DEPTH AND FREQUENCY ARRAY FROM ROOT
    def bfs(root):
        queue = deque([root])
        vis = [0] * n
        vis[root] = 1
        dep = 0
        while queue:
            for _ in range(len(queue)):
                node = queue.popleft()
                depth[node] = dep
                for nei in adj[node]:
                    if vis[nei]: continue
                    parent[nei] = node
                    vis[nei] = 1
                    queue.append(nei)
            dep += 1
    bfs(0)
    # CONSTRUCT THE SPARSE TABLE FOR THE BINARY JUMPING TO ANCESTORS IN TREE
    ancestor = [[-1] * n for _ in range(LOG)]
    ancestor[0] = parent[:]
    for i in range(1, LOG):
        for j in range(n):
            if ancestor[i - 1][j] == -1: continue
            ancestor[i][j] = ancestor[i - 1][ancestor[i - 1][j]]
    def kth_ancestor(node, k):
        for i in range(LOG):
            if (k >> i) & 1:
                node = ancestor[i][node]
        return node
    def lca(u, v):
        # ASSUME NODE u IS DEEPER THAN NODE v   
        if depth[u] < depth[v]:
            u, v = v, u
        # PUT ON SAME DEPTH BY FINDING THE KTH ANCESTOR
        k = depth[u] - depth[v]
        u = kth_ancestor(u, k)
        if u == v: return u
        for i in reversed(range(LOG)):
            if ancestor[i][u] != ancestor[i][v]:
                u, v = ancestor[i][u], ancestor[i][v]
        return ancestor[0][u]
    # CONSTRUCT THE DISCOVERY TIME ARRAY FOR EACH NODE
    disc = [0] * n
    timer = 0
    def dfs(u, p):
        nonlocal timer
        disc[u] = timer
        timer += 1
        for v in adj[u]:
            if v == p: continue
            dfs(v, u)
    dfs(0, -1)
    # CONSTRUCT THE AUXILIARY TREES FOR EACH SET S OF THE SAME COLOR
    tree_sets = defaultdict(list)
    for u in sorted(range(n), key = lambda i: disc[i]):
        tree_sets[colors[u]].append(u)
    # DP ON AUX TREES
    ans = 0
    def dp(u, c):
        nonlocal ans
        res, sum_ = 1, 0
        for v in virt[u]:
            child = dp(v, c)
            res = (res * (child + 1)) % MOD
            sum_ = (sum_ + child) % MOD
        if colors[u] == c:
            ans = (ans + res) % MOD
        else:
            ans = (ans + res - sum_ - 1) % MOD
            res = (res - 1) % MOD
        return res
    for c, S in tree_sets.items():
        P = set(S)
        m = len(S)
        for i in range(1, m):
            P.add(lca(S[i - 1], S[i]))
        P = sorted(P, key = lambda i: disc[i])
        parents = [None] * (len(P) - 1)
        for i in range(1, len(P)):
            par = lca(P[i - 1], P[i])
            parents[i - 1] = par
            virt[par].append(P[i])
        dp(P[0], c)
        for p in parents:
            virt[p].clear()
    print(ans)                 

if __name__ == '__main__':
    main()
```

# Atcoder Beginner Contest 341

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



# Atcoder Beginner Contest 342

## C - Many Replacement 

### Solution 1:  hash map

```py
import string
def main():
    N = int(input())
    S = list(input())
    Q = int(input())
    char_map = {ch: ch for ch in string.ascii_lowercase}
    for _ in range(Q):
        c, d = input().split()
        for ch in char_map:
            if char_map[ch] == c: char_map[ch] = d
    ans = "".join([char_map.get(ch) for ch in S])
    print(ans)
if __name__ == '__main__':
    main()
```

## D - Square Pair 

### Solution 1:  math, square, multiplicity of prime, counter

```py
def main():
    N = int(input())
    arr = list(map(int, input().split()))
    ans = 0
    counts = [0] * 200_000
    for num in arr:
        x, i = num, 2
        while i * i <= x:
            while x % (i * i) == 0:
                x //= i * i
            i += 1
        ans += counts[x]
        counts[x] += 1
    print(ans + counts[0] * (N - counts[0]))

if __name__ == '__main__':
    main()
```

## E - Last Train 

### Solution 1:  max heap, dp, math, backwards, from last station to station i

```py
import math
from heapq import heappush, heappop
def main():
    N, M = map(int, input().split())
    adj = [[] for _ in range(N)]
    ans = [-math.inf] * N
    for _ in range(M):
        l, d, k, c, u, v = map(int, input().split())
        u -= 1; v -= 1
        adj[v].append((u, l, d, k, c))
    maxheap = [(-math.inf, N - 1)]
    while maxheap:
        t, u = heappop(maxheap)
        t *= -1
        if t < ans[u]: continue
        for v, l, d, k, c in adj[u]:
            rem = t - l - c 
            nt = l + min(k - 1, rem // d) * d 
            if nt < 0: continue
            if nt > ans[v]:
                ans[v] = nt 
                heappush(maxheap, (-nt, v))
    for t in ans[:-1]:
        print(t if t > -math.inf else "Unreachable")
if __name__ == '__main__':
    main()
```

## F - Black Jack 

### Solution 1: 

```py

```

## G - Retroactive Range Chmax 

### Solution 1: 

```py

```



# Atcoder Beginner Contest 343

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



# Atcoder Beginner Contest 344

## C - A+B+C

### Solution 1:  hash table, set

```py
from itertools import product
def main():
    N = int(input())
    A = list(map(int, input().split()))
    M = int(input())
    B = list(map(int, input().split()))
    L = int(input())
    C = list(map(int, input().split()))
    Q = int(input())
    queries = list(map(int, input().split()))
    sums = set()
    for a, b, c in product(A, B, C):
        sums.add(a + b + c)
    for i in range(Q):
        print("Yes" if queries[i] in sums else "No")
```

## D - String Bags

### Solution 1:  kmp, dp

```py
import math
def kmp(s):
    n = len(s)
    pi = [0] * n
    for i in range(1, n):
        j = pi[i - 1]
        while j > 0 and s[i] != s[j]: 
            j = pi[j - 1]
        if s[j] == s[i]: j += 1
        pi[i] = j
    return pi

def main():
    T = input()
    n = len(T)
    N = int(input())
    bags = [None] * N
    for i in range(N):
        bags[i] = list(map(str, input().split()[1:]))
    dp = [math.inf] * (n + 1)
    dp[0] = 0 # empty string
    for bag in bags: # 100
        ndp = dp.copy()
        for s in bag: # 10
            ns = len(s)
            parr = kmp(s + "#" + T)[ns:] # 110
            for i in range(n + 1): # 100
                if parr[i] == ns:
                    ndp[i] = min(ndp[i], dp[i - ns] + 1)
        dp = ndp
    print(dp[-1] if dp[-1] < math.inf else -1)
    
if __name__ == '__main__':
    main()
```

## E - Insert or Erase

### Solution 1:  doubly linked list, insert, erase, nxt and prv pointers

```py
from collections import defaultdict
def main():
    N = int(input())
    arr = list(map(int, input().split()))
    prv = defaultdict(lambda: None)
    nxt = defaultdict(lambda: None)
    Q = int(input())
    for i in range(N):
        if i > 0:
            prv[arr[i]] = arr[i - 1]
        if i + 1 < N:
            nxt[arr[i]] = arr[i + 1]
    start = arr[0]
    def erase(x):
        prv[nxt[x]] = prv[x]
        nxt[prv[x]] = nxt[x]
    def insert(x, y): # insert y after x
        nxt[y] = nxt[x]
        prv[y] = x
        prv[nxt[x]] = y
        nxt[x] = y
    for _ in range(Q):
        query = list(map(int, input().split()))
        if query[0] == 1:
            x, y = query[1:]
            insert(x, y) # insert y after x
        else:
            x = query[1]
            if start == x:
                start = nxt[x]
            erase(x)
    ans = []
    while start is not None:
        ans.append(start)
        start = nxt[start]
    print(*ans)

if __name__ == '__main__':
    main()
```

## F - Earn to Advance 

### Solution 1:  dp, min actions, dp on grid

```py
from collections import defaultdict
from itertools import product
import math

def ceil(x, y):
    return (x + y - 1) // y 

def main():
    N = int(input())
    P = [list(map(int, input().split())) for _ in range(N)]
    R = [list(map(int, input().split())) for _ in range(N)]
    D = [list(map(int, input().split())) for _ in range(N)]
    dp = [[defaultdict(lambda: (math.inf, 0)) for _ in range(N)] for _ in range(N)]
    dp[0][0][P[0][0]] = (0, 0) # (action, money)
    for r, c in product(range(N), repeat = 2):
        if r > 0: # move down
            for payer, (actions, money) in dp[r - 1][c].items():
                need = max(0, D[r - 1][c] - money)
                take = ceil(need, payer)
                npayer = max(payer, P[r][c])
                nmoney = money - D[r - 1][c] + take * payer
                naction = actions + take + 1
                if naction < dp[r][c][npayer][0]: dp[r][c][npayer] = (naction, nmoney)
                elif naction == dp[r][c][npayer][0] and nmoney > dp[r][c][npayer][1]: dp[r][c][npayer] = (naction, nmoney)
        if c > 0: # move right
            for payer, (actions, money) in dp[r][c - 1].items():
                need = max(0, R[r][c - 1] - money)
                take = ceil(need, payer)
                npayer = max(payer, P[r][c])
                nmoney = money - R[r][c - 1] + take * payer
                naction = actions + take + 1
                if naction < dp[r][c][npayer][0]: dp[r][c][npayer] = (naction, nmoney)
                elif naction == dp[r][c][npayer][0] and nmoney > dp[r][c][npayer][1]: dp[r][c][npayer] = (naction, nmoney)
    print(min([x for x, _ in dp[-1][-1].values()]))

if __name__ == '__main__':
    main()
```

## G - Points and Comparison 

### Solution 1:  lines

```py

```


# Atcoder Beginner Contest 346

## E - Paint 

### Solution 1: offline query, sort, reverse, track count of blocked rows and cols

```py
MAXN = 200_000 + 5

def main():
    R, C, M = map(int, input().split())
    queries = [None] * M
    for i in range(M):
        t, a, x = map(int, input().split())
        queries[i] = (t, a, x)
    colors = [0] * MAXN
    row = [0] * R
    col = [0] * C
    num_rows = num_cols = 0
    for t, a, x in reversed(queries):
        a -= 1
        if t == 1: # repaint row
            if not row[a]:
                colors[x] += C - num_cols
                row[a] = 1
                num_rows += 1
        else: # repaint col
            if not col[a]:
                colors[x] += R - num_rows
                col[a] = 1
                num_cols += 1
    total = sum(colors)
    colors[0] += R * C - total
    ans = []
    for c, cnt in enumerate(colors):
        if cnt > 0: ans.append((c, cnt))
    print(len(ans))
    for color, cnt in ans:
        print(color, cnt)
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

# Atcoder Beginner Contest 348

## D - Medicines on Grid 

### Solution 1:  grid, created directed graph, find path from source node to target node within the directed graph, that is target is reachable from source

```py
from itertools import product
from collections import deque
def main():
    R, C = map(int, input().split())
    grid = [list(input()) for _ in range(R)]
    M = int(input())
    med = {}
    adj = [[] for _ in range(M + 1)]
    for i in range(M):
        r, c, v = map(int, input().split())
        r -= 1; c -= 1
        med[(r, c)] = (v, i)
    target = M
    neighborhood = lambda r, c: [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
    in_bounds = lambda r, c: 0 <= r < R and 0 <= c < C
    def bfs(r, c):
        E, i = med[(r, c)]
        vis = [[0] * C for _ in range(R)]
        q = deque([(r, c, E)])
        while q:
            r, c, e = q.popleft()
            for nr, nc in neighborhood(r, c):
                if not in_bounds(nr, nc) or vis[nr][nc] or grid[nr][nc] == "#": continue
                vis[nr][nc] = 1
                ne = e - 1
                if ne > 0: q.append((nr, nc, ne))
                if grid[nr][nc] == "T":
                    adj[i].append(target)
                elif (nr, nc) in med:
                    adj[i].append(med[(nr, nc)][1])
    # CONSTRUCT DIRECTED GRAPH WITH MEDICINE AND TARGET
    for r, c in med:
        bfs(r, c)
    vis = [0] * (M + 1)
    q = []
    for r, c in product(range(R), range(C)):
        if grid[r][c] == "S" and (r, c) in med:
            vis[med[(r, c)][1]] = 1
            q.append(med[(r, c)][1])
            break
    while q:
        u = q.pop()
        if u == target: return print("Yes")
        for v in adj[u]:
            if vis[v]: continue 
            vis[v] = 1
            q.append(v)
    print("No")

if __name__ == '__main__':
    main()
```

## E - Minimize Sum of Distances 

### Solution 1:  reroot dp tree, dp on tree

```cpp
const int INF = INT64_MAX, MAXN = 1e5 + 5;
int labels[MAXN], sums[MAXN], costs[MAXN], psums[MAXN], pcosts[MAXN], dp[MAXN];
int N;
vector<vector<int>> adj;

void dfs1(int u, int p) {
    sums[u] = labels[u];
    for (int v : adj[u]) {
        if (v == p) continue;
        dfs1(v, u);
        sums[u] += sums[v];
        costs[u] += sums[v] + costs[v];
    }
}
void dfs2(int u, int p) {
    dp[u] = costs[u] + pcosts[u];
    for (int v : adj[u]) {
        if (v == p) continue;
        psums[v] = psums[u] + sums[u] - sums[v];
        pcosts[v] = pcosts[u] + costs[u] - costs[v] - sums[v] + psums[v];
        dfs2(v, u);
    }
}

signed main() {
    cin >> N;
    adj.assign(N, {});
    for (int i = 0; i < N - 1; ++i) {
        int u, v;
        cin >> u >> v;
        u--; v--;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    for (int i = 0; i < N; ++i) {
        cin >> labels[i];
    }
    memset(sums, 0LL, sizeof(sums));
    memset(costs, 0LL, sizeof(costs));
    dfs1(0, -1);
    memset(psums, 0LL, sizeof(psums));
    memset(pcosts, 0LL, sizeof(pcosts));
    dfs2(0, -1);
    int ans = INF;
    for (int i = 0; i < N; i++) {
        ans = min(ans, dp[i]);
    }
    cout << ans << endl;
    return 0;
}
```

## F - Oddly Similar 

### Solution 1:  bitset, bit manipulation, bitwise xor to track odd counts

```py
const int MAXN = 2e3 + 5;
int N, M, A[MAXN][MAXN];

signed main() {
    cin >> N >> M;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            cin >> A[i][j];
        }
    }
    vector<bitset<MAXN>> values(1000, bitset<MAXN>());
    vector<bitset<MAXN>> bitmasks(N, bitset<MAXN>());
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            values[A[j][i]].set(j);
        }
        for (int j = 0; j < N; j++) {
            bitmasks[j] ^= values[A[j][i]];
        }
        for (int j = 0; j < 1000; j++) {
            values[j].reset();
        }
    }
    int ans = 0;
    for (int i = 0; i < N; i++) {
        ans += bitmasks[i].count();
    }
    if (M & 1) ans -= N;
    cout << ans / 2 << endl;
    return 0;
}
```

## G - Max (Sum - Max) 

### Solution 1: 

```py

```

```cpp
#include <bits/stdc++.h>

using namespace std;
using ll = long long;

constexpr ll inf = 1ll << 60;

int n;
vector<pair<ll, ll>> x; // x : (a, b)

void read() {
    cin >> n;
    x.resize(n);
    for (auto &[a, b] : x) {
        cin >> a >> b;
    }
    sort(x.begin(), x.end(), [&](const auto &a, const auto &b) {
        return a.second < b.second;
    });
}

template <class F>
vector<ll> monotone_maxima(F &f, int h, int w) {
    vector<ll> ret(h);
    auto sol = [&](auto &&self, const int l_i, const int r_i, const int l_j, const int r_j) -> void {
        const int m_i = (l_i + r_i) / 2;
        int max_j = l_j;
        ll max_val = -inf;
        for (int j = l_j; j <= r_j; ++j) {
            const ll v = f(m_i, j);
            if (v > max_val) {
                max_j = j;
                max_val = v;
            }
        }
        ret[m_i] = max_val;

        if (l_i <= m_i - 1) {
            self(self, l_i, m_i - 1, l_j, max_j);
        }
        if (m_i + 1 <= r_i) {
            self(self, m_i + 1, r_i, max_j, r_j);
        }
    };
    sol(sol, 0, h - 1, 0, w - 1);
    return ret;
}

/*
what is a and b in this array? 
a is a vector of values for the right interval
b is a vector of values for the left interval

monotone_maxima
*/
vector<ll> max_plus_convolution(const vector<ll> &a, const vector<ll> &b) {
    const int n = (int)a.size(), m = (int)b.size();
    auto f = [&](int i, int j) {
        if (i < j or i - j >= m) {
            return -inf;
        }
        return a[j] + b[i - j];
    };

    return monotone_maxima(f, n + m - 1, n);
}

vector<ll> sol(const int l, const int r) {
    if (r - l == 1) {
        const vector<ll> ret = {-inf, x[l].first - x[l].second};
        return ret;
    }
    const int m = (l + r) / 2;
    const auto res_l = sol(l, m);
    const auto res_r = sol(m, r);

    vector<ll> sorted_l(m - l);
    for (int i = l; i < m; ++i) {
        sorted_l[i - l] = x[i].first;
    }
    sort(sorted_l.begin(), sorted_l.end(), greater());
    for (int i = 1; i < m - l; ++i) {
        sorted_l[i] += sorted_l[i - 1];
    }
    sorted_l.insert(sorted_l.begin(), -inf);
    // O(n)
    auto res = max_plus_convolution(res_r, sorted_l);

    for (int i = 0; i < (int)res_l.size(); ++i) {
        res[i] = max(res[i], res_l[i]);
    }
    for (int i = 0; i < (int)res_r.size(); ++i) {
        res[i] = max(res[i], res_r[i]);
    }
    return res;
}

void process() {
    auto ans = sol(0, n);
    for (int i = 1; i <= n; ++i) {
        cout << ans[i] << endl;
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    read();
    process();
}
```

# Atcoder Beginner Contest 349

##

### Solution 1: 

```py

```

## F - Subsequence LCM 

### Solution 1:  number theory, prime factorization, dynamic programming, bitmasks, counting

```py
MOD = 998244353

def prime_factors(num):
    counts = Counter()
    for p in range(2, num):
        if p * p > num: break 
        if num % p != 0: continue 
        while num % p == 0: 
            counts[p] += 1
            num //= p
    if num > 1: counts[num] += 1
    return counts

def main():
    N, M = map(int, input().split())
    arr = list(filter(lambda x: M % x == 0, map(int, input().split())))
    pfcount = prime_factors(M)
    bits = list(sorted(pfcount))
    bvals = [pow(bit, pfcount[bit]) for bit in bits]
    k = len(bits)
    mcounts = [0] * (1 << k)
    for num in arr:
        mask = 0
        for i in range(k):
            if num % bvals[i] == 0:
                mask |= 1 << i
        mcounts[mask] += 1
    dp = [[0] * (1 << k) for _ in range(1 << k)] # dp[i][mask] 
    dp[0][0] = 1
    for i in range(1, 1 << k):
        ways = max(0, pow(2, mcounts[i], MOD) - 1) # number of ways to take it
        # 1 way to not take it
        for mask in range(1 << k):
            dp[i][mask] = (dp[i][mask] + dp[i - 1][mask]) % MOD # don't take 
            dp[i][mask | i] = (dp[i][mask | i] + dp[i - 1][mask] * ways) % MOD # take it
    ans = (dp[-1][-1] * pow(2, mcounts[0], MOD)) % MOD
    if M == 1: ans = (ans - 1) % MOD # subtract the empty set
    print(ans)

if __name__ == '__main__':
    main()
```

## G - Palindrome Construction 

### Solution 1:  manacher, greedy, dsu

This isn't completely the answer, but it was good practice to get this solution.  I couldn't figure out how to get the lexicographically smallest output.

```py
class UnionFind:
    def __init__(self, n: int):
        self.size = [1]*n
        self.parent = list(range(n))
    
    def find(self,i: int) -> int:
        while i != self.parent[i]:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i
    """
    returns true if the nodes were not union prior. 
    """
    def union(self,i: int,j: int) -> bool:
        i, j = self.find(i), self.find(j)
        if i!=j:
            if self.size[i] < self.size[j]:
                i,j=j,i
            self.parent[j] = i
            self.size[i] += self.size[j]
            return True
        return False
    def is_same_connected_components(self, i: int, j: int) -> bool:
        return self.find(i) == self.find(j)
    def size_(self, i):
        return self.size[self.find(i)]
    def __repr__(self) -> str:
        return f'parents: {[(i, parent) for i, parent in enumerate(self.parent)]}, sizes: {self.size}'

def main():
    n = int(input())
    arr = list(map(lambda x: int(x) + 1, input().split()))
    i = j = 0
    valid = True
    dsu = UnionFind(n)
    while i < n: 
        while i - j >= 0 and i + j < n and arr[i] - j > 0:
            dsu.union(i - j, i + j)
            j += 1
        k = 1
        while i - k >= 0 and k + arr[i - k] < j and valid:
            if arr[i - k] != arr[i + k]: valid = False 
            k += 1
        i += k
        j -= k
    for i in range(n):
        if i - arr[i] < 0 or i + arr[i]  >= n: continue
        u, v = dsu.find(i - arr[i]), dsu.find(i + arr[i])
        if u == v: 
            valid = False
            break
    if not valid: return print("No")
    ans = [1] * n
    values = {0: 1}
    for i in range(n):
        root = dsu.find(i)
        if root not in values:
            values[root] = len(values) + 1
        ans[i] = values[root]
    print("Yes")
    print(*ans)

if __name__ == "__main__":
    main()
```