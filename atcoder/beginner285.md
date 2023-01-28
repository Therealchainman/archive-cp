# Atcoder Beginner Contest 285

## Edge Checker 2

### Solution 1:  tree structured in a way where the parent is just the floor division by 2

```py
def main():
    a, b = map(int, input().split())
    return 'Yes' if b//2 == a else 'No'

if __name__ == '__main__':
    print(main())
```

## Longest Uncommon Prefix

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

## abc285_brutmhyhiizp

### Solution 1:

```py

```

## Change Usernames

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

## Substring of Sorted String

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