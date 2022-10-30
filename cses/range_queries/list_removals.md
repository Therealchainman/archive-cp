# List Removals

## Summary

### Solution 1:  segment tree + point updates + kth query

```py
import os,sys
from io import BytesIO, IOBase
from typing import List

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

"""
Kth Segment Tree 

- point updates
- point query
- kth element query

Sets the value at segment tree equal to val

Can be used to get the count of elements in a range, by setting val=1
"""
class SegmentTree:
    def __init__(self, n: int, neutral: int):
        self.neutral = neutral
        self.size = 1
        self.n = n
        while self.size<n:
            self.size*=2
        self.tree = [0 for _ in range(self.size*2)]

    def ascend(self, segment_idx: int) -> None:
        while segment_idx > 0:
            segment_idx -= 1
            segment_idx >>= 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.tree[segment_idx] = self.tree[left_segment_idx] + self.tree[right_segment_idx]
        
    def update(self, segment_idx: int, val: int) -> None:
        segment_idx += self.size - 1
        self.tree[segment_idx] = val
        self.ascend(segment_idx)

    # computes the kth one from right to left, so finds index where there are k ones to the right.  
    def k_query(self, k: int) -> int:
        left_segment_bound, right_segment_bound, segment_idx = 0, self.size, 0
        while right_segment_bound - left_segment_bound != 1:
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            mid_point = (left_segment_bound + right_segment_bound) >> 1
            if k <= self.tree[left_segment_idx]: # continue in the left branch
                segment_idx, right_segment_bound = left_segment_idx, mid_point
            else: # continue in right branch and decrease the number of 1s needed in the right branch
                segment_idx, left_segment_bound = right_segment_idx, mid_point
                k -= self.tree[left_segment_idx]
        return left_segment_bound
    
    def __repr__(self) -> str:
        return f"array: {self.tree}"

def main():
    n = int(input())
    arr = list(map(int,input().split()))
    segTree = SegmentTree(n, 0)
    for idx in range(n):
        segTree.update(idx, 1)
    result = [0]*n
    queries = map(int, input().split())
    for i, idx in enumerate(queries):
        index = segTree.k_query(idx)
        segTree.update(index, 0)
        result[i] = arr[index]
    return ' '.join(map(str, result))

if __name__ == '__main__':
    print(main())
```