import os,sys
from io import BytesIO, IOBase
from typing import List
from collections import deque
from math import inf

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

class Node:
    def __init__(self):
        self.segment_max = -inf
        self.maxLeftSubArray = -inf
        self.maxRightSubArray = -inf
        self.segment_sum = 0

    def __repr__(self):
        return f"Node(max: {self.segment_max}, max_left: {self.maxLeftSubArray}, max_right: {self.maxRightSubArray}, sum: {self.segment_sum})"

class SegmentTree:
    def __init__(self, n: int, neutral: int):
        self.neutral = neutral
        self.size = 1
        self.n = n
        while self.size<n:
            self.size*=2
        self.tree = [Node() for _ in range(self.size*2)]

    def ascend(self, segment_idx: int) -> None:
        while segment_idx > 0:
            segment_idx -= 1
            segment_idx >>= 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            node = self.tree[segment_idx]
            node.segment_sum = self.tree[left_segment_idx].segment_sum + self.tree[right_segment_idx].segment_sum
            node.maxLeftSubArray = max(self.tree[left_segment_idx].maxLeftSubArray, self.tree[left_segment_idx].segment_sum + self.tree[right_segment_idx].maxLeftSubArray)
            node.maxRightSubArray = max(self.tree[right_segment_idx].maxRightSubArray, self.tree[right_segment_idx].segment_sum + self.tree[left_segment_idx].maxRightSubArray)
            node.segment_max = max(self.tree[left_segment_idx].segment_max, self.tree[right_segment_idx].segment_max, self.tree[left_segment_idx].maxRightSubArray + self.tree[right_segment_idx].maxLeftSubArray)
        
    def update(self, segment_idx: int, val: int) -> None:
        segment_idx += self.size - 1
        self.tree[segment_idx].segment_sum = val
        self.tree[segment_idx].segment_max = val
        self.tree[segment_idx].maxLeftSubArray = val
        self.tree[segment_idx].maxRightSubArray = val
        self.ascend(segment_idx)
            
    def query(self, left: int, right: int) -> int:
        stack = [(0, self.n, 0)]
        result = self.neutral
        while stack:
            # BOUNDS FOR CURRENT INTERVAL and idx for tree
            left_segment_bound, right_segment_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if left_segment_bound >= right or right_segment_bound <= left: continue
            # COMPLETE OVERLAP
            if left_segment_bound >= left and right_segment_bound <= right:
                result = max(result, self.tree[segment_idx].segment_max)
                continue
            # PARTIAL OVERLAP
            mid_point = (left_segment_bound + right_segment_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            stack.extend([(left_segment_bound, mid_point, left_segment_idx), (mid_point, right_segment_bound, right_segment_idx)])
        return result
    
    def __repr__(self) -> str:
        return f"array: {self.tree}"

def main():
    n, m = map(int, input().split())
    arr = list(map(int, input().split()))
    neutral = 0
    st = SegmentTree(n, neutral)
    for i, x in enumerate(arr):
        st.update(i, x)
    result = []
    for _ in range(m):
        idx, val = map(int, input().split())
        idx -= 1
        st.update(idx, val)
        result.append(st.query(0, n))
    return '\n'.join(map(str, result))

if __name__ == '__main__':
    print(main())