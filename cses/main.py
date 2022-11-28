import os,sys
from io import BytesIO, IOBase
from typing import *
from math import *

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

from math import inf
class SegmentTree:
    def __init__(self, n: int, neutral: int, initial: int):
        self.neutral = neutral
        self.size = 1
        self.n = n
        self.initial_val = initial
        while self.size<n:
            self.size*=2
        self.operations = [initial for _ in range(self.size*2)]
        self.values = [neutral for _ in range(self.size*2)]

    def modify_op(self, x: int, y: int, segment_len: int) -> int:
        return x + y*segment_len
    
    def calc_op(self, x: int, y: int) -> int:
        return x + y

    def ascend(self, segment_idx: int) -> None:
        while segment_idx > 0:
            segment_idx -= 1
            segment_idx >>= 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            segment_len = right_segment_idx - left_segment_idx
            self.values[segment_idx] = self.calc_op(self.values[left_segment_idx], self.values[right_segment_idx])
            self.values[segment_idx] = self.modify_op(self.values[segment_idx], self.operations[segment_idx], segment_len)
        
    def update(self, left: int, right: int, val: int) -> None:
        stack = [(0, self.size, 0)]
        segments = []
        while stack:
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # COMPLETE OVERLAP
            if segment_left_bound >= left and segment_right_bound <= right:
                segment_len = segment_right_bound - segment_left_bound
                self.operations[segment_idx] = self.modify_op(self.operations[segment_idx], val, 1)
                self.values[segment_idx] = self.modify_op(self.values[segment_idx], val, segment_len)
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
                segment_len = segment_right_bound - segment_left_bound
                modified_val = self.modify_op(self.values[segment_idx], operation_val, segment_len)
                result = self.calc_op(result, modified_val)
                continue
            # PARTIAL OVERLAP
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            segment_len = min(right, segment_right_bound) - max(left, segment_left_bound)
            operation_val = self.modify_op(operation_val, self.operations[segment_idx], segment_len)
            stack.extend([(mid_point, segment_right_bound, right_segment_idx, operation_val), (segment_left_bound, mid_point, left_segment_idx, operation_val)])
        return result
    
    def __repr__(self) -> str:
        return f"operations array: {self.operations}, values array: {self.values}"

def main():
    n, m = map(int, input().split())
    neutral = 0
    initial = 0
    segTree = SegmentTree(n, neutral, initial)
    results = []
    for _ in range(m):
        query = list(map(int, input().split()))
        if query[0] == 1:
            # range update
            _, left, right, val = query
            segTree.update(left, right, val)
        else:
            # point query
            _, left, right = query
            results.append(segTree.query(left, right))
    return '\n'.join(map(str,results))

if __name__ == '__main__':
    print(main())