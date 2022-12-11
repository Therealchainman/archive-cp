
from typing import List
import math
from collections import *

class SegmentTree:
    def __init__(self, n: int, neutral: int, initial_arr: List[int]):
        self.neutral = neutral
        self.size = 1
        self.n = n
        while self.size<n:
            self.size*=2
        self.nodes = [neutral for _ in range(self.size*2)] # distance to nearest leftmost equal element
        self.next = [neutral]*self.n # distance to nearest rightmost equal element
        self.build(initial_arr)

    def build(self, initial_arr: List[int]) -> None:
        nearLeft = {}
        for i, segment_idx in enumerate(range(self.n)):
            segment_idx += self.size - 1
            val = initial_arr[i]
            if val in nearLeft:
                left_index = nearLeft[val]
                self.nodes[segment_idx] = i - left_index
                self.next[left_index] = i
            nearLeft[val] = i
            self.ascend(segment_idx)
    
    def calc_op(self, x: int, y: int) -> int:
        return min(x, y)

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
        return f"nodes array: {self.nodes}, next array: {self.next}"

def main():
    n, m = map(int, input().split())
    arr = list(map(int, input().split()))
    queries = []
    answer = [-1]*m
    for i in range(m):
        left, right = map(lambda x: int(x)-1, input().split())
        queries.append((left, right, i))
    queries.sort() # offline query mode
    neutral = math.inf
    minSegTree = SegmentTree(n, neutral, arr)
    index = 0
    for left, right, idx in queries:
        while index < left:
            if minSegTree.next[index] != neutral:
                minSegTree.update(minSegTree.next[index], neutral)
            index += 1
        query_res = minSegTree.query(left, right+1)
        if query_res != neutral:
            answer[idx] = query_res

    return '\n'.join(map(str, answer))

if __name__ == '__main__':
    print(main())