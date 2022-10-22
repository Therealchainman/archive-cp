from math import inf
from typing import Callable, Type, List
from copy import deepcopy
class Node:
    def __init__(self, even_val: int = 0, odd_val: int = 0):
        self.even_sum = even_val
        self.odd_sum = odd_val
    def __repr__(self) -> str:
        return f"even: {self.even_sum}, odd: {self.odd_sum}"
class SegmentTree:
    def __init__(self, n: int, neutral: Type[Node], func: Callable[[int, int], int]):
        self.neutral = neutral
        self.size = 1
        self.func = func
        self.n = n
        while self.size<n:
            self.size*=2
        self.tree = [deepcopy(neutral) for _ in range(self.size*2)]
        
    def update(self, idx: int, val: int, update_func: Callable[[List[List[int]],int,int], None]) -> None:
        is_odd = idx&1
        idx += self.size - 1
        update_func(self.tree, idx, val, is_odd)
        while idx > 0:
            idx -= 1
            idx >>= 1
            self.tree[idx] = self.func(self.tree[2*idx+1], self.tree[2*idx+2])
            
    def query(self, l: int, r: int) -> int:
        stack = [(0, self.size, 0)]
        result = self.neutral
        while stack:
            # BOUNDS FOR CURRENT INTERVAL and idx for tree
            left_bound, right_bound, idx = stack.pop()
            # OUT OF BOUNDS
            if left_bound >= r or right_bound <= l: continue
            # CHECK IF CURRENT BOUNDS ARE WITHIN THE l and r
            if left_bound >= l and right_bound <= r:
                result = self.func(result, self.tree[idx])
                continue
            mid = (left_bound + right_bound)>>1
            stack.extend([(mid, right_bound, 2*idx+2), (left_bound, mid, 2*idx+1)])
        return result
    
    def __repr__(self) -> str:
        return f"array: {self.tree}"

def update_func(tree: List[Node], idx: int, val: int, is_odd: bool) -> None:
    if is_odd:
        tree[idx].odd_sum = val
    else:
        tree[idx].even_sum = val

def sum_func(node_left: Type[Node], node_right: Type[Node]) -> Node:
    return Node(node_left.even_sum+node_right.even_sum, node_left.odd_sum+node_right.odd_sum)

def main():
    n = int(input())
    arr = map(int,input().split())
    m = int(input())
    neutral = Node(0)
    sumSeg = SegmentTree(n, neutral, sum_func)
    results = []
    for i, num in enumerate(arr):
        sumSeg.update(i, num, update_func)
    for _ in range(m):
        type_, x, y = map(int,input().split())
        if type_ == 1:
            node = sumSeg.query(x-1,y)
            result = node.even_sum - node.odd_sum if x&1 else node.odd_sum - node.even_sum
            results.append(result)
        else:
            sumSeg.update(x-1, y, update_func)
    return '\n'.join(map(str,results))

if __name__ == '__main__':
    print(main())