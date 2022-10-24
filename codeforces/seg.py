from math import inf
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
            self.ascend(segment_idx, 1)

    def modify_op(self, x: int, y: int) -> int:
        return x + y
    
    def calc_op(self, x: int, y: int, z: int) -> int:
        return (x + y)*z

    def ascend(self, segment_idx: int, segment_len: int) -> None:
        while segment_idx > 0:
            segment_idx -= 1
            segment_idx >>= 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            segment_len <<= 1
            self.values[segment_idx] = self.calc_op(self.values[left_segment_idx], self.values[right_segment_idx], segment_len)
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
                segments.append((segment_idx, min(segment_right_bound, right) - max(segment_left_bound, left)))
                continue
            # PARTIAL OVERLAP
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
        for segment_idx, segment_len in segments:
            self.ascend(segment_idx, segment_len)
            
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
                result = self.calc_op(result, modified_val, min(segment_right_bound, right) - max(segment_left_bound, left))
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