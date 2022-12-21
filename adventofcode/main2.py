from parse import compile
import time
from z3.z3 import *

def main():
    with open('input.txt', 'r') as f:
        data = f.read().splitlines()
        pat = compile("{}: {} {} {}")
        pat2 = compile("{}: {:d}")
        s = Solver()
        arr = {}
        for line in data:
            if isinstance(pat.parse(line), type(None)):
                name, num = pat2.parse(line)
                arr[name] = Int(name)
                s.add(arr[name] == num)
            else:
                name, left, op, right = pat.parse(line)
                if name not in arr:
                    arr[name] = Int(name)
                if left not in arr:
                    arr[left] = Int(left)
                if right not in arr:
                    arr[right] = Int(right)
                if name == 'root':
                    s.add(arr[name] == (arr[left] <= arr[right]))
                else:
                    s.add(arr[name] == eval(f'arr[left] {op} arr[right]'))
        # I know that the left_result is decreasing function, 
        # f(x) compared to right_result
        # when to less it means you need to move to the left segment
        # when greater than it means you need to move to the right segment
        # FFFFFFFFFFFFTTTTTTTT, want to return the first T
        s.check()
        return s.model()[arr['root']]
        left, right = 0, 100_000_000_000_000
        while left < right:
            mid = (left + right)>>1
            s.add(arr['humn'] != s.model()[arr['humn']])
            s.add(arr['humn'] == mid)
            if s.check():
                right = mid
            else:
                left = mid + 1
        return left
        
if __name__ == '__main__':
    start_time = time.perf_counter()
    print(main())
    end_time = time.perf_counter()
    print(f'Time Elapsed: {end_time - start_time} seconds')