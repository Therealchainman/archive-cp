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
                s.add(arr[name] == eval(f'arr[left] {op} arr[right]'))
        s.check()
        return s.model()[arr['root']]

if __name__ == '__main__':
    start_time = time.perf_counter()
    print(main())
    end_time = time.perf_counter()
    print(f'Time Elapsed: {end_time - start_time} seconds')