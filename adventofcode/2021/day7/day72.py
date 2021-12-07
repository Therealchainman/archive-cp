from collections import namedtuple
with open("inputs/input.txt", "r") as f:
    lines = list(map(int, f.read().split(',')))
    mx = max(lines)
    mn = min(lines)
    lo = mn
    hi = mx
    res = 100000000000000
    print(mx)
    for i in range(lo, hi+1):
        s = 0
        for crab in lines:
            n = abs(crab-i)
            k = n*(n+1)//2
            s += k
        res = min(res, s)
    print(res)