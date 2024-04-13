# Maximum Manhattan Distance of pair of Points

This can be solved in O(n) time

example where it uses the smax - smin, and dmax - dmin to find the two index that lead to maximum manhattan distance


```py
def max_manhattan_distance(points, remove = -1):
    smin = dmin = math.inf
    smax = dmax = -math.inf
    smax_i = smin_i = dmax_i = dmin_i = None
    for i, (x, y) in enumerate(points):
        if remove == i: continue
        s = x + y
        d = x - y
        if s > smax:
            smax = s
            smax_i = i
        if s < smin:
            smin = s
            smin_i = i
        if d > dmax:
            dmax = d
            dmax_i = i
        if d < dmin:
            dmin = d
            dmin_i = i
    return (smax_i, smin_i) if smax - smin >= dmax - dmin else (dmax_i, dmin_i)
```