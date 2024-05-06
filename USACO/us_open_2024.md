# US OPEN 2024

## Painting Fence Posts

### Solution 1:  prefix sum, geometry, line segments, binary search, rectilinear polygon, sort, undirected graph, difference array to increment interval, 

```py
from collections import defaultdict
import bisect
def manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)
def get_fence_post(x1, y1, x_points, y_points, ind):
    P = len(ind)
    pos = bisect.bisect_left(x_points[x1], (y1, -1))
    if pos < len(x_points[x1]):
        if x_points[x1][pos][0] == y1: 
            _, s = x_points[x1][pos]
            return s
        if pos & 1:
            _, s = x_points[x1][pos - 1]
            _, e = x_points[x1][pos]
            if (ind[s] + 1) % P != ind[e]:
                s, e = e, s
            return s
    pos = bisect.bisect_left(y_points[y1], (x1, -1))
    if pos < len(y_points[y1]):
        if y_points[y1][pos][0] == x1: 
            _, s = y_points[y1][pos]
            return s
        if pos & 1:
            _, s = y_points[y1][pos]
            _, e = y_points[y1][pos ^ 1]
            if (ind[s] + 1) % P != ind[e]: # wrong order, cause s should be one behind e
                s, e = e, s
            return s
    assert(False)
def get_dist(i, j, psum):
    return psum[j] - (psum[i - 1] if i > 0 else 0)
def main():
    N, P = map(int, input().split())
    points = [None] * P
    x_points = defaultdict(list)
    y_points = defaultdict(list)
    for i in range(P):
        x, y = map(int, input().split())
        points[i] = (x, y)
        x_points[x].append((y, i))
        y_points[y].append((x, i))
    adj = [[] for _ in range(P)]
    for x, vals in x_points.items():
        vals.sort()
        for i in range(len(vals)):
            adj[vals[i][1]].append(vals[i ^ 1][1])
    for y, vals in y_points.items():
        vals.sort()
        for i in range(len(vals)):
            adj[vals[i][1]].append(vals[i ^ 1][1])
    path = [0] # path through the points
    ind = {0: 0}
    while True:
        for v in adj[path[-1]]:
            if len(path) > 1 and path[-2] == v: continue
            ind[v] = len(path)
            path.kxc(v)
            break
        if path[-1] == 0: break
    path.pop()
    ind[0] = 0
    dist_points = [0] * (P + 1)
    for i in range(1, P + 1):
        x1, y1 = points[path[i - 1]]
        x2, y2 = points[path[i % P]]
        dist_points[i] = manhattan_distance(x1, y1, x2, y2) + dist_points[i - 1]
    per = dist_points[-1]
    diff = [0] * (P + 1)
    for i in range(N):
        x1, y1, x2, y2 = map(int, input().split())
        p1 = get_fence_post(x1, y1, x_points, y_points, ind) # start post start point
        p2 = get_fence_post(x2, y2, x_points, y_points, ind) # end post start point
        pc1, pc2 = ind[p1], ind[p2]
        along1 = manhattan_distance(x1, y1, *points[p1])
        along2 = manhattan_distance(x2, y2, *points[p2])
        dist1 = dist_points[pc1] + along1
        dist2 = dist_points[pc2] + along2
        # make sure dist1 < dist2
        if dist1 > dist2:
            dist1, dist2 = dist2, dist1
            p1, p2 = p2, p1
            pc1, pc2 = pc2, pc1
            along1, along2 = along2, along1
        d = dist2 - dist1
        if 2 * d <= per:
            if along1 > 0: pc1 += 1
            diff[pc1] += 1
            diff[pc2 + 1] -= 1
        else:
            if along2 > 0: pc2 += 1
            diff[pc2] += 1 # right segment
            diff[0] += 1 # left segment
            diff[pc1 + 1] -= 1 # end left segment
    ans = [0] * P
    psum = 0
    for i in range(P):
        psum += diff[i]
        ans[path[i]] = psum
    print("\n".join(map(str, ans)))

if __name__ == '__main__':
    main()
```