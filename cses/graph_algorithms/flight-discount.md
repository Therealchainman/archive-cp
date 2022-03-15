# Flight Discount


```py
from collections import defaultdict
from math import inf
from heapq import heappush, heappop
import io
import os
input = io.BytesIO(os.read(0,os.fstat(0).st_size)).readline
 
def dijkstra_forward(src):
    dist = [inf]*(n+1)
    minheap = []
    heappush(minheap, (0, src))
    while minheap:
        distance, city = heappop(minheap)
        if dist[city] < inf: continue
        dist[city] = distance
        for ncity, nw in graph[city]:
            ndistance = distance + nw
            if ndistance < dist[ncity]:
                heappush(minheap, (ndistance, ncity))
    return dist

def dijkstra_reverse(src):
    dist = [inf]*(n+1)
    minheap = []
    heappush(minheap, (0, src))
    while minheap:
        distance, city = heappop(minheap)
        if dist[city] < inf: continue
        dist[city] = distance
        for ncity, nw in rgraph[city]:
            ndistance = distance + nw
            if ndistance < dist[ncity]:
                heappush(minheap, (ndistance, ncity))
    return dist
 
if __name__ == '__main__':
    n, m = map(int, input().split())
    graph = defaultdict(list)
    rgraph = defaultdict(list)
    for _ in range(m):
        a, b, c = map(int, input().split())
        graph[a].append((b,c))
        rgraph[b].append((a,c))
    dist1 = dijkstra_forward(1)
    distn = dijkstra_reverse(n)
    best_discount_route = inf
    for city in range(1,n+1):
        for ncity, nw in graph[city]:
            if dist1[city] == inf or distn[ncity] == inf: continue
            cost = dist1[city] + nw//2 + distn[ncity]
            best_discount_route = min(best_discount_route, dist1[city] + nw//2 + distn[ncity])
    print(best_discount_route)
```