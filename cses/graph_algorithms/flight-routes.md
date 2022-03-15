# Flight Routes


```py
from collections import defaultdict
from math import inf
from heapq import heappush, heappop
import io,os
input = io.BytesIO(os.read(0,os.fstat(0).st_size)).readline
 
def dijkstra(src, k):
    dist = [inf]*(n+1)
    count = [0]*(n+1)
    ANS = []
    minheap = []
    heappush(minheap, (0, src))
    while minheap:
        distance, city = heappop(minheap)
        if count[city] >= k: continue
        if city == n:
            ANS.append(distance)
        count[city] += 1
        for ncity, nw in graph[city]:
            ndistance = distance + nw
            if count[ncity] < k:
                heappush(minheap, (ndistance, ncity))
    return " ".join(map(str, ANS))
 
if __name__ == '__main__':
    n, m, k = map(int, input().split())
    graph = defaultdict(list)
    for _ in range(m):
        a, b, c = map(int, input().split())
        graph[a].append((b,c))
    print(dijkstra(1, k))
```