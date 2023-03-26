# Leetcode Weekly Contest 284

## 

### Solution: 

```py

```

## 

### Solution: 

```py

```

## 

### Solution: 

```py

```

## 2203. Minimum Weighted Subgraph With the Required Paths

### Solution: forward + reverse graph dijkstra algorithm

```py
from heapq import heappush, heappop

class Solution:
    def minimumWeight(self, n: int, edges: List[List[int]], src1: int, src2: int, dest: int) -> int:
        Graphs = namedtuple('Graphs', ['forward', 'reverse'])
        graphs = Graphs([[] for _ in range(n)], [[] for _ in range(n)])
        for x, y, w in edges:
            graphs.forward[x].append((y,w))
            graphs.reverse[y].append((x,w))
            
        def dijkstra_forward(src):
            dist = defaultdict(lambda: math.inf)
            vis = set()
            heap = []
            heappush(heap, (0, src))
            dist[src] = 0
            while heap:
                cost, node = heappop(heap)
                if node in vis: continue
                for nei, nw in graphs.forward[node]:
                    vis.add(node)
                    ncost = cost + nw
                    if ncost < dist[nei]:
                        heappush(heap, (ncost, nei))
                        dist[nei] = ncost
            return dist
        
        def dijkstra_reverse(src):
            dist = defaultdict(lambda: math.inf)
            vis = set()
            heap = []
            heappush(heap, (0, src))
            dist[src] = 0
            while heap:
                cost, node = heappop(heap)
                if node in vis: continue
                for nei, nw in graphs.reverse[node]:
                    vis.add(node)                   
                    ncost = cost + nw
                    if ncost < dist[nei]:
                        heappush(heap, (ncost, nei))
                        dist[nei] = ncost
            return dist
        dest_dist = dijkstra_reverse(dest)
        if dest_dist[src1]==math.inf or dest_dist[src2]==math.inf: return -1
        src1_dist = dijkstra_forward(src1)
        src2_dist = dijkstra_forward(src2)
        best = math.inf
        for i in range(n):
            best = min(best, dest_dist[i]+src1_dist[i]+src2_dist[i])
        return best
```

Improved dijkstra implementation most likely

```py
from heapq import heappush, heappop

class Solution:
    def minimumWeight(self, n: int, edges: List[List[int]], src1: int, src2: int, dest: int) -> int:
        Graphs = namedtuple('Graphs', ['forward', 'reverse'])
        graphs = Graphs(defaultdict(list), defaultdict(list))
        for x, y, w in edges:
            graphs.forward[x].append((y,w))
            graphs.reverse[y].append((x,w))
            
        def dijkstra_forward(src):
            dist = {}
            heap = []
            heappush(heap, (0, src))
            while heap:
                cost, node = heappop(heap)
                if node in dist: continue
                dist[node] = cost
                for nei, nw in graphs.forward[node]:
                    ncost = cost + nw
                    if ncost < dist.get(nei,math.inf):
                        heappush(heap, (ncost, nei))
            return dist
        
        def dijkstra_reverse(src):
            dist = {}
            heap = []
            heappush(heap, (0, src))
            while heap:
                cost, node = heappop(heap)
                if node in dist: continue
                dist[node] = cost
                for nei, nw in graphs.reverse[node]:                
                    ncost = cost + nw
                    if ncost < dist.get(nei, math.inf):
                        heappush(heap, (ncost, nei))
            return dist
        dest_dist = dijkstra_reverse(dest)
        if src1 not in dest_dist or src2 not in dest_dist: return -1
        src1_dist = dijkstra_forward(src1)
        src2_dist = dijkstra_forward(src2)
        best = math.inf
        for i in range(n):
            best = min(best, dest_dist.get(i, math.inf)+src1_dist.get(i, math.inf)+src2_dist.get(i, math.inf))
        return best
```

For Numba experiment try this testcase 

5