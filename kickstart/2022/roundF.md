# Google Kickstart 2022 Round F

## Summary

## Sort the Fabrics

### Solution 1:  custom sorts + zip + dataclass

```py
from dataclasses import make_dataclass
def main():
    n = int(input())
    Fabric = make_dataclass('Fabric', [('color', str), ('durability', int), ('unique', int)])
    fabrics = []
    for _ in range(n):
        c, d, u = input().split()
        fabrics.append(Fabric(c,int(d),int(u)))
    ada_sort = sorted(fabrics, key=lambda fabric: (fabric.color, fabric.unique))
    charles_sort = sorted(fabrics, key=lambda fabric: (fabric.durability, fabric.unique))
    return sum(x==y for x,y in zip(ada_sort, charles_sort))
    
if __name__ == '__main__':
    T = int(input())
    for t in range(1,T+1):
        print(f'Case #{t}: {main()}')
```

## Water Container System

### Solution 1:  bfs + undirected graph + counter

```py
from collections import Counter, deque
def main():
    n, q = map(int,input().split())
    adjList = [[] for _ in range(n+1)]
    for _ in range(n-1):
        u, v = map(int,input().split())
        adjList[u].append(v)
        adjList[v].append(u)
    levels = Counter() # count of water containers at each level
    queue = deque([(1,None)])
    lv = 0
    while queue:
        size = len(queue)
        for _ in range(size):
            node, parent_node = queue.popleft()
            levels[lv] += 1
            for nei_node in adjList[node]:
                if nei_node == parent_node: continue
                queue.append((nei_node, node))
        lv += 1
    for _ in range(q):
        input()
    lv = 0
    containers_filled = 0
    while q > 0:
        q -= levels[lv]
        if q >= 0:
            containers_filled += levels[lv]
            lv += 1
    return containers_filled
if __name__ == '__main__':
    T = int(input())
    for t in range(1,T+1):
        print(f'Case #{t}: {main()}')
```

## Story of Seasons

### Solution 1:

```py

```

## Scheduling a Meeting

### Solution 1:

```py

```