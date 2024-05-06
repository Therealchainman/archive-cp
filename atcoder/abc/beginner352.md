# Atcoder Beginner Contest 352

## E - Clique Connect 

### Solution 1:  greedy, minimum spanning tree, union find

```py
from collections import defaultdict
def main():
    n, m = map(int, input().split())
    dsu = UnionFind(n)
    queries = [None] * m
    sets = [None] * m
    for i in range(m):
        k, w = map(int, input().split())
        nodes = list(map(lambda x: int(x) - 1, input().split()))
        queries[i] = (w, i)
        sets[i] = nodes
    queries.sort()
    cost = 0
    for w, i in queries:
        unions = defaultdict(list)
        du = u = None
        for node in sets[i]:
            du = dsu.find(node)
            u = node
            unions[du].append(node)
        for s, vals in unions.items():
            if s == du: continue 
            dsu.union(u, vals[0])
            cost += w
    if all(dsu.find(i) == dsu.find(0) for i in range(n)):
        print(cost)
    else:
        print(-1)

if __name__ == '__main__':
    main()
```

## F - Estimate Order 

### Solution 1: 

```py

```

## G - Socks 3 

### Solution 1:  combinatorics, probability, expectation value, FFT convolution, product of polynomial

```py

```