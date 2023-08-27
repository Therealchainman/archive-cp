# Binary Jumping

Binary jumping is an algorithm that works for finding kth ancestor, LCA of trees and also for finding the kth successor in a functional/successor graph.

## kth successor in functional graph

Example code for using it to find the kth successor in a functional graph. 

```py
succ = [[0] * n for _ in range(LOG)]
succ[0] = [s - 1 for s in successor]
for i in range(1, LOG):
    for j in range(n):
        succ[i][j] = succ[i - 1][succ[i - 1][j]]
for x, k in queries:
    x -= 1
    for i in range(LOG):
        if (k >> i) & 1:
            x = succ[i][x]
    print(x + 1)
```

## sum from node to kth successor in functional graph

With the addition of the idea of sparse table for range sum queries, you can modify it to work with a successor graph and calculate range sums whenever you traverse k edges starting from any node. 

```py
LOG = 35
succ = [[0] * n for _ in range(LOG)]
succ[0] = successor[:]
st = [[0] * n for _ in range(LOG)]
st[0] = list(range(n))
for i in range(1, LOG):
    for j in range(n):
        st[i][j] = st[i - 1][j] + st[i - 1][succ[i - 1][j]]
        succ[i][j] = succ[i - 1][succ[i - 1][j]]
res = 0
for j in range(n):
    sum_ = 0
    for i in range(LOG):
        if ((k + 1) >> i) & 1:
            sum_ += st[i][j]
            j = succ[i][j]
    res = max(res, sum_)
return res
```