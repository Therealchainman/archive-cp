# Atcoder Beginner Contest 357

## E - Reachability in Functional Graph 

### Solution 1:  union find, topological order, cycle detection, functional graph, dynamic programming

```py
class UnionFind:
    def __init__(self, n: int):
        self.size = [1]*n
        self.parent = list(range(n))
    
    def find(self,i: int) -> int:
        while i != self.parent[i]:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def same(self,i: int,j: int) -> bool:
        i, j = self.find(i), self.find(j)
        if i!=j:
            if self.size[i] < self.size[j]:
                i,j=j,i
            self.parent[j] = i
            self.size[i] += self.size[j]
            return False
        return True

from collections import deque

def main():
    n = int(input())
    indegrees = [0] * n
    edges = list(map(lambda x: int(x) - 1, input().split()))
    dsu = UnionFind(n)
    for i, x in enumerate(edges):
        dsu.same(i, x)
        indegrees[x] += 1
    cycle, vis, comp = [0] * n, [0] * n, [[] for _ in range(n)]
    comp_cycle = [0] * n
    dp = [0] * n
    ans = 0
    def search(u):
        parent = {u: None}
        is_cycle = False
        while True:
            vis[u] = 1
            v = edges[u]
            if v in parent: 
                is_cycle = True
                break
            if vis[v]: break
            parent[v] = u
            u = v
        if is_cycle:
            crit_point = parent[edges[u]]
            cnt = 0
            while u != crit_point:
                cycle[u] = 1
                cnt += 1
                u = parent[u]
            return cnt
        return 0
    def travel(nodes):
        nonlocal ans
        res = 0
        q = deque()
        for i in nodes:
            if indegrees[i] == 0: q.append(i)
        while q:
            u = q.popleft()
            vis[u] = 1
            if cycle[u]:
                res += dp[u]
                continue
            dp[u] += 1
            ans += dp[u]
            v = edges[u]
            if cycle[v]:
                res += dp[u]
                continue
            dp[v] += dp[u]
            indegrees[v] -= 1
            if indegrees[v] == 0:
                q.append(v)
        return res
    for i in range(n):
        if vis[i]: continue
        cycle_len = search(i)
        if cycle_len > 0: comp_cycle[dsu.find(i)] = cycle_len
        comp[dsu.find(i)].append(i)
        ans += cycle_len * cycle_len
    vis = [0] * n
    for i in range(n):
        if vis[i] or cycle[i]: continue
        length = travel(comp[dsu.find(i)])
        ans += length * comp_cycle[dsu.find(i)]
    print(ans)
    
if __name__ == '__main__':
    main()
```

## 

### Solution 1: 

```cpp

```

## G - Stair-like Grid 

### Solution 1: 

```cpp

```