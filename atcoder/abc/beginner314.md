# Atcoder Beginner Contest 314

## A - 3.14

### Solution 1:  string slice

```py
def main():
    pi = "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"
    n = int(input())
    print(pi[:n + 2])
 
if __name__ == '__main__':
    main()
```

## B - Roulette

### Solution 1:  set 

```py
def main():
    n = int(input())
    bets = [set() for _ in range(n)]
    for i in range(n):
        c = int(input())
        b = map(int, input().split())
        bets[i].update(b)
    x = int(input())
    min_bets = 40
    betters = []
    for i in range(n):
        if x in bets[i] and len(bets[i]) < min_bets:
            min_bets = len(bets[i])
            betters = [i + 1]
        elif x in bets[i] and len(bets[i]) == min_bets: betters.append(i + 1)
    print(len(betters))
    print(*betters)
 
if __name__ == '__main__':
    main()
```

## C - Rotate Colored Subsequence

### Solution 1:  simulation + equivalence class

For each equivalence class add all the indices in that equivalence class and the nloop over the equivalence classes and rotate within them.  Cause you can completely identify one from the index of the characters in that class or color

```py
def main():
    n, m = map(int, input().split())
    s = input()
    colors = list(map(int, input().split()))
    res = [None] * n
    indices = [[] for _ in range(m + 1)]
    for i in range(n):
        indices[colors[i]].append(i)
    for i in range(1, m + 1):
        for j in range(len(indices[i])):
            res[indices[i][j]] = s[indices[i][j - 1]]
    print(''.join(res))
 
if __name__ == '__main__':
    main()
```

## D - LOWER

### Solution 1:  greedy + trick + set to upper or lower case and only perform necessary queries

```py
def main():
    n = int(input())
    s = input()
    q = int(input())
    lower = upper = False
    start = -1
    queries = []
    res = list(s)
    for i in range(q):
        t, x, c = input().split()
        t = int(t)
        x = int(x) - 1
        if t == 1: 
            res[x] = c
            queries.append((i, x, c))
        else:
            lower = True if t == 2 else False
            upper = False if t == 2 else True
            start = i
    if lower:
        res = list(map(lambda x: x.lower(), res))
    if upper:
        res = list(map(lambda x: x.upper(), res))
    for i, x, c in queries:
        if i <= start: continue
        res[x] = c
    print("".join(res))
    
if __name__ == '__main__':
    main()
```

## E - Roulettes

### Solution 1:  expected value + expected amount + contribution

```py

```

## F - A Certain Game

### Solution 1:  disjoint set union + directed rooted tree + arborescence + expected value + inverse modular

The expected value is either win or lose so the value is 1 or 0, so it turns out it will just be the sum of probabilities for a player and each time his team wins.

creating a directed graph that is also like a directed rooted tree, or a out-tree, also called an arborescence

![image](images/a_certain_game.png)

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

    """
    returns true if the nodes were not union prior. 
    """
    def union(self,i: int,j: int) -> bool:
        i, j = self.find(i), self.find(j)
        if i!=j:
            if self.size[i] < self.size[j]:
                i,j=j,i
            self.parent[j] = i
            self.size[i] += self.size[j]
            return True
        return False
    
    def __repr__(self) -> str:
        return f'parents: {[(i, parent) for i, parent in enumerate(self.parent)]}, sizes: {self.size}'
    
def mod_inverse(num, mod):
    return pow(num, mod - 2, mod)

from collections import defaultdict

def main():
    mod = 998244353
    n = int(input())
    dsu = UnionFind(n + 1)
    res = [0] * (n + 1)
    root = n + 1
    adj_list = defaultdict(list)
    nodes = {i: i for i in range(1, n + 1)}
    for _ in range(n - 1):
        p, q = map(int, input().split())
        root += 1
        sz_p, sz_q = dsu.size[dsu.find(p)], dsu.size[dsu.find(q)]
        sz = sz_p + sz_q
        inv = mod_inverse(sz, mod)
        adj_list[root].append((nodes[dsu.find(p)], sz_p * inv))
        adj_list[root].append((nodes[dsu.find(q)], sz_q * inv))
        dsu.union(p, q)
        nodes[dsu.find(p)] = root
    # dfs that computes the sum of probabilities going down each root
    stk = [(root, 0)]
    while stk:
        node, pr = stk.pop()
        if node <= n:
            res[node] = pr
        for child, wei in adj_list[node]:
            stk.append((child, (pr + wei) % mod))
    print(*res[1:])

if __name__ == '__main__':
    main()
```

## G - Amulets

### Solution 1: 

```py

```

