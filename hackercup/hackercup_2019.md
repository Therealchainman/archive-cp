# Meta Hacker Cup 2019

# Round 2

## Problem A: On the Run

```py
def main():
    man_dist = lambda r1, c1, r2, c2: abs(r1 - r2) + abs(c1 - c2)
    R, C, K = map(int, input().split())
    A, B = map(int, input().split())
    res = 1
    for _ in range(K):
        r, c = map(int, input().split())
        dist = man_dist(A, B, r, c)
        res &= dist % 2 == 0
    if K == 1:
        return "N"
    return "Y" if res else "N"

if __name__ == '__main__':
    T = int(input())
    for t in range(1, T + 1):
        print(f"Case #{t}: {main()}")
```

## Problem B: Bitstrings as a Service

```py
from collections import defaultdict

class UnionFind:
    def __init__(self, n: int):
        self.size = [1]*n
        self.parent = list(range(n))
    
    def find(self,i: int) -> int:
        while i != self.parent[i]:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i
    
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

def main():
    N, M = map(int, input().split())
    dsu = UnionFind(N)
    for i in range(M):
        x, y = map(int, input().split())
        x -= 1
        y -= 1
        d = y - x + 1
        for i in range(d // 2):
            dsu.union(x + i, y - i)
    vis = [False] * N
    components = defaultdict(list)
    arr = []
    for i in range(N):
        root = dsu.find(i)
        if vis[root]: continue
        vis[root] = True
        components[dsu.size[root]].append(root)
        arr.append(dsu.size[root])
    # partition problem
    half_sum = N // 2
    dp = [[False] * (half_sum + 1) for _ in range(len(arr))]
    dp[0][0] = True
    if arr[0] <= half_sum:
        dp[0][arr[0]] = True
    for i in range(1, len(arr)):
        for j in range(half_sum + 1):
            dp[i][j] = dp[i - 1][j] or (j >= arr[i] and dp[i - 1][j - arr[i]])
    while not dp[-1][half_sum]:
        half_sum -= 1
    # backtrack through the dynamic table to determine the elements picked
    i = len(arr) - 1
    partition1, partition2 = set(), set()
    for i in reversed(range(len(arr))):
        comp = components[arr[i]].pop()
        if i > 0 and half_sum > 0 and not dp[i - 1][half_sum]:
            partition1.add(comp)
            half_sum -= arr[i]
        elif i == 0 and half_sum > 0 and dp[i][half_sum]:
            partition1.add(comp)
            half_sum -= arr[i]
        else:
            partition2.add(comp)
    res = [0] * N
    for i in range(N):
        if dsu.find(i) in partition2:
            res[i] = 1
    return "".join(map(str, res))

if __name__ == '__main__':
    T = int(input())
    for t in range(1, T + 1):
        print(f"Case #{t}: {main()}")
```

## Problem D: Seafood

dynamic programming, greedy, binary search, monotonic stack, motonoic decreasing sequence

This probably doesn't work, it needs another way to choose the optimal sequence of clams to pick up. There is a trick to it but it is hard to figure it out. Actually this gets accepted lol.  Still not understanding, just added that cnt = 100, under the assumption that looking too far back becomes suboptimal and it is only way it could possibly be a nlogn solution about and not O(n^2)



```py
import math

def main():
    N = int(input())
    P1, P2, Ap, Bp, Cp, Dp = map(int, input().split())
    H1, H2, Ah, Bh, Ch, Dh = map(int, input().split())
    O = input()
    data = [None] * N
    for i in range(N):
        if i == 0:
            pos, har = P1, H1
        elif i == 1:
            pos, har = P2, H2
        else:
            pos = (Ap * data[i - 2][0] + Bp * data[i - 1][0] + Cp) % Dp + 1
            har = (Ah * data[i - 2][1] + Bh * data[i - 1][1] + Ch) % Dh + 1
        data[i] = (pos, har, O[i])
    data.sort()
    positions, hardness, O = zip(*data)
    if max((h for i, h in enumerate(hardness) if O[i] == "C"), default = 0) >= max((h for i, h in enumerate(hardness) if O[i] == "R"), default = 0): return -1
    def bsearch(target, stack):
        left, right = 0, len(stack) - 1
        while left < right:
            mid = (left + right + 1) >> 1
            if hardness[stack[mid]] > target: left = mid
            else: right = mid - 1
        return left
    last_clam = first_clam = N
    clam_hardness = rock_hardness = 0
    rstack, lstack, clams = [], [], []
    # construct left stack and find the first clam
    for i in range(N):
        if O[i] == "C":
            first_clam = i
            break
        else:
            while lstack and hardness[lstack[-1]] <= hardness[i]:
                lstack.pop()
            lstack.append(i)
    # construct the right stack and find the last clam
    for i in reversed(range(N)):
        if O[i] == "C":
            last_clam = i
            break
        else:
            while rstack and hardness[rstack[-1]] <= hardness[i]:
                rstack.pop()
            rstack.append(i)
    # construct the clams with monotonically decreasing hardness from the last clam and those that will not be taken care of from a rock before the last clam
    # that is the if a rock with greater hardness is between a clam and the last clam.
    for i in range(last_clam, first_clam - 1, -1):
        if O[i] == "C":
            if hardness[i] > clam_hardness and hardness[i] >= rock_hardness: clams.append(i)
            clam_hardness = max(clam_hardness, hardness[i])
        else:
            rock_hardness = max(rock_hardness, hardness[i])
    """
    dynamic programming over the clams
    """
    clams = clams[::-1]
    nc = len(clams)
    # nearest rock with greater hardness but to the left of the clam
    nearest = [-1] * nc
    j = 0
    for i in range(first_clam, last_clam + 1):
        if O[i] == "C":
            if j < len(clams) and i == clams[j] and lstack and hardness[lstack[0]] > hardness[clams[j]]:
                k = bsearch(hardness[clams[j]], lstack)
                nearest[j] = lstack[k]
            j += i == clams[j]
        else:
            while lstack and hardness[lstack[-1]] <= hardness[i]:
                lstack.pop()
            lstack.append(i)
    rnearest = [-1] * nc
    j = nc - 1
    for i in range(last_clam, first_clam - 1, -1):
        if O[i] == "C":
            # print("j", j)
            if j >= 0 and i == clams[j] and rstack and hardness[rstack[0]] > hardness[clams[j]]:
                k = bsearch(hardness[clams[j]], rstack)
                rnearest[j] = rstack[k]
            j -= i == clams[j]
        else:
            while rstack and hardness[rstack[-1]] <= hardness[i]:
                rstack.pop()
            rstack.append(i)
    dp = [math.inf] * (nc + 1)
    dp[0] = 0
    i = 1
    for k in range(first_clam, last_clam):
        if O[k] == "C":
            if clams[i - 1] != k: continue
            cnt = 0
            for j in range(i - 1, -1, -1):
                cnt += 1
                if cnt == 100: break
                if nearest[j] == -1: continue
                n = nearest[j]
                dp[i] = min(dp[i], dp[j] + 2 * (positions[clams[i - 1]] - positions[n]))
            i += 1
    # case when moving to the left
    # ****L***R***L****
    # ------------>
    #         <----
    # case when moving to the right
    # *****L*****L*****R
    # ----------------->
    # print("clams", clams, "dp", dp, "lstack", lstack, "rstack", rstack)
    # print("rnearest", rnearest)
    for j in range(nc - 1, -1, -1):
        if nearest[j] != -1:
            dp[-1] = min(dp[-1], dp[j] + 2 * positions[clams[-1]] - positions[nearest[j]])
        if rnearest[j] != -1:
            dp[-1] = min(dp[-1], dp[j] + positions[rnearest[j]])
    return dp[-1]

if __name__ == '__main__':
    T = int(input())
    for t in range(1, T + 1):
        print(f"Case #{t}: {main()}")
```

##

```py

```

##

```py

```

##

```py

```

##

```py

```