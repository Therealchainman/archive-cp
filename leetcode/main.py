def prime_sieve(lim):
    primes = [1] * lim
    primes[0] = primes[1] = 0
    p = 2
    while p * p <= lim:
        if primes[p]:
            for i in range(p * p, lim, p):
                primes[i] = 0
        p += 1
    return primes

class Solution:
    def countPaths(self, n: int, edges: List[List[int]]) -> int:
        ps = prime_sieve(n + 1)
        adj_list = [[] for _ in range(n + 1)]
        for u, v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
        res = 0
        def dfs(u, parent, pc, before, after):
            nonlocal res
            if pc:
                res += before
            children = 0
            for v in adj_list[u]:
                if v == parent: continue
                children += 1
                npc = pc + ps[v]
                if npc == 2:
                    nbefore = after
                    nafter = 1
                    npc = 1
                else:
                    nbefore = before + (npc == 0)
                    nafter = after + (npc == 1)
                dfs(v, u, npc, nbefore, nafter)
        dfs(1, -1, 0, 0, 0)
        return res
"""
5
[[1,2],[1,3],[2,4],[2,5]]
6
[[1,2],[1,3],[2,4],[3,5],[3,6]]
18
[[1,2],[2,3],[3,5],[5,7],[7,9],[9,10],[10,13],[1,4],[4,6],[6,8],[8,11],[11,12],[12,14],[14,15],[15,16],[16,17],[17,18]]
4
[[1,2],[4,1],[3,4]]
9
[[7,4],[3,4],[5,4],[1,5],[6,4],[9,5],[8,7],[2,8]]
"""