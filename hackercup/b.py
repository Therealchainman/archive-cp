import sys

sys.setrecursionlimit(10 ** 6)

# name = "jacks_candy_shop_sample_input.txt"
name = "jacks_candy_shop_input.txt"

sys.stdout = open(f"outputs/{name}", "w")
sys.stdin = open(f"inputs/{name}", "r")

from collections import deque
import heapq

def main():
    N, M, A, B = map(int, input().split())
    parents = [-1] * N
    indegrees = [0] * N
    adj = [[] for _ in range(N)]
    indegrees[0] = 1
    for i in range(1, N):
        parents[i] = int(input())
        adj[i].append(parents[i])
        adj[parents[i]].append(i)
        indegrees[i] += 1
        indegrees[parents[i]] += 1
    candies = [0] * N
    for i in range(M):
        candy = (A * i + B) % N
        candies[candy] += 1
    sz, heavy = [0] * N, [-1] * N
    def dfs(u, p):
        sz[u] = 1
        heaviest_child = 0
        for v in adj[u]:
            if v == p: continue
            csz = dfs(v, u)
            if csz > heaviest_child:
                heavy[u] = v
                heaviest_child = csz
            sz[u] += csz
        return sz[u]
    dfs(0, -1)
    heaps = [[] for _ in range(N)]
    queue = deque()
    for i in range(N):
        if indegrees[i] == 1:
            queue.append(i)
    res = 0
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v == parents[u]: 
                indegrees[v] -= 1
                if indegrees[v] == 1:
                    queue.append(v)
                continue
            if v == heavy[u]:
                heaps[u] = heaps[v]
        heapq.heappush(heaps[u], -u)
        for v in adj[u]:
            if v == parents[u]: continue
            if v == heavy[u]: continue
            while heaps[v]:
                heapq.heappush(heaps[u], heapq.heappop(heaps[v]))
        while candies[u] > 0 and heaps[u]:
            res += abs(heapq.heappop(heaps[u]))
            candies[u] -= 1
    return res

if __name__ == '__main__':
    T = int(input())
    for t in range(1, T + 1):
        print(f"Case #{t}: {main()}")