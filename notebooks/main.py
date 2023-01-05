
import math
from typing import List

def bellmanFord(n: int, src: int, dest: int, edges: List[List[int]]) -> List[int]:
    dist = [math.inf]*n
    # parents = [-1]*n
    dist[src] = 0
    for _ in range(n - 1):
        any_relaxed = False
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                any_relaxed = True
                dist[v] = dist[u] + w
                # parents[v] = u
        if not any_relaxed: break
    # check for any negative cycles
    for u, v, w in edges:
        if dist[v] > dist[u] + w: return []
    return dist

def main(t: int) -> bool:
    n = int(input())
    if n == 0: return False
    edges = []*n*n
    name = [None]*n
    for i in range(n):
        line = input().split()
        name[i] = line[0]
        for j, w in enumerate(map(int, line[1:])):
            edges.append((i, j, w))
    q = int(input())
    print(f'Case #{t}:')
    for _ in range(q):
        src, dest = map(int, input().split())
        dist = bellmanFord(n, src, edges)
        print(dist)
        if not dist:
            print('NEGATIVE CYCLE')
        elif dist[dest] == math.inf:
            print(f'{name[src]}-{name[dest]}: NOT REACHABLE')
        else:
            print(f'{name[src]}-{name[dest]}: {dist[dest]}')
    return True
if __name__ == '__main__':
    t = 1
    while True:
        if not main(t): break
        t += 1


Negative weight cycle

