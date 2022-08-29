import sys
from collections import Counter
problem = sys.argv[0].split('.')[0]
validation = ''

def main():
    n, m, q = map(int,f.readline().split())
    graph = [set() for _ in range(n+1)]
    capacity = Counter()
    for _ in range(m):
        u, v, c = map(int,f.readline().split())
        graph[u].add(v)
        graph[v].add(u)
        if u > v:
            u, v = v, u
        capacity[(u,v)] = c
    indirect_capacity = Counter()
    cache_indirect = set()
    maxFlight = []
    for _ in range(q):
        x, y = map(int,f.readline().split())
        if x > y:
            x, y = y, x
        if (x,y) not in cache_indirect: 
            for interm in graph[x]&graph[y]:
                if x > interm:
                    s1 = (interm, x)
                else:
                    s1 = (x, interm)
                if y > interm:
                    s2 = (interm, y)
                else:
                    s2 = (y, interm)
                c1, c2 = capacity[s1], capacity[s2]
                if c1 < c2:
                    indirect_capacity[(x,y)] += c1
                else:
                    indirect_capacity[(x,y)] += c2
            cache_indirect.add((x,y))
        maxFlight.append(2*capacity[(x,y)]+indirect_capacity[(x,y)])
    return ' '.join(map(str, maxFlight))

if __name__ == '__main__':
    result = []
    with open(f'inputs/{problem}_{validation}input.txt', 'r') as f:
        T = int(f.readline())
        result = ['']*T
        for t in range(1,T+1):
            result[t-1] = f'Case #{t}: {main()}'
    with open(f'outputs/{problem}_output.txt', 'w') as f:
        f.write('\n'.join(result))