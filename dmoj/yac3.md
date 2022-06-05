# Yet Another Contest 3



```py
from functools import reduce
def main():
    num_shells, num_swaps, start_pos, end_pos = map(int, input().split())
    # BUILDING THE INCOMPLETE SWAPS
    swaps = []
    for _ in range(num_swaps):
        line = input()
        if line.find('-1') == 0:
            swaps.append(-1)
        else:
            u, v = map(int,line.split())
            swaps.append((u,v))
    # FINDING THE INTERMEDIATE POSITION AND INDEX FOR WHERE NEED TO PUT THE STONE TO FIND IT IN THE END_POS
    intermediate_pos = end_pos
    intermediate_index = 0
    for i in range(len(swaps))[::-1]:
        swap = swaps[i]
        if swap == -1:
            intermediate_index = i
            break
        elif intermediate_pos in swap:
            intermediate_pos = reduce(lambda prev, cur: prev^cur, swap, intermediate_pos)
    # FUNCTION TO GET THE NEIGHBORS OF CURRENT STONE LOCATION
    def get_neighbors(stone_pos):
        u, v = stone_pos-2, stone_pos-1
        while u < 1:
            u += 1 
            v += 1
        if u == stone_pos:
            u += 1
            v += 1
        if v == stone_pos:
            v += 1
        return u, v
    # BUILDING THE CORRECT SWAPS FOR THE STONE TO GO FROM START_POS TO END_POS
    corrected_swaps = []
    current_pos = start_pos
    for i, swap in enumerate(swaps):
        if i == intermediate_index:
            if current_pos == intermediate_pos:
                u, v = get_neighbors(current_pos)
                corrected_swaps.append((u,v))
            else:
                corrected_swaps.append((current_pos, intermediate_pos))
            current_pos = intermediate_pos
        elif swap == -1:
            u, v = get_neighbors(current_pos)
            corrected_swaps.append((u,v))
        else:
            corrected_swaps.append(swap)
            if current_pos in swap:
                current_pos = reduce(lambda prev, cur: prev^cur, swap, current_pos)
    return '\n'.join(map(lambda x: f'{x[0]} {x[1]}', corrected_swaps))

if __name__ == '__main__':
    print(main())
```


```py
from collections import defaultdict, deque
from itertools import product
from math import inf
def main():
    N = int(input())
    edges = [tuple(map(int,input().split())) for _ in range(N-1)]
    # BUILD THE ADJACENCY LIST FOR THE GRAPH
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    # MATRIX WITH SHORTEST PATH FROM NODE I TO NODE J
    dist = [[inf]*(N+1) for _ in range(N+1)]
    # COMPUTE SHORTEST PATH TO ALL NODES FROM A SINGLE SOURCE NODE WITH BFS
    def single_source_shortest_path(src):
        queue = deque([(src,0)])
        dist[src][src] = 0
        while queue:
            node, cost = queue.popleft()
            for nei_node in graph[node]:
                ncost = cost + 1
                if ncost < dist[src][nei_node]:
                    dist[src][nei_node] = ncost
                    queue.append((nei_node, ncost))
    for i in range(1,N+1):
        single_source_shortest_path(i)
    meeting_counts = [0]*(N+1)
    # FOR EACH SCENARIO COMPUTE FIND THE OPTIMAL MEETING LOCATION
    for person1, person2, person3 in product(range(1,N+1), repeat=3):
        min_dist, meeting_node = inf, 0
        for node in range(1,N+1):
            cur_dist = dist[person1][node]+dist[person2][node]+dist[person3][node]
            if cur_dist < min_dist:
                min_dist = cur_dist
                meeting_node = node
        meeting_counts[meeting_node] += 1
    return ' '.join(map(str, meeting_counts[1:]))
        
if __name__ == '__main__':
    print(main())
```



```py
from collections import defaultdict, deque
from itertools import product
from math import inf
def main():
    N, M = map(int,input().split())
    queries = sorted([tuple(map(int,input().split())) for _ in range(M)], key=lambda x: (x[1],x[2]))
    curr_depth = start_depth = 1
    current_index = 1
    for t, l, r in queries:
        if t == 1: # increasing
            while current_index < l:
                curr_depth -= 1
                current_index += 1
            start_depth = min(start_depth, curr_depth)
            for i in range(current_index, r+1):
                curr_depth += 1
            
        else:
    
        
if __name__ == '__main__':
    print(main())
```