# Google Kickstart 2022 Round F

## Summary

## Sort the Fabrics

### Solution 1:  custom sorts + zip + dataclass

```py
from dataclasses import make_dataclass
def main():
    n = int(input())
    Fabric = make_dataclass('Fabric', [('color', str), ('durability', int), ('unique', int)])
    fabrics = []
    for _ in range(n):
        c, d, u = input().split()
        fabrics.append(Fabric(c,int(d),int(u)))
    ada_sort = sorted(fabrics, key=lambda fabric: (fabric.color, fabric.unique))
    charles_sort = sorted(fabrics, key=lambda fabric: (fabric.durability, fabric.unique))
    return sum(x==y for x,y in zip(ada_sort, charles_sort))
    
if __name__ == '__main__':
    T = int(input())
    for t in range(1,T+1):
        print(f'Case #{t}: {main()}')
```

## Water Container System

### Solution 1:  bfs + undirected graph + counter

```py
from collections import Counter, deque
def main():
    n, q = map(int,input().split())
    adjList = [[] for _ in range(n+1)]
    for _ in range(n-1):
        u, v = map(int,input().split())
        adjList[u].append(v)
        adjList[v].append(u)
    levels = Counter() # count of water containers at each level
    queue = deque([(1,None)])
    lv = 0
    while queue:
        size = len(queue)
        for _ in range(size):
            node, parent_node = queue.popleft()
            levels[lv] += 1
            for nei_node in adjList[node]:
                if nei_node == parent_node: continue
                queue.append((nei_node, node))
        lv += 1
    for _ in range(q):
        input()
    lv = 0
    containers_filled = 0
    while q > 0:
        q -= levels[lv]
        if q >= 0:
            containers_filled += levels[lv]
            lv += 1
    return containers_filled
if __name__ == '__main__':
    T = int(input())
    for t in range(1,T+1):
        print(f'Case #{t}: {main()}')
```

## Story of Seasons

### Solution 1:  max heap + sort + iterate backwards + add seeds to max heap or active pool of seeds that can mature in time at this current day + always plant the seeds with maximum value first

```py
from heapq import *
        
def main():
    D, N, X = map(int, input().split())
    seeds = [None]*N
    for i in range(N):
        Q, L, V = map(int, input().split())
        seeds[i] = (L, Q, V)
    seeds.sort()
    max_heap = [] # (value, seed index)
    cur_day = res = seed_index = remaining_harvest_cur_day = 0 # counting backwards from day D
    while cur_day < D:
        if not max_heap and seed_index == N: # planted all and harvested all seeds
            break
        if seed_index < N and seeds[seed_index][0] == cur_day: # push some more seeds into the max heap
            maturation_time, quantity, value = seeds[seed_index]
            heappush(max_heap, (-value, -quantity))
            seed_index += 1
        elif not max_heap: # if no seeds to plant, move to the next day when you can plant a seed that you will be able to harvest on time
            maturation_time, quantity, value = seeds[seed_index]
            heappush(max_heap, (-value, -quantity))
            seed_index += 1
            remaining_harvest_cur_day = 0
            cur_day = maturation_time
        elif remaining_harvest_cur_day == 0: # harvest the seeds with the full day available
            value, quantity = map(abs, heappop(max_heap))
            max_days = min(D, seeds[seed_index][0]) if seed_index < N else D # it has at least maximum value up to max_days
            days_to_harvest = max_days - cur_day
            harvest = min(X * days_to_harvest, quantity)
            res += harvest * value
            num_days_to_harvest = harvest//X
            cur_day += num_days_to_harvest
            remaining_harvest_cur_day = 0
            if harvest % X != 0:
                remaining_harvest_cur_day = X - harvest % X
            if quantity - harvest > 0: # if there are still seeds left, push them back into the max heap
                heappush(max_heap, (-value, -(quantity - harvest)))
        else: # remaining_harvest_cur_day > 0
            value, quantity = map(abs, heappop(max_heap))
            harvest = min(remaining_harvest_cur_day, quantity)
            res += harvest * value
            remaining_harvest_cur_day -= harvest
            if quantity - harvest > 0: # if there are still seeds left, push them back into the max heap
                heappush(max_heap, (-value, -(quantity - harvest)))
            if remaining_harvest_cur_day == 0:
                cur_day += 1
    return res

if __name__ == '__main__':
    T = int(input())
    for t in range(1, T+1):
        print(f'Case #{t}: {main()}')
```

## Scheduling a Meeting

### Solution 1:  sliding window + prefix count + storing variables to allow O(1) time update for each window so can solve in O(D + N + Mlog(M)) time

```py
import math

def main():
    N, K, X, D = map(int, input().split())
    M = int(input())
    meeting_starts, meeting_ends = [None] * M, [None] * M
    for i in range(M):
        P, L, R = map(int, input().split())
        meeting_starts[i] = (L, P)
        meeting_ends[i] = (R, P)
    meeting_starts.sort()
    meeting_ends.sort()
    tech_lead_meeting_counts = [0] * (N+1) # The count of meetings for each tech lead in the window
    meeting_counts = [0] * (M + 1) # The count of tech leads with this many meetings in the window
    meeting_counts[0] = N # Initially, all tech leads have 0 meetings
    start = end = cur_count = 0
    prefix_count_meetings = N
    window_meeting_count = 0 # The number of meetings in the window for K tech leads
    res = math.inf
    # sliding window of size X
    for right in range(D + 1):
        # PROCESS MEETINGS THAT END FOR CURRENT WINDOW
        left = right - X
        while left >= 0 and end < M and meeting_ends[end][0] == left:
            tech_lead = meeting_ends[end][1]
            prev_count = tech_lead_meeting_counts[tech_lead]
            meeting_counts[prev_count] -= 1
            tech_lead_meeting_counts[tech_lead] -= 1
            meeting_counts[prev_count - 1] += 1
            if prev_count <= cur_count:
                window_meeting_count -= 1
            if prev_count == cur_count and prefix_count_meetings - meeting_counts[cur_count] == K:
                cur_count -= 1
                prefix_count_meetings = K
            elif prev_count == cur_count + 1:
                prefix_count_meetings += 1
            end += 1
        # PERFORM ACTION TO UPDATE RESULTS
        if left >= 0:
            res = min(res, window_meeting_count)
        # PROCESS NEW MEETINGS THAT START FOR NEXT WINDOW
        while start < M and meeting_starts[start][0] == right:
            tech_lead = meeting_starts[start][1]
            prev_count = tech_lead_meeting_counts[tech_lead]
            meeting_counts[prev_count] -= 1
            tech_lead_meeting_counts[tech_lead] += 1
            meeting_counts[prev_count + 1] += 1
            if prev_count < cur_count:
                window_meeting_count += 1
            elif prev_count == cur_count: # if this tech lead was at the border of the window
                prefix_count_meetings -= 1 # The number of tech leads with this many meetings in the window
            if prefix_count_meetings < K: # the prefix has become too small so that means this tech lead was pivotal, so you need to increment window meeting to include this meeting this person added
                window_meeting_count += 1
                prefix_count_meetings += meeting_counts[prev_count + 1] # add the number of tech leads with this many meetings in the window
                cur_count += 1 # increment the current count
            start += 1
    return res

if __name__ == '__main__':
    T = int(input())
    for t in range(1, T+1):
        print(f'Case #{t}: {main()}')
```

### Solution 2: prefix sums + tree searching

```py

```