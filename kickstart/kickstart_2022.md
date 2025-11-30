# Google Kickstart 2022

# Google Kickstart 2022 Round B

## Summary

Not too bad of a contest, I solved what I am capable of solving

## Infinity Area

### Solution 1: Simulation + Math

```py
from math import pi

def main():
    R, A, B = map(int,input().split())
    sum_area = 0
    while R > 0:
        # RIGHT CIRCLE
        sum_area += pi*R*R
        # LEFT CIRCLE
        R*=A
        sum_area += pi*R*R
        R//=B
    return sum_area


if __name__ == '__main__':
    T = int(input())
    for t in range(1,T+1):
        print(f"Case #{t}: {main()}")
```

## Palindromic Factors

### Solution 1: Efficient find factors + math

```py
from math import sqrt

def is_palindrome(num):
    return str(num) == str(num)[::-1]

def main():
    A = int(input())
    num_pal = 0
    seen = set()
    for i in range(1, int(sqrt(A))+1):
        if A%i==0:
            if i not in seen:
                if is_palindrome(i):
                    num_pal += 1
                seen.add(i)
            if A//i not in seen:
                if is_palindrome(A//i):
                    num_pal += 1
                seen.add(A//i)
    return num_pal


if __name__ == '__main__':
    T = int(input())
    for t in range(1,T+1):
        print(f"Case #{t}: {main()}")
```

## Unlock the Padlock

### Solution 1: BFS + memoization for best paths to unlock

```py
from collections import deque, defaultdict
from math import inf
def main():
    N, D = map(int,input().split())
    V = list(map(int,input().split()))
    compressed = [V[0]]
    best = inf
    for v in V[1:]:
        if compressed[-1] == v: continue
        compressed.append(v)
    queue = deque([(i, i, val, 0) for i, val in enumerate(compressed)])
    memo = defaultdict(lambda: inf)
    while queue:
        left, right, val, ops = queue.popleft()
        if right-left+1 == len(compressed) and val == 0:
            best = min(best, ops)
            continue
        if right-left+1 == len(compressed):
            queue.append((left, right, 0, min(ops+val,ops+D-val)))
            continue
        if left > 0 and right + 1 < len(compressed) and compressed[left-1] == compressed[right+1]:
            dist1 = (val - compressed[left-1])%D
            dist2 = (compressed[left-1] - val)%D
            cost1 = ops+dist1
            cost2 = ops+dist2
            if cost1 < memo[(left-1, right+1, compressed[left-1])]:
                memo[(left-1,right+1,compressed[left-1])] = cost1
                queue.append((left-1, right+1, compressed[left-1], cost1))
            if cost2 < memo[(left-1, right+1, compressed[left-1])]:
                memo[(left-1,right+1,compressed[left-1])] = cost2
                queue.append((left-1,right+1,compressed[left-1], cost2))
            continue
        if left > 0:
            dist1 = (val - compressed[left-1])%D
            dist2 = (compressed[left-1] - val)%D
            cost1 = ops+dist1
            cost2 = ops+dist2
            if cost1 < memo[(left-1,right, compressed[left-1])]:
                memo[(left-1,right,compressed[left-1])] = cost1
                queue.append((left-1, right, compressed[left-1], cost1))
            if cost2 < memo[(left-1,right, compressed[left-1])]:
                memo[(left-1,right,compressed[left-1])] = cost2
                queue.append((left-1, right, compressed[left-1], cost2))
        if right + 1 < len(compressed):
            dist1 = (val - compressed[right+1])%D
            dist2 = (compressed[right+1] - val)%D
            cost1, cost2 = ops+dist1, ops+dist2
            if cost1 < memo[(left, right+1,compressed[right+1])]:
                memo[(left,right+1,compressed[right+1])] = cost1
                queue.append((left, right+1,compressed[right+1], cost1))
            if cost2 < memo[(left, right+1,compressed[right+1])]:
                memo[(left,right+1,compressed[right+1])] = cost2
                queue.append((left, right+1,compressed[right+1], cost2))
    return best


if __name__ == '__main__':
    T = int(input())
    for t in range(1,T+1):
        print(f"Case #{t}: {main()}")
```

## Hamiltonian Tour

### Solution 1: 

```py

```

# Google Kickstart 2022 Round C

## Summary

## New Password

### Solution 1: string

```py
import string
SPECIAL_CHARACTERS = '#@*&'
string.digits
string.ascii_lowercase
def main():
    num_digits = num_lower = num_upper = num_special = 0
    N = int(input())
    S = input()
    for ch in S:
        num_digits += (ch in string.digits)
        num_lower += (ch in string.ascii_lowercase)
        num_upper += (ch in string.ascii_uppercase)
        num_special += (ch in SPECIAL_CHARACTERS)
    num_missing = 7 - N
    to_add = []
    if num_digits == 0:
        to_add.append('1')
        num_missing -= 1
    if num_lower == 0:
        to_add.append('a')
        num_missing -= 1
    if num_upper == 0:
        to_add.append('A')
        num_missing -= 1
    if num_special == 0:
        to_add.append('#')
        num_missing -= 1
    while num_missing > 0:
        num_missing -= 1
        to_add.append('1')
    return S + "".join(to_add)
    
if __name__ == '__main__':
    T = int(input())
    for t in range(1,T+1):
        print(f"Case #{t}: {main()}")
```

## Range Partition

### Solution 1: math + greedy

```py
import sys
sys.setrecursionlimit(1000000)
POS = "POSSIBLE"
IMP = "IMPOSSIBLE"
def main():
    N, X, Y = map(int,input().split())
    sum_N = N*(N+1)//2
    if sum_N%(X+Y)!=0: return IMP
    partition_sum = (sum_N//(X+Y))*X
    arr = []
    def partition(N, partition_sum):
        if N == 0 or partition_sum == 0: return
        if N > partition_sum:
            partition(N-1, partition_sum)
        else:
            arr.append(N)
            partition(N-1, partition_sum-N)
    partition(N,partition_sum)
    return POS + '\n' f'{len(arr)}' '\n' + ' '.join(map(str, arr))
            

if __name__ == '__main__':
    T = int(input())
    for t in range(1,T+1):
        print(f"Case #{t}: {main()}")
```

## Ants and Sticks

###

```py
from collections import namedtuple
from math import inf
def main():
    N, L = map(int,input().split())
    Ant = namedtuple('Ant', ['pos', 'id', 'dir'])
    ants_arr = []
    for i in range(1,N+1):
        loc, dir_ = map(int,input().split())
        ants_arr.append(Ant(loc, i, dir_ if dir_ == 1 else -1))
    ant_order = []
    left_edge, right_edge = -1, L
    ants_arr.sort(key=lambda ant: ant.pos)
    while len(ants_arr) > 1:
        event_time = inf
        for i, ant in enumerate(ants_arr):
            if i == 0 and ant.dir == -1: # leftmost ant traveling to left side
                event_time = min(event_time, ant.pos - left_edge)
            elif i == len(ants_arr) - 1 and ant.dir == 1:
                event_time = min(event_time, right_edge - ant.pos)
            if i > 0 and ants_arr[i-1].dir == 1 and ant.dir == -1: # left ant moving rightwards, right ant moving leftwards will have collision event
                event_time = min(event_time, (ant.pos - ants_arr[i-1].pos)/2)
        # move all ants to event_time
        for i, ant in enumerate(ants_arr):
            ants_arr[i] = ant._replace(pos=ant.pos+ant.dir*event_time)
        # find events that took place
        remaining_ants = []
        for i, ant in enumerate(ants_arr):
            if i == 0 and ant.pos == left_edge: 
                ant_order.append(ant.id)
                continue
            if i == len(ants_arr) - 1 and ant.pos == right_edge: 
                ant_order.append(ant.id)
                continue
            if remaining_ants and remaining_ants[-1].pos == ant.pos: 
                # left one was going right, right was going left
                remaining_ants[-1] = remaining_ants[-1]._replace(dir=-1)
                ant = ant._replace(dir=1)
            remaining_ants.append(ant)
        ants_arr = remaining_ants
    ant_order.append(ants_arr[-1].id)
    return ' '.join(map(str, ant_order))

if __name__ == '__main__':
    T = int(input())
    for t in range(1,T+1):
        print(f"Case #{t}: {main()}")

```

# Google Kickstart 2022 Round D

## Summary

##  P1: Image Labeler

### Solution 1:  greedy + math + sort

```py
def main():
    n, m = map(int,input().split())
    arr = sorted(list(map(float,input().split())))
    arr1 = arr[:n-m+1]
    arr2 = arr[n-m+1:]
    n1 = len(arr1)
    median = arr1[n1//2] if n1%2 else (arr1[n1//2]+arr1[n1//2-1])/2
    return sum(arr2) + median
    
    
    
if __name__ == '__main__':
    T = int(input())
    for t in range(1,T+1):
        print(f'Case #{t}: {main()}')
```

# Google Kickstart 2022 Round E

## Summary

##  P1: Color Game

### Solution 1:  greedy + math

```py
def main():
    n = int(input())
    score = 0
    loc = 0
    while loc < n:
        score += 1
        loc += 5
    return score

if __name__ == '__main__':
    T = int(input())
    for t in range(1,T+1):
        print(f'Case #{t}: {main()}')
```

## P2: Students and Mentors

### Solution 1:  binary search + hash table

```py
from collections import defaultdict
from bisect import bisect_right
def main():
    n = int(input())
    ratings = list(map(int,input().split()))
    index = defaultdict(list)
    for i, r in enumerate(ratings):
        index[r].append(i)
    result = [-1]*n
    ratings = sorted(list(set(ratings)))
    for r in ratings:
        i = bisect_right(ratings, 2*r) - 1
        if ratings[i] == r and len(index[r]) == 1:
            i -= 1
        if i < 0: continue
        for j in index[r]:
            result[j] = ratings[i]
    return ' '.join(map(str, result))

if __name__ == '__main__':
    T = int(input())
    for t in range(1,T+1):
        print(f'Case #{t}: {main()}')
```

## P3: Matching Palindrome

### Solution 1:  dynamic programming + manacher's algorithm + shortest prefix palindrome with the rest being suffix palindrome + P = QX and QXQ is palindrome too

```py
def main():
    delim_char, begin_char, end_char = '#', '$', '^'
    n = int(input())
    s = begin_char + input()
    arr = [s[i//2] if i%2 == 0 else delim_char for i in range(2*n+2)] + [end_char]
    d = [0]*(2*n+3)
    left = right = 1
    for i in range(1,2*n+2):
        d[i] = max(0, min(right-i, d[left + (right-i)]))
        while arr[i-d[i]] == arr[i+d[i]]:
            d[i] += 1
        if i+d[i] > right:
            left, right = i-d[i], i+d[i]
    for prefix_center in range(1,2*n):
        if d[prefix_center] < 2: continue
        prefix_left, prefix_right = prefix_center - d[prefix_center] + 1, prefix_center + d[prefix_center]- 1
        if prefix_left != 1: continue # required for it to be a prefix palindrome
        suffix_center = (2*n+1-prefix_right)//2+prefix_right
        suffix_radius = 2*n+1-suffix_center+1 # required radius to be a suffix palindrome
        if d[suffix_center] == suffix_radius:
            return ''.join(filter(lambda x: x!=delim_char, arr[prefix_left:prefix_right+1]))
    return s[1:]

    
if __name__ == '__main__':
    T = int(input())
    for t in range(1,T+1):
        print(f'Case #{t}: {main()}')
```

![image](images/matching_palindromes.png)

## Pizza Delivery

### Solution 1: 

```py

```

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

# Google Kickstart 2022 Round F

## 

### Solution 1: 

```py
def main():
    m, n, p = map(int, input().split())
    arr = [0]*n
    for i in range(1, m+1):
        walk = list(map(int, input().split()))
        if i == p:
            him = walk
        else:
            for j in range(n):
                arr[j] = max(arr[j], walk[j])
    return sum(max(0, x - y) for x, y in zip(arr, him))
    
if __name__ == '__main__':
    T = int(input())
    for t in range(1, T+1):
        print(f'Case #{t}: {main()}')
```

## 

### Solution 1: 

```py
from math import hypot
def main():
    Rs, Rh = map(int, input().split())
    N = int(input())
    disks = []
    squared_dist = lambda x, y: x*x + y*y
    intersects = lambda d: d <= (Rh+Rs)*(Rh+Rs)
    for _ in range(N):
        x, y = map(int, input().split())
        disks.append((x, y, 0))
    M = int(input())
    for _ in range(M):
        x, y = map(int, input().split())
        disks.append((x, y, 1))
    x, y = 2, 3
    disks = list(filter(lambda x: intersects(squared_dist(x[0], x[1])), disks))
    disks.sort(key = lambda x: squared_dist(x[0], x[1]))
    red = yellow = 0
    prev_team = -1
    for _, _, team in disks:
        if prev_team != -1 and prev_team != team: break
        red += (team==0)
        yellow += team
        prev_team = team
    return f'{red} {yellow}'
    
if __name__ == '__main__':
    T = int(input())
    for t in range(1, T+1):
        print(f'Case #{t}: {main()}')
```

## 

### Solution 1: 

```py
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    total = 0
    for i in range(n):
        prefix = 0
        for j in range(i, n):
            prefix += arr[j]
            if prefix < 0: break
            total += prefix
    return total
    
if __name__ == '__main__':
    T = int(input())
    for t in range(1, T+1):
        print(f'Case #{t}: {main()}')
```

## 

### Solution 1: 

```py

```

# Google Kickstart 2022 Round H

## Running in Circles

### Solution 1:  modular arithmetic + traveling along a circle + use remaining distance + what was previous direction when touch start

```py
def main():
    L, N = map(int, input().split())
    laps = pos = 0
    start = None
    for _ in range(N):
        dist, dir = input().split()
        dist = int(dist)
        remainingLaps = pos if dir == 'A' else (L - pos)%L
        sign = 1 if dir == 'C' else -1
        pos = (pos + sign*dist) % L
        if dist >= remainingLaps:
            currentLaps = 1 if remainingLaps > 0 and start == dir else 0
            dist -= remainingLaps
            currentLaps += dist // L
            laps += currentLaps
            start = dir
    return laps
if __name__ == '__main__':
    T = int(input())
    for t in range(1, T+1):
        print(f'Case #{t}: {main()}')
```

## Magical Well Of Lilies

### Solution 1: 

```py
from math import *
def main():
    L = int(input())
    res = L
    for i in range(3, L+1):
        cand = 2*(L-i)//i + (L-i)%i + 4 + i
        res = min(res, cand)
    return res
if __name__ == '__main__':
    T = int(input())
    for t in range(1, T+1):
        print(f'Case #{t}: {main()}')
```

## Electricity

### Solution 1:  two dfs + dfs to compute the size of decreasing segments in subtree of current node + dfs to compute the size of a larger parent and it's size of decreasing segments

```py
from sys import *
setrecursionlimit(int(1e6))
def main():
    N = int(input())
    arr = list(map(int, input().split()))
    adj_list = [[] for _ in range(N)]
    for _ in range(N-1):
        u, v = map(int, input().split())
        adj_list[u-1].append(v-1)
        adj_list[v-1].append(u-1)
    size = [1]*N
    def smaller(node, parent):
        for child in adj_list[node]:
            if child == parent: continue
            child_small_segment_size = smaller(child, node)
            if arr[child] < arr[node]:
                size[node] += child_small_segment_size
        return size[node]
    smaller(0, -1)
    def larger(node, parent):
        for child in adj_list[node]:
            if child == parent: continue
            if arr[child] > arr[node]:
                size[child] += size[node]
            larger(child, node)
    larger(0, -1)
    return max(size)
if __name__ == '__main__':
    T = int(input())
    for t in range(1, T+1):
        print(f'Case #{t}: {main()}')
```

## Level Design

### Solution 1:  union find + graph + connected components + knapsack + iterative dp + O(n^2) time

TLEs on test case 2

```py
from math import *
class UnionFind:
    def __init__(self):
        self.size = dict()
        self.parent = dict()
    
    def find(self,i: int) -> int:
        if i not in self.parent:
            self.size[i] = 1
            self.parent[i] = i
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
    
    @property
    def root_count(self):
        return sum(node == self.find(node) for node in self.parent)

    def __repr__(self) -> str:
        return f'parents: {[(i, parent) for i, parent in enumerate(self.parent)]}, sizes: {self.size}'
        
def main():
    N = int(input())
    arr = list(map(int, input().split()))
    dsu = UnionFind()
    for i, num in enumerate(arr, start = 1):
        dsu.union(i, num)
    cycleSizes = []
    for i in range(1, N+1):
        # i is a representative (root) node for a connected component
        if i == dsu.find(i):
            cycleSizes.append(dsu.size[i])
    dp = [inf]*(N+1)
    dp[0] = 0
    for size in cycleSizes:
        for i in range(N-size, -1, -1):
            dp[i+size] = min(dp[i+size], dp[i]+1)
        for i in range(1, size):
            dp[i] = min(dp[i], 1)
        dp[size] = 0
    return ' '.join(map(str, dp[1:]))
    

if __name__ == '__main__':
    T = int(input())
    for t in range(1, T+1):
        print(f'Case #{t}: {main()}')
```

```py

import math
from collections import deque
class UnionFind:
    def __init__(self):
        self.size = dict()
        self.parent = dict()
    
    def find(self,i: int) -> int:
        if i not in self.parent:
            self.size[i] = 1
            self.parent[i] = i
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
    
    @property
    def root_count(self):
        return sum(node == self.find(node) for node in self.parent)

    def __repr__(self) -> str:
        return f'parents: {[(i, parent) for i, parent in enumerate(self.parent)]}, sizes: {self.size}'
        
def main():
    N = int(input())
    arr = list(map(int, input().split()))
    dsu = UnionFind()
    for i, num in enumerate(arr, start = 1):
        dsu.union(i, num)
    cycleSizes = [0]*(N+1)
    for i in range(1, N+1):
        # i is a representative (root) node for a connected component
        if i == dsu.find(i):
            cycleSizes[dsu.size[i]] += 1
    # bounded knapsack problem
    dp = [math.inf]*(N+1)
    dp[0] = 0
    for cycle_len in range(1, N + 1):
        cnt = cycleSizes[cycle_len]
        if cnt == 0: continue
        # simulates adding to existing solutions
        # this will be ran approximatley sqrt(N) times
        # sliding window for each gap
        for i in range(N, N - cycle_len, -1):
            min_window = deque()
            for right in range(i, -1, -cycle_len):
                left = right - cnt*cycle_len
                if min_window and min_window[0][1] >= right:
                    min_window.popleft()
                while min_window and dp[left] + cnt <= min_window[-1][0] + (right - min_window[-1][1])//cycle_len:
                    min_window.pop()
                min_window.append((dp[left], left))
                dp[right] = min(dp[right], min_window[0][0] + (right - min_window[0][1])//cycle_len)
    # simulates breaking, can always perform minimum swaps and then you can always break to get something smaller so that requires 1 extra move
    min_swaps = math.inf
    for i in reversed(range(1, N+1)):
        dp[i] = min(dp[i], min_swaps+1) # +1 for breaking
        min_swaps = min(min_swaps, dp[i])
    return ' '.join(map(str, [x -1 for x in dp[1:]]))
    

if __name__ == '__main__':
    T = int(input())
    for t in range(1, T+1):
        print(f'Case #{t}: {main()}')
```