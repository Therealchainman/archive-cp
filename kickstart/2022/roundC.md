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

##

###
