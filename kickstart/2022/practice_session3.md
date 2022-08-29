# Google Practice with Kick Start Session #3

## Record Breaker

### Solution 1: prefix max + loop

```py
def main():
    n = int(input())
    days = list(map(int,input().split()))
    prefixMax = 0
    num_records = 0
    for i in range(n):
        if (i==0 or days[i] > prefixMax) and (i==n-1 or days[i]>days[i+1]):
            num_records += 1
        prefixMax = max(prefixMax, days[i])
    return num_records

if __name__ == '__main__':
    T = int(input())
    for t in range(1,T+1):
        print(f'Case #{t}: {main()}')
```

## Wiggle Walk

### Solution 1: build possible moves based on position in 2d space + merge intervals in 2d space

```py
from collections import defaultdict, namedtuple
def main():
    N, R, C, sr, sc = map(int,input().split())
    instructions = input()
    Delta = namedtuple('Delta', ['dr','dc'])
    visited = defaultdict(lambda: Delta(0,0))
    visited[(sr,sc)] = Delta(1,1)
    Movements = namedtuple('Movements', ['left', 'right', 'up', 'down'])
    cur_moves = Movements(1,1,1,1)
    dirs = {'E': (0,1), 'W': (0,-1), 'S': (1, 0), 'N': (-1, 0)}
    for instruction in instructions:
        dr, dc = dirs[instruction]
        if dr == 1:
            sr += cur_moves.down
        if dr == -1:
            sr -= cur_moves.up
        if dc == 1:
            sc += cur_moves.right
        if dc == -1:
            sc -= cur_moves.left
        # print('dr', dr,'dc',dc)
        # print('sr', sr, 'sc', sc)
        above_len, below_len = visited[(sr-1,sc)].dr, visited[(sr+1,sc)].dr
        left_len, right_len = visited[(sr,sc-1)].dc, visited[(sr,sc+1)].dc
        cur_moves = Movements(left_len+1, right_len+1, above_len+1, below_len+1)
        # print(cur_moves)
        row_len = above_len + below_len + 1
        col_len = left_len + right_len + 1
        # print('row_len', row_len, 'col_len', col_len)
        visited[(sr+below_len, sc)] = visited[(sr+below_len, sc)]._replace(dr=row_len)
        visited[(sr-above_len, sc)] = visited[(sr-above_len, sc)]._replace(dr=row_len)
        visited[(sr,sc+right_len)] = visited[(sr,sc+right_len)]._replace(dc=col_len)
        visited[(sr,sc-left_len)] = visited[(sr,sc-left_len)]._replace(dc=col_len)
        # print(visited)
    return f'{sr} {sc}'

if __name__ == '__main__':
    T = int(input())
    for t in range(1,T+1):
        print(f'Case #{t}: {main()}')
```

```py
from collections import defaultdict, namedtuple
from dataclasses import make_dataclass, field
def main():
    N, R, C, sr, sc = map(int,input().split())
    instructions = input()
    Delta = make_dataclass('Delta', [('dr', int, field(default=0)), ('dc', int, field(default=0))])
    visited = defaultdict(Delta)
    visited[(sr,sc)] = Delta(1,1)
    Movements = namedtuple('Movements', ['left', 'right', 'up', 'down'])
    cur_moves = Movements(1,1,1,1)
    dirs = {'E': (0,1), 'W': (0,-1), 'S': (1, 0), 'N': (-1, 0)}
    for instruction in instructions:
        dr, dc = dirs[instruction]
        if dr == 1:
            sr += cur_moves.down
        if dr == -1:
            sr -= cur_moves.up
        if dc == 1:
            sc += cur_moves.right
        if dc == -1:
            sc -= cur_moves.left
        # print('dr', dr,'dc',dc)
        # print('sr', sr, 'sc', sc)
        above_len, below_len = visited[(sr-1,sc)].dr, visited[(sr+1,sc)].dr
        left_len, right_len = visited[(sr,sc-1)].dc, visited[(sr,sc+1)].dc
        cur_moves = Movements(left_len+1, right_len+1, above_len+1, below_len+1)
        # print(cur_moves)
        row_len = above_len + below_len + 1
        col_len = left_len + right_len + 1
        # print('row_len', row_len, 'col_len', col_len)
        visited[(sr+below_len, sc)].dr = row_len
        visited[(sr-above_len, sc)].dr = row_len
        visited[(sr,sc+right_len)].dc = col_len
        visited[(sr,sc-left_len)].dc = col_len
        # print(visited)
    return f'{sr} {sc}'

if __name__ == '__main__':
    T = int(input())
    for t in range(1,T+1):
        print(f'Case #{t}: {main()}')
```

## GBus Count

### Solution 1:  update interval and track active_busses accross interval

```py
from collections import defaultdict
def main():
    n = int(input())
    routes = list(map(int,input().split()))
    p = int(input())
    upper_bound = 0
    queries = defaultdict(list)
    for i in range(p):
        c = int(input())
        upper_bound = max(upper_bound, c)
        queries[c].append(i)
    cities = [0]*(upper_bound+1)
    answer = [0]*p
    for i in range(0,len(routes),2):
        left, right = routes[i], routes[i+1]
        if left <= upper_bound:
            cities[left] += 1
        if right < upper_bound:
            cities[right+1] -= 1
    active_buses = 0
    for i in range(upper_bound+1):
        active_buses += cities[i]
        for index in queries[i]:
            answer[index] = active_buses
    return ' '.join(map(str, answer))
    
if __name__ == '__main__':
    T = int(input())
    for t in range(1,T+1):
        print(f'Case #{t}: {main()}')
        if t<T:
            input()
```

## Sherlock and Watson Gym Secrets

### Solution 1:

```py

```