# Yet Another Contest 1

# P1: Mixed Doubles

## Solution: Greedy + Math

The problem with my initial solution was that I had a dumb bug.  The first bug I had I am not certain. 
But writing the solution in the manner below, I had a bug where I want a to be smaller. But I was actually
making a the larger element, had my inequality flipped for b < a.  I need to use better variable names next time.

```py
MOD = int(1e9)+7
N, K = map(int,input().split())
data = [sorted(list(map(lambda x: int(x), input().split()))), sorted(list(map(lambda x: int(x), input().split())))]
a, b = data[0][-1], data[1][-1]
res = 0
for man, woman in zip(data[0][:-1], data[1][:-1]):
    res = (res+(man*woman)%MOD)%MOD
if b < a:
    a, b = b, a

if a + K <= b:
    a += K
else:
    K -= (b-a)
    a += (b-a)
    a += K//2
    K -= K//2
    b += K
res = (res+(a*b)%MOD)%MOD
print(res)
```

# P2: A Boring Problem

## Solution: Graph + Dynamic Programming

This one is tricky as well.  I want to figure out the linear solution.  

I was thinking I could use some sliding window algorithm.  But I can't determine the strategy


```py
N = int(input())
colors = input()
graph = [[] for _ in range(N+1)]
for _ in range(N-1):
    u, v = map(int,input().split())
    graph[u].append(v)
    graph[v].append(u)
N-=2
print(sum(i for i in range(1,N+1)))
```

# P3: Fluke

## Solution: Strategy Stealing 

This seems very straight forward, but I realized, how do I mark the grid as I go through the recursion. 
The only way I can think is if I use dfs+backtracking.  But that would make the time complexity
exponential and umm for N^2 possible moves that would be 2^(N^2), or 2^100,000.  I definitely can't brute
force that.  So I was kind of stuck at the part of figuring out how to keep and update of the grid as I 
iterate through the possible states.  Cause at each recursive call the possible moves depend on the 
current grid.  

I was completely wrong about my approach.  It turns out this is a problem called strategy stealing.  I've never 
heard of that before though.  It seems that as long as N is even, if you allow player 2 to go first.
You will always have a guaranteed place to go if you make a move by rotating the grid by 180 degrees
If it is odd, you need to go first to take the center spot. 

Stratey stealing is from cobinatorial game theory, and it involves when the second player cannot have 
a guaranteed winning strategy.  This argument applies to any symmetric game.  

What is a symmetric game? It is a game where player 1 can steal player 2 strategy.  

```py
import sys

def solve():
  N, T = map(int,input().split())
  sys.stdout.flush()
  first_player = 2 if N%2==0 else 1 
  for _ in range(T):
    print(first_player,flush=True)
    if first_player == 1:
      print(f'{N//2+1} {N//2+1}',flush=True)
    while True:
      r, c = map(int, input().split())
      sys.stdout.flush()
      if r==0 and c==0: break
      if r==-1 and c==-1: return
      print(f'{N-r+1} {N-c+1}',flush=True)
      
solve()
```