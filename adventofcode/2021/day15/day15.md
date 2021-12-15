# Notes from today

# Part 1

Dijkstra algorithm with a visited set, to get the cheapest path through the grid from top left to bottom right.

Using a minheap datastructure

```py
import heapq
with open("inputs/input.txt", "r") as f:
    data = []
    lines = f.read().splitlines()
    for line in lines:
        data.append([int(x) for x in line])
    heap = []
    R, C = len(data), len(data[0])
    heapq.heappush(heap,(0,0,0))
    vis = set()
    vis.add((0,0))
    while heap:
        cost, r, c = heapq.heappop(heap)
        if r==R-1 and c==C-1:
            print(cost)
            break
        for dr in range(-1,2):
            for dc in range(-1,2):
                if abs(dr+dc)==1:
                    nr, nc = r+dr, c+dc
                    if 0<=nr<R and 0<=nc<C and (nr,nc) not in vis:
                        vis.add((nr,nc))
                        heapq.heappush(heap,(cost+data[nr][nc],nr,nc))
```

# Part 2

Really the only caveat is that I need to add to the original grid and still return the min cost, so still dijkstra but with 
modification


```py
import heapq
with open("inputs/input.txt", "r") as f:
    data = []
    lines = f.read().splitlines()
    for line in lines:
        data.append([int(x) for x in line])
    heap = []
    R, C = len(data), len(data[0])
    for k in range(1,5):
        for j in range(C):
            for i in range((k-1)*R,k*R):
                if i+R == len(data):
                    data.append([])
                data[i+R].append(data[i][j]+1 if data[i][j]<9 else 1)
    R = len(data)
    for k in range(1,5):
        for i in range(R):
            for j in range((k-1)*C, k*C):
                data[i].append(data[i][j]+1 if data[i][j]<9 else 1)
    C = len(data[0])
    heapq.heappush(heap,(0,0,0))
    vis = set()
    vis.add((0,0))
    while heap:
        cost, r, c = heapq.heappop(heap)
        if r==R-1 and c==C-1:
            print(cost)
            break
        for dr in range(-1,2):
            for dc in range(-1,2):
                if abs(dr+dc)==1:
                    nr, nc = r+dr, c+dc
                    if 0<=nr<R and 0<=nc<C and (nr,nc) not in vis:
                        vis.add((nr,nc))
                        heapq.heappush(heap,(cost+data[nr][nc],nr,nc))
```

Improvement on this would be to do online generation of the map fromt he single grid we start with.  Then I just need to generate the 
values from the original with a formula and that would make it more efficient. 


Let's do the math

We have a grid with R rows and C cols

I begin by taking values from within, but eventually
nr >= R suppose, in which case nr/R = 1
So to get it's value I need to take nr - 1*R from the original grid, which takes me to row=0, then I need to add to the value data[nr][nc] + 1
but if data[nr][nc]+1>9 then I need to actually add data[nr][nc]+1-9.
Sure and the same for nc>=C

Consider the next case:  nr>=R and nc>=C 

You can get it's value will be if x=nr/R and y = nc/C
then we have data[nr][nc] + x + y

Solves it in 1.2 seconds on my hardware.  

```py
import heapq
with open("inputs/input.txt", "r") as f:
    data = []
    lines = f.read().splitlines()
    for line in lines:
        data.append([int(x) for x in line])
    heap = []
    R, C = len(data), len(data[0])
    heapq.heappush(heap,(0,0,0))
    vis = set()
    vis.add((0,0))
    while heap:
        cost, r, c = heapq.heappop(heap)
        if r==5*R-1 and c==5*C-1:
            print(cost)
            break
        for dr in range(-1,2):
            for dc in range(-1,2):
                if abs(dr+dc)==1:
                    nr, nc = r+dr, c+dc
                    if 0<=nr<5*R and 0<=nc<5*C and (nr,nc) not in vis:
                        x, y = nr//R, nc//C
                        nval = data[nr-x*R][nc-y*C] + x + y
                        nval = nval if nval<=9 else nval-9
                        vis.add((nr,nc))
                        heapq.heappush(heap,(cost+nval,nr,nc))
```