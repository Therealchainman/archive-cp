
# Part 1

We are given 100 digits in 100 lines
To parse the data I convert from the file to a string with f.read(), then I use split to 
create a list that uses whitespace delimiter by default so it will use the line break. 
And then I can iterate through these strings to create and array of integers.

I used a sum function the value + 1 for the lowest points by checking that all the locations around it are larger if
it is within the grid. 

```py
data = []
lines = f.read().split()
for line in lines:
    data.append([int(x) for x in line])
R, C = len(data), len(data[0])
sumRisk = sum(data[i][j] + 1 for i in range(R) for j in range(C) if all(data[i][j] < data[i+dr][j+dc] for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)) if 0 <= i+dr < R and 0 <= j+dc < C))
print(sumRisk)
```

# Part 2

This part is trickier, but it turns out you can use the fact that 9s are never going to be part of a basin as stipulated in 
puzzle statement. 

Let's look at a particularly interesting edge case

999
979
289

This will return two basins [2,2] of size 2, but that can't be right, cause both basins have the 8, but is that possible.
It says in the statement a basins is all locations that eventually flow downward to a single low point.  But 8 is flowing
down to two low points. And it states all location will be a part of exactly one basin.  So how can 8 be a part of 2 basins.  
So we can't just do a simple bfs that moves outward to neighbors that are larger. 

This following dfs solution use memoization to avoid it computing 8 as being part of two basins. 
So basically it is dfs so it goes towards a path that leads to a low point.  
Then it will return that low point.  It say I hit a point that I already know the low point is, it will just return
that via the memoization.  It is most efficient to save values from previous.  


```py
data = []
lines = f.read().split()
for line in lines:
    data.append([int(x) for x in line])
R, C = len(data), len(data[0])
@lru_cache(maxsize=None)
def dfs(x, y):
    for i, j in ((x, y-1), (x, y+1), (x-1, y), (x+1, y)):
        if 0 <= i < R and 0 <= j < C and data[i][j] < data[x][y]:
            return dfs(i,j)
    return (x,y)
basins = Counter(dfs(i,j) for i in range(R) for j in range(C) if data[i][j] != 9)
heights = sorted(list(basins.values()))
print(heights[-1]*heights[-2]*heights[-3])
```