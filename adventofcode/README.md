


# Reading inputs in python

I like to use the following to read in input from an input.txt

```py
with open("inputs/input.txt", "r") as f
```
This reads it in as text format. 

If you are given an input that is a grid of integers such as 

```
111222
232322
234242
```

This is a clean way to create a 2d array of integers.
```py
data = []
lines = f.read().splitlines()
for line in lines:
    data.append([int(x) for x in line])
```

If you are given an input that is a grid of characters, and don't want to convert to integers

```py
data = f.read().splitlines()
```

If you are given a digits on each line separate by a comma and then you have two line breaks before
there is an input with a different structure.

```
1,3
2,4
3,5

fold along x=7
fold along y=8
```

Here I'm adding the data into a set data structure
```py
points, folds = f.read().split('\n\n') # splits the different input structures that are seprated by two line breaks
self.data = {tuple(map(int,points.split(','))) for points in points.split('\n')} # read in comma separated digits that are line breaked
self.folds = [(fold[11], int(fold[13])) for fold in folds.split('\n')]
```

A method to create pairs of characters by offseting the iterable.  
```
abcdef => (ab, bc, cd, de, ef)
```

```py
for k, v in zip(template, template[1:]):
    s = k + v
    freqs[s]+=1
```

Using regex to parse a more complex string input such as the following. 

It finds all the integers in the string, there are four.  

```
target area: x=20..30, y=-10..-5
```

```py
xmin, xmax, ymin, ymax = list(map(int, re.findall(r'[-\d]+', f.read())))
```

The best trick perhaps for complex inputs that have a pattern to them is to use this

For this example I had to create two patterns, because there were two possible variants, then all you have to do is pat.parse(line)

```py
from parse import compile
pat = compile("Valve {} has flow rate={:d}; tunnels lead to valves {}")
pat2 = compile("Valve {} has flow rate={:d}; tunnel leads to valve {}")

valve, flow, neighbors = pat.parse(line) if not isinstance(pat.parse(line), type(None)) else pat2.parse(line)
```

# Outputing 

This is a method to create an output when I'm given points that should be marked.  This will create a grid
and add the marks to the 2d array. and then convert it to a string for easier reading in terminal. 

```py
def displayData(self):
    self.maxX = 0
    self.maxY = 0
    for x, y in self.data:
        self.maxX = max(self.maxX, x)
        self.maxY = max(self.maxY, y)
    grid = [[' ' for x in range(self.maxX+1)] for y in range(self.maxY+1)]
    for x, y in self.data:
        grid[y][x] = '#'
    return "\n".join(["".join(row) for row in grid])
```

# Debugging tricks

## If output text too large for terminal, write it to a file. 

This is a method to write all print statements in the python script to a file before you close it. 
it sets the system stdout to be piped to this file.  
```py
import sys
sys.stdout = open('output.txt', 'w')
sys.stdout.close()
```

https://github.com/taylorott/Advent_of_Code/blob/main/src/Year_2022/Day22/Solution.py
My solution is able to programmatically determine how edges line up on the cube. The key intuition is to identify "inner corners" of the 2D pattern (for example where regions 1/3/4 meet, 3/4/5 meet, or 4/5/6 in the test case provided in the problem). These corners correspond to the starting points for how you would "zip up the cube".

Once these inner corners have been identified, you can travel along the perimeter of the 2D pattern in the two opposite directions (moving one unit-length line segment at a time). Each of these line-segment pair (one segment for the two directions we are traveling) will end up fusing when we fold the cube, so we can convert this into adjacency information for the corresponding grid-points in the 2D pattern.

The one thing to keep track of is that you need to know when to stop this zipping process. The termination criterion for a single zip is to see if, while traveling along the perimeter of the 2D pattern in opposite directions, you have to round two corners simultaneously (rounding a single corner corresponds to a single fold in the 2D pattern).
