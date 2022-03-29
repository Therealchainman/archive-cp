# DMOPC '21 March Contest

## Summary

I was only able to solve problem 1 today.  I struggled on the rest.  I am going to upsolve at least problems 2-4


## Chika Grids

This problem was just greedy algorithm, easy start. 

Just had to keep the prefix max and previous columns max values.

```py
def main():
    N, M = map(int,input().split())
    grid = [list(map(int,input().split())) for _ in range(N)]
    pcol = [0]*M
    for r in range(N):
        prow = 0
        for c in range(M):
            if grid[r][c]==0:
                grid[r][c] = max(prow,pcol[c])+1
            elif grid[r][c] <= prow or grid[r][c] <= pcol[c]:
                return -1
            prow = pcol[c] = grid[r][c]
    return "\n".join([" ".join(map(str,row)) for row in grid])

if __name__ == '__main__':
    print(main())
```

## Knitting Scarves

This one may not be that difficult, I was thinking I may be able to use a hash table, or I tried coming at it from backwards.
But these don't lead to anything,  It is just going to be an inefficient algorithm, if I just write something that
iterates through and moves all the integers to the new location. This would be O(NQ), which is entirely slow. 

This is just brute force, I just couldn't come up with a good prefix table to help.  

When you move a range of integers, you have to update their locations right?  that seems time intensive. 
The only thing I know about is that there is something that could be lazily evaluated.  But yeah 
I don't think that exists.  

```py

```

## Whack-an-Ishigami

Directed graph problem, that can have cycles, I tried some topological ordering ideas, but that didn't lead 
anywhere.  I was not going to be able to solve this.  

If you consider a simpler case where it is a tree, that helps a little maybe because I can just 

```py

```

## Rocket Race

Interesting one, I believe you can solve it without segment tree, but it didn't work my greedy method.  
After that I don't have any more good ideas for now.  

```py

```

