

# Division 2

## Problem: Data Center

This metric is called the chebyshev distance, which can be converted into 
a manhattan distance by rotating the coordinates 45 degrees. 

And the problem can be reduced to minimize the manhattan distance

Can use a prefix for the sum of distance along each dimension for what is to the left and to the right?  

## Problem: Skyscraper

This problem can be solved with segment tree and lazy propagation for the range update

The segment tree has two queries, the first is to the find the left index, and that looks
for the last element that is greater than an amount.

The second query finds the right index and that looks for the first element that is greater than an amount.

This way we get the left and right skyscraper that the current skyscraper cannot see beyond. 

all functions are O(logN)
except the range update which is O(NlogN)

## Problem: Help Crawlers

Same as [New Flight Routes](<https://cses.fi/problemset/task/1685>)

Basically using indegrees and outdegrees to solve

## Problem: Treasure

Iterative DP works for this problem. 

## Problem: Broken Message



## Problem: Malicious Data



