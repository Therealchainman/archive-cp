# Quora Programming Challenge 2022

# Practice 

## Problem: Digest Recommendation

### Solution: hash sets + intersection of hashsets

Create a dictionary for mapping story to the creator called
`story_creator` 

user and story are 1-indexed, there will be [1,n] stories
and [1,m] users.

p + q follows 
I imagine the follows as just being directed edges
p user->user edges
q user->story edges

Using sets for everything TLE, so I have to optimize for this platform by creating arrays since the data is so small. 
That way I'll have TC: (n^2m)