# Leetcode Weekly Contest 282

## 2185. Counting Words With a Given Prefix

### Solution: loop

```py
class Solution:
    def prefixCount(self, words: List[str], pref: str) -> int:
        return sum(1 for word in words if word[:len(pref)] == pref)
```

## 2186. Minimum Number of Steps to Make Two Strings Anagram II

### Solution: hashmap + loop

```py
class Solution:
    def minSteps(self, s: str, t: str) -> int:
        f1, f2 = [0]*26, [0]*26
        for c in s:
            f1[ord(c)-ord('a')] += 1
        for c in t: 
            f2[ord(c)-ord('a')] += 1
        return sum(abs(c1-c2) for c1, c2 in zip(f1, f2))
```

## 2187. Minimum Time to Complete Trips

### Solution 1: binary search 

```py
class Solution:
    def minimumTime(self, time: List[int], totalTrips: int) -> int:
        lo, hi = 1, min(time)*totalTrips
        def possible(est):
            return sum(est//t for t in time) >= totalTrips
            
        while lo < hi:
            mid = (lo+hi) >> 1
            if possible(mid):
                hi = mid
            else:
                lo = mid + 1
            
        return lo
```

### Solution 2: binary search + bisect + greedy

```py
class Solution:
    def minimumTime(self, time: List[int], totalTrips: int) -> int:
        return bisect.bisect_left(range(min(time)*totalTrips), totalTrips, key = lambda ctime: sum((ctime//t for t in time)))
```

## 2188. Minimum Time to Finish the Race

I should graph what is happening with a tire over time do a little bit of data analysis.

But first let's see if I can make some headway with this knowledge

So I know the time for remaining on a tire grows at an exponential rate

I can have 100,000 tires potentially to choose from 
I can perform a changeTime at any point, but we don't want to do that, it will be too 
complex of problem

One solution is to pack all the possiblities into a min_heap

But let's consider this as a graph

what if we have all our initial nodes for lap 1, which could be any of the 100,000 tires
Then we have to traverse the node to the next lap, now we either traverse with same tire and 
have an edge weight that is the cost of lap 2, or we change tires and have that cost. 

Anyway all of my assumptions were wrong, I should have realized that just map
the best price to continue to go straight for sometime, and realize the 
worst case is we'd want to go straight for 18 times,  via an analysis of upper bounds
if r=2, the cheapest, at the point of 2^18 > max(changeTime) + max(f) = 2e5
Even if it were the most expensive change time and slowest fresh tires, at some point, 
with the smallest r, it will no longer make since after 18 laps on any tire to continue. 
So we can assume we only go straight upto 18 times.  

### Solution: Brute force for going straight + dynamic programming for computing minimum cost for each lap

TC: O(N^2), where N = numLaps

```py
class Solution:
    def minimumFinishTime(self, tires: List[List[int]], changeTime: int, numLaps: int) -> int:
        nochange = [math.inf]*19
        LIMIT = int(2e5)
        for f, r in tires:
            cur_time = f
            total_time = cur_time
            nochange[1] = min(nochange[1], total_time)
            for i in range(2,19):
                cur_time *= r
                total_time += cur_time
                if total_time > LIMIT: break
                nochange[i] = min(nochange[i],total_time)
        
        dp = [math.inf]*(numLaps+1)
        for i in range(1,numLaps+1):
            if i<19:
                dp[i] = min(dp[i], nochange[i]) 
            for j in range(1,i//2+1):
                dp[i] = min(dp[i], dp[j] + changeTime + dp[i-j])
                
        return dp[-1]
```

