# Leetcode Weekly Contest 290

## Summary

## 2248. Intersection of Multiple Arrays

### Solution 1:  set intersection

```py
class Solution:
    def intersection(self, nums: List[List[int]]) -> List[int]:
        intersect_set = set(nums[0])
        for i in range(1,len(nums)):
            intersect_set &= set(nums[i])
        return sorted(intersect_set)
```

```py
class Solution:
    def intersection(self, nums: List[List[int]]) -> List[int]:
        return sorted(set.intersection(*[set(nums[i]) for i in range(len(nums))]))
```

## 2249. Count Lattice Points Inside a Circle

### Solution 1: brute force with reduced search space

```py

class Solution:
    def countLatticePoints(self, circles: List[List[int]]) -> int:
        cnt = 0
        maxx = max(x+r for x, _, r in circles)
        maxy = max(y+r for _, y, r in circles)
        minx = min(x-r for x, _, r in circles)
        miny = min(y-r for _,y,r in circles)
        for x, y in product(range(minx, maxx+1), range(miny,maxy+1)):
            for xc, yc, r in circles:
                if math.hypot(abs(x-xc),abs(y-yc)) <= r:
                    cnt+=1
                    break
        return cnt
```

## 2250. Count Number of Rectangles Containing Each Point

### Solution 1: sort + binary search on the large width, and iterate through the small heights, height << width

```py

class Solution:
    def countRectangles(self, rectangles: List[List[int]], points: List[List[int]]) -> List[int]:
        n=len(points)
        counts = [0]*n
        rects = defaultdict(list)
        for l, h in rectangles:
            rects[h].append(l)
        heights = sorted([h for h in rects.keys()])
        for h in heights:
            rects[h].sort()
        for i, (x, y) in enumerate(points):
            hstart = bisect_left(heights, y)
            for j in range(hstart, len(heights)):
                cur_h = heights[j]
                k = len(rects[cur_h]) - bisect_left(rects[cur_h], x)
                counts[i] += k
        return counts
```

## 2251. Number of Flowers in Full Bloom

### Solution 1: line sweep with pointer for persons + sort

```py
class Solution:
    def fullBloomFlowers(self, flowers: List[List[int]], persons: List[int]) -> List[int]:
        n = len(persons)
        persons = sorted([(p, i) for i, p in enumerate(persons)])
        answer = [0]*n
        events = []
        for s, e in flowers:
            events.append((s, 1))
            events.append((e+1,-1))
        events.sort()
        i = 0
        bloomed = 0 
        for ev, delta in events:
            while i < n and persons[i][0] < ev:
                p, index  = persons[i]
                answer[index] = bloomed
                i+=1
            bloomed += delta
            if i == n: break
        return answer
        
```

### Solution 2: two binary search for start and end

```py
class Solution:
    def fullBloomFlowers(self, flowers: List[List[int]], persons: List[int]) -> List[int]:
        start, end = sorted(s for s, e in flowers), sorted(e for s,e in flowers)
        return [bisect_right(start,time) - bisect_left(end, time) for time in persons]
```