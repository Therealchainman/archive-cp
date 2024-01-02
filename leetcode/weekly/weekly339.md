# Leetcode Weekly Contest 339

## 2609. Find the Longest Balanced Substring of a Binary String

### Solution 1:  groupby to get the 0s and 1s together + use the min of the groups

```py
class Solution:
    def findTheLongestBalancedSubstring(self, s: str) -> int:
        res = cur = 0
        for key, grp in groupby(s):
            cnt = len(list(grp))
            if key == '0':
                cur = cnt
            else:
                cur = min(cur, cnt)
                res = max(res, 2*cur)
        return res
```

## 2610. Convert an Array Into a 2D Array With Conditions

### Solution 1:  implementation

```py
class Solution:
    def findMatrix(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        last = [0] * (n + 1)
        ans = []
        for num in nums:
            if len(ans) == last[num]:
                ans.append([])
            ans[last[num]].append(num)
            last[num] += 1
        return ans              
```

## 2611. Mice and Cheese

### Solution 1:  sorted + difference array

The idea to give the ones with largest difference is because the difference is calculating when you get the most value from having mouse 1 eat the cheese at index i.  So it makes sense to give it to mouse 1, since you only going to give k, then give the rest to mouse 2. 

```py
class Solution:
    def miceAndCheese(self, reward1: List[int], reward2: List[int], k: int) -> int:
        n = len(reward1)
        diff = sorted([(r1 - r2, i) for i, (r1, r2) in enumerate(zip(reward1, reward2))], reverse = True)
        res = 0
        vis = [0]*n
        for i in range(k):
            index = diff[i][1]
            vis[index] = 1
            res += reward1[index]
        return res + sum([reward2[i] for i in range(n) if not vis[i]])
```

### Solution 2:  nlargest + zip + sum

Observe that you only need the k largest elements from the difference array and add them to the sum of giving all of it to mouse 2.

```py
class Solution:
    def miceAndCheese(self, reward1: List[int], reward2: List[int], k: int) -> int:
        return sum(reward2) + sum(nlargest(k, (r1 - r2 for r1, r2 in zip(reward1, reward2))))
        
```

## 2612. Minimum Reverse Operations

### Solution 1:  bfs + sortedlist + parity + O(nlogn) 

Basic idea is that whenever you are given a range of index, you will take increments of two

This is the pattern for the right side, but note it is symmetric so it does the same moving to the left of the current index as well. 

But basically the observation is if you are at i, you increment by 2, so therefore just need to figure out parity and then go through available nodes

k
1  i
2  i+1
3  i, i+2
4  i+1, i+3
5  i, i+2, i+4
6  i+1, i+3, i+5
7  i, i+2, i+4, i+6

To store available nodes put them in two sortedlist so that it follows the parity, so all index even in 0th element and all index odd in 1st element, which is just adding them to a sortedlist.  Reason for sortedlist is so that can remove elements from it in O(logn) time.  This is way to prevent rechecking already visited nodes. No reason to revisit, already found minimun reverse operations.

You begin a bfs from the current p.

Then for each pos you find the left and right bounds, which is tricky to derive, but write it down on a piece of paper and you can derive it.  

for instance k = 4 you have

so looking at the 1,2,3,4 step in here, you have these bounds that are need to move 1 to p-3 position or to p + 3 position and so on. 

4 [p, p+3] p+3
3 [p-1, p+2] p+1 
2 [p-2, p+1] p-1
1 [p-3, p] p-3

Now consider a formula to derive the left bounds for the index, is it p-3, or p-1, or p+1

Well if you push it back as far as you can so pos - k + 1, this will get it to as far back it can be, 
Then there is a pattern of increment based on the left point being all the way over there

p    p+3  +3
p-1  p+1  +2
p-2  p-1  +1
p-3  p-3  +0

see there is a  pattern if you push it all the way back, and it is basically that it the different increments by 1 for each iteration farther away

so for example if p+1 should be left bound, and I've pushed left to p-1 then I need to do p-1 + 2, but how can I find 2

it can be found by taking k - 1 + left - pos, cause basically k - 1 + p - 1 - p = k - 1 - 1 = 4 - 2 = 2, which is correct, and think about it, basically as left decreases, the delta becomes

so increment is k - 1 + left - pos so you can find current left position for the neighbors by taking left + increment

In addition you can find right pointer in similar manner, just take 
right = min(n - 1, pos + k - 1) - k + 1, so basically when you add the k it will be the right point above, but so you can use the same formula as above, just move it back k - 1. 

This way it follows the same pattern above, and can be calcualted in same way, and you can find the appropriate max. 

but basically you find the leftmost you can go, and the rightmost, cause that is all it can visit.

```py
from sortedcontainers import SortedList

class Solution:
    def minReverseOperations(self, n: int, p: int, banned: List[int], k: int) -> List[int]:
        nodes = [SortedList(), SortedList()]
        banned = set(banned)
        for i in range(n):
            if i == p or i in banned: continue
            nodes[i%2].add(i)
        queue = deque([p])
        dist = [-1]*n
        dist[p] = 0
        while queue:
            pos = queue.popleft()
            left = max(0, pos - k + 1)
            left = 2*left + k - 1 - pos
            right = min(n - 1, pos + k - 1) - k + 1
            right = 2*right + k - 1 - pos
            used = []
            for nei_pos in nodes[left%2].irange(left, right):
                dist[nei_pos] = dist[pos] + 1
                queue.append(nei_pos)
                used.append(nei_pos)
            for i in used:
                nodes[left%2].remove(i)
        return dist
```