## 271. Encode and Decode Strings

### Solution 1:  Chunked Transfer Encoding + string

Stores the size of each chunk as a prefix and then there is the delimiter '#'.

```py
class Codec:
    def encode(self, strs: List[str]) -> str:
        """Encodes a list of strings to a single string.
        """
        n = len(strs)
        result = [None] * n
        for i in range(n):
            size = len(strs[i])
            result[i] = f"{size}#{strs[i]}"
        return "".join(result)

    def decode(self, s: str) -> List[str]:
        """Decodes a single string to a list of strings.
        """
        result = []
        i = 0
        while i < len(s):
            j = s.find('#', i)
            size = int(s[i:j])
            str_ = s[j + 1:j + 1 + size]
            result.append(str_)
            i = j + 1 + size
        return result
        
# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.decode(codec.encode(strs))
```

## 920. Number of Music Playlists

### Solution 1:  dynamic programming

dp[i][j] = number of playlists of length i that have exactly j unique songs

Two transition states:
1. play song
Play a new song that has not been played before this will increase the distinct songs by one
You need to determine how many songs you can play in this instance, it is going to be the total number of songs - number of unique songs played so far + 1. For example, 
xxxx 4 unique songs played, and there are n = 10 total songs, then you can play either song 5,6,7,8,9,10, which is 10 - 4 = 6 songs
The transition state looks like dp[i - 1][j - 1] * (n - (j - 1)), because you are coming from state with j - 1 unique songs played.
2. replay song
Play a song that has been played before, this will not increase the distinct songs played
Because you can only play a replayed song after k other songs you need to consider this
if j = 5, so 5 unique songs played 
and k = 3, so you need to play 3 songs between so for instance, 1,2,3,4,5,x => you can only replay 1 and 2 so that means there are 2 songs you can replay for 5 unique songs always, 
because there must be 3 songs in window that are distinct, so j - k is the songs you can play in this scenario so multipy by that
you get dp[i - 1][j] * (j - k)

why is it multiplication? 
Because at each state you have x possible songs you can play, and so if there are 4 ways to get to thst state, you can now take those 4 ways call them x1, x2, x3, x4
and if x = 3
you can now add 1 to end of all 4 states and 2 to end of all 4 states and 3 to end of all 4 states, so that is 3 * 4 = 12, or x * num_ways
another way I think of it is take this

_ _ _ _ _
        ^
      4
so you know there are 4 ways to fill in the first 4 slots, now for the current slot you are at, if you have x choices, then you are going to add it to end of all the previous 4 ways, so you get 4 * x ways now


```py
class Solution:
    def numMusicPlaylists(self, n: int, goal: int, k: int) -> int:
        mod = int(1e9) + 7
        dp = [[0] * (n + 1) for _ in range(goal + 1)]
        dp[0][0] = 1
        for i, j in product(range(1, goal + 1), range(1, n + 1)):
            dp[i][j] = (dp[i - 1][j - 1] * (n - j + 1) + dp[i - 1][j] * max(j - k, 0)) % mod
        return dp[goal][n]
```

### Solution 2:  math + combinatorics + inclusion exclusion principle

```py

```

## 1378. Replace Employee ID With The Unique Identifier

### Solution 1:  left join + merge in pandas

```py
import pandas as pd

def replace_employee_id(employees: pd.DataFrame, employee_uni: pd.DataFrame) -> pd.DataFrame:
    df = (
        employees
        .merge(employee_uni, how = 'left', on = 'id')
        .drop(columns = ['id'])
    )
    return df
```

## 81. Search in Rotated Sorted Array II

### Solution 1:  binary search + linear search when stuck

```py
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        n = len(nums)
        left, right = 0, n - 1
        while left < right:
            mid = (left + right + 1) >> 1
            if nums[right] == nums[left] == nums[mid]:
                while left < right and nums[left] == nums[mid]:
                    left += 1
                left -= 1
                while left < right and nums[right] == nums[mid]:
                    right -= 1
            elif nums[right] <= nums[left] <= nums[mid]:
                if target >= nums[mid] or target <= nums[right]:
                    left = mid
                else:
                    right = mid - 1
            elif nums[mid] <= nums[right] <= nums[left]:
                if nums[mid] <= target <= nums[right]:
                    left = mid
                else:
                    right = mid - 1
            else:
                if target >= nums[mid]:
                    left = mid
                else:
                    right = mid - 1
        return nums[left] == target
```

### Solution 2: Binary search

Binary search but with two arrays you have array S and array F, both are 
non-decreasing arrays.  But normall the arrays are array S + array F, but with the 
pivot point it get's rotated and you have array F + array S

The following you just need to consider 4 cases to solve the problem

special case: This is for it you can't determine which array mid belongs in, which is the case 
when nums[mid]==nums[lo], because you don't know which array it belongs to.
case 1: if mid in F and target in F, then you just need to look at comparison of target to nums[mid]
case 2: if mid in S and target in S, then you just need to look at comparison of target to nums[mid]
case 3: if mid in F and target in S, then you need to look to right of mid
case 3: if mid in S and target in F, then you need to look to left of mid



```py
class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        """
        Break it down to two arrays
        [F][S]
        """
        def can_binary_search(start_val, cur_val):
            return cur_val != start_val
        # in the F array if this is true
        def find_arr(start_val, cur_val):
            return cur_val>=start_val
        lo, hi = 0, len(nums)-1
        while lo < hi:
            mid = (lo+hi)>>1
            if nums[mid]==target: return True
            if not can_binary_search(nums[lo],nums[mid]):
                lo += 1
                continue
            target_arr = find_arr(nums[lo], target)
            mid_arr = find_arr(nums[lo], nums[mid])
            if target_arr ^ mid_arr:
                if mid_arr:
                    lo = mid+1
                else:
                    hi = mid
            else:
                if nums[mid]<target:
                    lo=mid+1
                else:
                    hi=mid
        return nums[lo] == target
```

## 2814. Minimum Time Takes to Reach Destination Without Drowning

### Solution 1:  multisource bfs + single source bfs

Need a bfs for the flood and the person, and update flood and then position of person

```py
class Solution:
    def minimumSeconds(self, land: List[List[str]]) -> int:
        start, target, empty, stone, flood = 'S', 'D', '.', 'X', '*'
        R, C = len(land), len(land[0])
        frontier, queue = deque(), deque()
        for r, c in product(range(R), range(C)):
            if land[r][c] == start: queue.append((r, c))
            elif land[r][c] == flood: frontier.append((r, c))
        in_bounds = lambda r, c: 0 <= r < R and 0 <= c < C
        neighborhood = lambda r, c: [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        steps = 0
        while queue:
            # update the flooded cells
            for _ in range(len(frontier)):
                r, c = frontier.popleft()
                for nr, nc in neighborhood(r, c):
                    if not in_bounds(nr, nc) or land[nr][nc] not in (empty, start): continue
                    land[nr][nc] = flood
                    frontier.append((nr, nc))
            # update possible places you can be
            steps += 1
            for _ in range(len(queue)):
                r, c = queue.popleft()
                if land[r][c] == target: return steps
                for nr, nc in neighborhood(r, c):
                    if not in_bounds(nr, nc) or land[nr][nc] not in (empty, target): continue
                    if land[nr][nc] == target: return steps
                    land[nr][nc] = start
                    queue.append((nr, nc))
        return -1
```

## 215. Kth Largest Element in an Array

### Solution 1:  nlargest + heapq

```py
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        return nlargest(k, nums)[-1]
```

##

### Solution 1:

```py

```

##

### Solution 1:

```py

```

##

### Solution 1:

```py

```

##

### Solution 1:

```py

```