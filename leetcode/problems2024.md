## 1043. Partition Array for Maximum Sum

### Solution 1:  dynammic programming, O(n^2)

dp[i] = maximum sum of partitioning arr[:i + 1] into segments of length at most k when setting the values equal to the max value in each segment. 

For each position i it computes the maximum sum that can be achieved by partitioning the array up to and including the ith element.

Then it increases the size of the current partition that includes i, by moving the j pointer back until it reaches the max size of k.  And it tracks the maximum element in that partition, as that will be the value of all elements in the partition.  And then it computes the value by taking the maximum sum of the partition up to j, and adding the value of the partition to the sum.  And then it updates the dp[i + 1] with the maximum value of the partition.

```py
class Solution:
    def maxSumAfterPartitioning(self, arr: List[int], k: int) -> int:
        n = len(arr)
        dp = [-math.inf] * (n + 1)
        dp[0] = 0
        for i in range(n):
            segmax = -math.inf
            for j in range(i, max(-1, i - k), -1):
                segmax = max(segmax, arr[j])
                dp[i + 1] = max(dp[i + 1], dp[j] + (i - j + 1) * segmax)
        return dp[-1]
```

## 49. Group Anagrams

### Solution 1:  sort, groupby, counter

```py
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        ans = []
        prev = Counter({"i": -1})
        for s in sorted(strs, key = lambda x: sorted(list(x))):
            freq = Counter(s)
            if prev == freq:
                ans[-1].append(s)
            else:
                ans.append([s])
            prev = freq
        return ans
```

```py
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        ans = []
        strs.sort(key = sorted)
        for k, grp in groupby(strs, key = sorted):
            ans.append(list(grp))
        return ans
```

## 368. Largest Divisible Subset

### Solution 1:  sort, dynamic programming, parent array to track best path, backtrack in parent array

```py
class Solution:
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        nums.sort()
        n = len(nums)
        dp = [0] * n
        parent = [-1] * n
        for i in range(n):
            for j in range(i):
                if nums[i] % nums[j] == 0 and dp[j] >= dp[i]:
                    dp[i] = dp[j] + 1
                    parent[i] = j
        ans = []
        i = max(range(n), key = lambda i: dp[i])
        while i != -1:
            ans.append(nums[i])
            i = parent[i]
        return ans
```

## 1463. Cherry Pickup II

### Solution 1:  iterative dp, space optimized, maximize
(column robot 1 occupies, column robot 2 occupies)
And just compute maximum for every possible transition.  

```py
class Solution:
    def cherryPickup(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        dp = [[-math.inf] * C for _ in range(C)]
        dp[0][-1] = grid[0][0] + grid[0][-1]
        in_bounds = lambda c: 0 <= c < C
        for r in range(1, R):
            ndp = [[-math.inf] * C for _ in range(C)]
            for c1, c2 in product(range(C), repeat = 2):
                if dp[c1][c2] == -math.inf: continue
                for nc1, nc2 in product(range(c1 - 1, c1 + 2), range(c2 - 1, c2 + 2)):
                    if not in_bounds(nc1) or not in_bounds(nc2): continue
                    ndp[nc1][nc2] = max(ndp[nc1][nc2], dp[c1][c2] + grid[r][nc1] + (grid[r][nc2] if nc1 != nc2 else 0))
            dp = ndp
        return max(max(row) for row in dp)
```

## 169. Majority Element

### Solution 1:  Boyer-Moore Voting Algorithm

```py
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        n = len(nums)
        ans = cnt = 0
        for num in nums:
            if cnt == 0: ans = num
            if ans == num: cnt += 1
            else: cnt -= 1
        return ans
```

## 1481. Least Number of Unique Integers after K Removals

### Solution 1:  count, sort

```py
class Solution:
    def findLeastNumOfUniqueInts(self, arr: List[int], k: int) -> int:
        freq = sorted(Counter(arr).values(), reverse = True)
        while k > 0:
            x = freq.pop()
            k -= x
            if k < 0: freq.append(x)
        return len(freq)
```

## 201. Bitwise AND of Numbers Range

### Solution 1:  bit manipulation

![image](images/bitwise_and_range.png)

Observation 1:
All the bits to the right of a flipped bit between left and right will also be flipped in the range.

```py
class Solution:
    def rangeBitwiseAnd(self, left: int, right: int) -> int:
        ans = 0
        try:
            start = next(dropwhile(lambda i: not ((right >> i) & 1), reversed(range(32))))
        except:
            return ans
        for i in range(start, -1, -1):
            if (right >> i) != (left >> i): break
            if (right >> i) & 1: ans |= (1 << i)
        return ans
```

## 2092. Find All People With Secret

### Solution 1:  undirected graph, dfs

Form an many undirected graphs at each time step.  And the person who has a secret spreads to everyone they can reach.  So use a dfs through the graph from each person that knows a secret let it flow to everyone. 

```py
class Solution:
    def findAllPeople(self, n: int, meetings: List[List[int]], firstPerson: int) -> List[int]:
        know = [0] * n
        know[0] = know[firstPerson] = 1
        edge_lists = defaultdict(list)
        for u, v, t in sorted(meetings, key = lambda pair: pair[-1]):
            edge_lists[t].append((u, v))
        def dfs(src):
            stk = [src]
            vis.add(src)
            while stk:
                u = stk.pop()
                know[u] = 1
                for v in adj[u]:
                    if v in vis: continue
                    vis.add(v)
                    stk.append(v)
        for edges in edge_lists.values():
            adj = defaultdict(list)
            nodes, vis = set(), set()
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
                nodes.update([u, v])
            for u in nodes:
                if u in vis: continue
                if not know[u]: continue
                dfs(u)
        return [i for i in range(n) if know[i]]
```

## 543. Diameter of Binary Tree

### Solution 1: recursion, dfs

```py
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        ans = 0
        def dfs(u):
            nonlocal ans
            if not u: return 0
            llen, rlen = dfs(u.left), dfs(u.right)
            ans = max(ans, llen + rlen)
            return max(llen, rlen) + 1
        dfs(root)
        return ans
```

## 513. Find Bottom Left Tree Value

### Solution 1:  bfs

```py
class Solution:
    def findBottomLeftValue(self, root: Optional[TreeNode]) -> int:
        q = deque([root])
        while q:
            first = q[0].val
            for _ in range(len(q)):
                u = q.popleft() 
                q.extend(filter(None, (u.left, u.right)))
        return first
```

## 1609. Even Odd Tree

### Solution 1:  hash map, dfs

```py
class Solution:
    def isEvenOddTree(self, root: Optional[TreeNode]) -> bool:
        last = {}
        def dfs(u, depth = 0):
            if not u: return True
            if depth & 1:
                if u.val & 1: return False
                if u.val >= last.get(depth, math.inf): return False
                last[depth] = u.val
            else:
                if u.val % 2 == 0: return False
                if u.val <= last.get(depth, -math.inf): return False
                last[depth] = u.val
            return dfs(u.left, depth + 1) and dfs(u.right, depth + 1)
        return dfs(root)
```

## 1750. Minimum Length of String After Deleting Similar Ends

### Solution 1:  two pointers 

```py
class Solution:
    def minimumLength(self, s: str) -> int:
        l, r = 0, len(s) - 1
        while l < r:
            if s[l] != s[r]: break 
            ch = s[l]
            while l <= r and s[l] == ch: l += 1 
            while l <= r and s[r] == ch: r -= 1
        return max(0, r - l + 1)
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