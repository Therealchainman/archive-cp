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

## 791. Custom Sort String

### Solution 1:  sort with key, string find

```py
class Solution:
    def customSortString(self, order: str, s: str) -> str:
        return "".join(sorted(s, key = lambda ch: order.find(ch)))
```

## 1171. Remove Zero Sum Consecutive Nodes from Linked List

### Solution 1: linked list, sentinel node, start and end node for range and prefix sum

```py
class Solution:
    def removeZeroSumSublists(self, head: Optional[ListNode]) -> Optional[ListNode]:
        front = ListNode(0, head)
        start = front
        while start:
            psum = 0
            end = start.next
            while end:
                psum += end.val
                if psum == 0: start.next = end.next
                end = end.next
            start = start.next
        return front.next
```

## 930. Binary Subarrays With Sum

### Solution 1:  sliding window, frequency array, natural sequence sum formula

```py
class Solution:
    def numSubarraysWithSum(self, nums: List[int], goal: int) -> int:
        f = lambda n: n * (n + 1) // 2
        if goal == 0: return sum(f(len(list(grp))) for k, grp in groupby(nums) if k == 0)
        n = len(nums)
        freq = [1] * (n + 1)
        ans = 0
        for i in reversed(range(n)):
            if nums[i] == 0: freq[i] = freq[i + 1] + 1
        j = wsum = 0
        for i in range(n):
            wsum += nums[i]
            while wsum > goal:
                wsum -= nums[j]
                j += 1
            if wsum == goal: ans += freq[j]
        return ans
```

## 238. Product of Array Except Self

### Solution 1:  prefix and suffix multiplication

```py
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        ans = list(accumulate(reversed(nums), func = operator.mul))
        ans.reverse()
        pmul = 1
        for i in range(n):
            ans[i] = pmul * (ans[i + 1] if i + 1 < n else 1)
            pmul *= nums[i]
        return ans
```

## 452. Minimum Number of Arrows to Burst Balloons

### Solution 1: line sweep, greedy, stack

```py
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        START = -1
        END = 1
        n = len(points)
        events = []
        for i, (s, e) in enumerate(points):
            events.append((s, START, i))
            events.append((e, END, i))
        events.sort()
        arrows = [-1] * n
        stk = []
        shots = 1
        for _, d, i in events:
            if d == END and arrows[i] == -1: # set of balloons not burst yet
                while stk: # burst everything that has been seen
                    j = stk.pop()
                    arrows[j] = shots
                shots += 1 # will need 1 more shot for any more balloons not burst by this shot
            elif d == START:
                stk.append(i)
        return max(arrows)
```

## 621. Task Scheduler

### Solution 1:  maxheap, array, greedy

place the most frequent task first in each cycle, cycles are of length n + 1, save in array for characters so they are not used more than once in cycle, but will be added back into max heap after the cycle. 

```py
class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        freq = Counter(tasks)
        heapify(maxheap := [(-freq[ch], ch) for ch in string.ascii_uppercase if freq[ch] > 0])
        ans = cur = 0
        while maxheap:
            stk = []
            for i in range(n + 1):
                cur += 1
                if not maxheap: continue
                _, ch = heappop(maxheap)
                ans = cur
                freq[ch] -= 1
                stk.append(ch)
            for ch in stk:
                if freq[ch] > 0: heappush(maxheap, (-freq[ch], ch))
        return ans
```

## 41. First Missing Positive

### Solution 1: answer in range [1,n], swap elements to correct index

```py
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        nums.append(0)
        n = len(nums)
        for i in range(n):
            index = None
            while 0 <= nums[i] < n and nums[i] != index:
                index = nums[i]
                nums[index], nums[i] = nums[i], nums[index]
        for i in range(1, n):
            if nums[i] != i: return i
        return n
```

## 713. Subarray Product Less Than K

### Solution 1:  two pointers, prefix calculation with multiplication

```py
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        n = len(nums)
        ans = j = 0
        pmul = 1
        for i in range(n):
            pmul *= nums[i]
            while j <= i and pmul >= k:
                pmul //= nums[j]
                j += 1
            ans += i - j + 1
        return ans
```

## 992. Subarrays with K Different Integers

### Solution 1:  two sliding windows, two pointers, frequency array

```py
class Solution:
    def subarraysWithKDistinct(self, nums: List[int], k: int) -> int:
        p1 = p2 = ans = d1 = d2 = 0
        n = len(nums)
        f1, f2 = [0] * (n + 1), [0] * (n + 1)
        for i in range(n):
            f1[nums[i]] += 1
            f2[nums[i]] += 1
            if f1[nums[i]] == 1: d1 += 1
            if f2[nums[i]] == 1: d2 += 1
            while d1 > k:
                f1[nums[p1]] -= 1
                if f1[nums[p1]] == 0: d1 -= 1
                p1 += 1
            while d2 >= k:
                f2[nums[p2]] -= 1
                if f2[nums[p2]] == 0: d2 -= 1
                p2 += 1
            ans += p2 - p1
        return ans
```

## 678. Valid Parenthesis String

### Solution 1: two pointers, greedy

```py
class Solution:
    def checkValidString(self, s: str) -> bool:
        lo = hi = 0
        for i, ch in enumerate(s):
            if ch == "(": 
                lo += 1
                hi += 1
            elif ch == ")":
                lo -= 1
                hi -= 1
            else:
                lo -= 1
                hi += 1
            if hi < 0: return False
            lo = max(0, lo)
        return lo == 0
```

## 950. Reveal Cards In Increasing Order

### Solution 1: simulation deque

```py
class Solution:
    def deckRevealedIncreasing(self, deck: List[int]) -> List[int]:
        n = len(deck)
        pos = [0] * n
        dq = deque(range(n))
        i = 0
        while dq:
            ndq = deque()
            while dq:
                v = dq.popleft()
                if i % 2 == 0:
                    pos[v] = i // 2
                else:
                    ndq.append(v)
                i += 1
            dq = ndq
        deck.sort()
        ans = [deck[pos[i]] for i in range(n)]
        return ans
```

## 402. Remove K Digits

### Solution 1:  monotonic queue, deque

```py
class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        num += "0"
        q = deque()
        for x in map(int, num):
            while k > 0 and q and q[-1] > x:
                q.pop()
                k -= 1
            q.append(x)
        while len(q) > 1 and q[0] == 0: q.popleft() # remove leading 0s
        if len(q) > 1: q.pop() # remove artificial tail "0"
        return "".join(map(str, q))
```

## 2953. Count Complete Substrings

### Solution 1:  fixed sized sliding window, frequency, difference array

```py
class Solution:
    def count(self, s, e, word, sz, k):
        len_ = e - s
        under_count = over_count = ans = 0
        if sz > len_: return ans
        freq = [0] * 26
        unicode = lambda ch: ord(ch) - ord("a")
        for i in range(s, e):
            v = unicode(word[i])
            freq[v] += 1
            if freq[v] == 1: under_count += 1
            if freq[v] == k: under_count -= 1
            elif freq[v] == k + 1: over_count += 1
            if i >= s + sz - 1:
                if over_count == under_count == 0: ans += 1
                v = unicode(word[i - sz + 1])
                freq[v] -= 1
                if freq[v] == k: over_count -= 1
                elif freq[v] == k - 1: under_count += 1
                if freq[v] == 0: under_count -= 1
        return ans
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        n = len(word)
        diff_arr = [0] * n
        unicode = lambda ch: ord(ch) - ord("a")
        chdiff = lambda i, j: abs(unicode(word[i]) - unicode(word[j]))
        for i in range(n - 1):
            diff_arr[i] = chdiff(i, i + 1)
        diff_arr[-1] = 3
        queries = []
        start = 0
        for end in range(n):
            if diff_arr[end] > 2:
                queries.append((start, end + 1))
                start = end + 1
        ans = 0
        for nc in range(1, 27):
            sz = nc * k
            for s, e in queries:
                ans += self.count(s, e, word, sz, k)
        return ans
```

## 2954. Count the Number of Infection Sequences

### Solution 1:  dp, counting, multinomial coefficient, factorials

```py
MOD = int(1e9) + 7
def mod_inverse(num):
    return pow(num, MOD - 2, MOD)
def factorials(n):
    fact = [1]*(n + 1)
    for i in range(1, n + 1):
        fact[i] = (fact[i - 1] * i) % MOD
    inv_fact = [1]*(n + 1)
    inv_fact[-1] = mod_inverse(fact[-1])
    for i in range(n - 1, -1, -1):
        inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % MOD
    return fact, inv_fact
class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        blocks = []
        sick.insert(0, -1)
        sick.append(n)
        for i in range(1, len(sick)):
            blocks.append(sick[i] - sick[i - 1] - 1)
        fact, inv_fact = factorials(n)
        ans = fact[sum(blocks)]
        m = len(blocks)
        for i in range(m):
            ans = (ans * inv_fact[blocks[i]]) % MOD
            if 0 < i < m - 1: ans = (ans * pow(2, max(0, blocks[i] - 1), MOD)) % MOD
        return ans
```

## 2977. Minimum Cost to Convert String II

### Solution 1:  dijkstra, rolling hash, minimize dp

TLE

```py
def dijkstra(adj, src, dst):
    N = len(adj)
    min_heap = [(0, src)]
    vis = set()
    while min_heap:
        cost, u = heapq.heappop(min_heap)
        if u == dst: return cost
        if u in vis: continue
        vis.add(u)
        for v, w in adj[u]:
            if v in vis: continue
            heapq.heappush(min_heap, (cost + w, v))
    return math.inf

class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        p, MOD1, MOD2 = 31, int(1e9) + 7, int(1e9) + 9
        coefficient = lambda x: ord(x) - ord('a') + 1
        transformation = {}
        add = lambda h, mod, ch: ((h * p) % mod + coefficient(ch)) % mod
        n = len(source)
        for l in range(n):
            hash1 = hash2 = 0
            for r in range(l, n):
                hash1 = add(hash1, MOD1, source[r]) 
                hash2 = add(hash2, MOD1, target[r])
                transformation[hash1] = hash2
        edges = defaultdict(lambda: math.inf)
        for u, v, w in zip(original, changed, cost):
            hash1 = hash2 = 0
            for i in range(len(u)):
                hash1 = add(hash1, MOD1, u[i]) 
                hash2 = add(hash2, MOD1, v[i])
            edges[(hash1, hash2)]
            edges[(hash1, hash2)] = min(edges[(hash1, hash2)], w)
        adj = defaultdict(list)
        for (u, v), w in edges.items():
            adj[u].append((v, w))
        transitions = [[math.inf] * n for _ in range(n)]
        for l in range(n):
            hash1 = hash2 = 0
            for r in range(l, n):
                hash1 = add(hash1, MOD1, source[r]) 
                hash2 = add(hash2, MOD1, target[r])
                if hash1 == hash2: transitions[l][r] = 0
                else: transitions[l][r] = dijkstra(adj, hash1, hash2)
        dp = [math.inf] * n
        for r in range(n):
            for l in range(r + 1):
                cur = transitions[l][r]
                if l > 0: cur += dp[l - 1]
                dp[r] = min(dp[r], cur)
        return dp[-1] if dp[-1] < math.inf else -1
```

## 3008. Find Beautiful Indices in the Given Array II

### Solution 1:  z algorithm, string matching, two pointers

```py
def z_algorithm(s: str) -> list[int]:
    n = len(s)
    z = [0]*n
    left = right = 0
    for i in range(1,n):
        # BEYOND CURRENT MATCHED SEGMENT, TRY TO MATCH WITH PREFIX
        if i > right:
            left = right = i
            while right < n and s[right-left] == s[right]:
                right += 1
            z[i] = right - left
            right -= 1
        else:
            k = i - left
            # IF PREVIOUS MATCHED SEGMENT IS NOT TOUCHING BOUNDARIES OF CURRENT MATCHED SEGMENT
            if z[k] < right - i + 1:
                z[i] = z[k]
            # IF PREVIOUS MATCHED SEGMENT TOUCHES OR PASSES THE RIGHT BOUNDARY OF CURRENT MATCHED SEGMENT
            else:
                left = i
                while right < n and s[right-left] == s[right]:
                    right += 1
                z[i] = right - left
                right -= 1
    return z

class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        n, na, nb = len(s), len(a), len(b)
        z_arr_a = z_algorithm(a + "$" + s)[na + 1:]
        z_arr_b = z_algorithm(b + "$" + s)[nb + 1:]
        arra = [i for i, x in enumerate(z_arr_a) if x == na]
        arrb = [i for i, x in enumerate(z_arr_b) if x == nb]
        j = 0
        ans = []
        for i in arra:
            while j < len(arrb) and arrb[j] < i and i - arrb[j] > k: j += 1
            if j == len(arrb): break
            if abs(i - arrb[j]) <= k: ans.append(i)
        return ans
```

## 3049. Earliest Second to Mark Indices II

### Solution 1:

```py

```

## 3017. Count the Number of Houses at a Certain Distance II

### Solution 1:

```py

```

## 2968. Apply Operations to Maximize Frequency Score

### Solution 1:  binary search size of subarray, sort, rolling median deivation, prefix sum

```py
class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        n = len(nums)
        nums.sort()
        psum = list(accumulate(nums))
        def sum_(i, j):
            return psum[j] - (psum[i - 1] if i > 0 else 0)
        def deviation(i, j, mid):
            lsum = (mid - i + 1) * nums[mid] - sum_(i, mid)
            rsum = sum_(mid, j) - (j - mid + 1) * nums[mid]
            return lsum + rsum
        def RMD(nums, k): # rolling median deviation
            n = len(nums)
            ans = math.inf
            l = 0
            for r in range(k - 1, n):
                mid = (l + r) >> 1
                ans = min(ans, deviation(l, r, mid))
                if k % 2 == 0:
                    ans = min(ans, deviation(l, r, mid - 1))
                l += 1
            return ans
        l, r = 1, n
        while l < r:
            m = (l + r + 1) >> 1
            if RMD(nums, m) <= k:
                l = m
            else:
                r = m - 1
        return l
```

## 752. Open the Lock

### Solution 1:  bfs, deque, hash table, set

```py
class Solution:
    def openLock(self, deadends: List[str], target: str) -> int:
        n = len(deadends)
        q = deque([[0, 0, 0, 0]])
        vis = set(deadends)
        start = "0000"
        if target == start: return 0
        if start in vis: return -1
        vis.add(start)
        cnt = 0
        def create_lock(i, delta):
            lock[i] = (lock[i] + delta) % 10
            new_lock = lock[::]
            lock[i] = (lock[i] - delta) % 10
            return new_lock
        while q:
            cnt += 1
            for _ in range(len(q)):
                lock = q.popleft()
                for i in range(4):
                    new_lock = create_lock(i, 1)
                    key = "".join(map(str, new_lock))
                    if key not in vis: 
                        if key == target: return cnt
                        vis.add(key)
                        q.append(new_lock)
                    new_lock = create_lock(i, -1)
                    key = "".join(map(str, new_lock))
                    if key not in vis: 
                        if key == target: return cnt
                        vis.add(key)
                        q.append(new_lock)
        return -1
```

## 1289. Minimum Falling Path Sum II

### Solution 1:  dynamic programming, space optimized, save two minimums

```py
class Solution:
    def minFallingPathSum(self, grid: List[List[int]]) -> int:
        N = len(grid)
        dp1 = dp2 = 0
        i1 = i2 = -1
        for row in grid:
            ndp1 = ndp2 = math.inf
            ni1 = ni2 = -1
            for i in range(N):
                if i != i1:
                    cand = dp1 + row[i]
                else:
                    cand = dp2 + row[i]
                if cand < ndp1:
                    ndp2 = ndp1
                    ni2 = ni1
                    ndp1 = cand
                    ni1 = i
                elif cand < ndp2:
                    ndp2 = cand
                    ni2 = i
            dp1, dp2 = ndp1, ndp2
            i1, i2 = ni1, ni2
        return dp1
```

## 506. Relative Ranks

### Solution 1:  dictionary, sort

```py
class Solution:
    def findRelativeRanks(self, score: List[int]) -> List[str]:
        n = len(score)
        ans = [None] * n
        place = {}
        place[1] = "Gold Medal"
        place[2] = "Silver Medal"
        place[3] = "Bronze Medal"
        for r, i in enumerate(sorted(range(n), key = lambda x: score[x], reverse = True), start = 1):
            ans[i] = place.get(r, str(r))
        return ans
```

## 786. K-th Smallest Prime Fraction

### Solution 1:  binary search, floats, two pointers

```py
class Solution:
    def kthSmallestPrimeFraction(self, arr: List[int], k: int) -> List[int]:
        n = len(arr)
        def possible(target):
            x, y = 1, 100_000
            j = ans = 0
            for i in range(n):
                j = i + 1
                while j < n and arr[i] / arr[j] >= target: j += 1
                if j < n:
                    ans += n - j
                    if arr[i] * y > x * arr[j]: x, y = arr[i], arr[j]
            return ans, x, y
        lo, hi = 0, 1.0
        while lo < hi:
            mid = (lo + hi) / 2
            cnt, x, y = possible(mid)
            if cnt == k: return [x, y]
            if cnt < k:
                lo = mid
            else:
                hi = mid
        return []
```

## 861. Score After Flipping Matrix

### Solution 1:  greedy, bit manipulation, binary

```py
class Solution:
    def matrixScore(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        for i in range(R):
            if grid[i][0]: continue
            for j in range(C):
                grid[i][j] ^= 1
        col_count = [[0] * 2 for _ in range(C)]
        for r, c in product(range(R), range(C)):
            col_count[c][grid[r][c]] += 1
        for r, c in product(range(R), range(C)):
            if col_count[c][0] > col_count[c][1]: grid[r][c] ^= 1
        ans = sum(int("".join(map(str, row)), 2) for row in grid)
        return ans
```

## 1219. Path with Maximum Gold

### Solution 1:  undirected graph, bitmask dynamic programming, counter, index compression

```py
class Solution:
    def getMaximumGold(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        nodes = {(-1, -1): 0}
        for r, c in product(range(R), range(C)):
            if grid[r][c] > 0:
                nodes[(r, c)] = len(nodes)
        n = len(nodes)
        amt = [0] * n
        adj = [[] for _ in range(n)]
        neighborhood = lambda r, c: [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
        in_bounds = lambda r, c: 0 <= r < R and 0 <= c < C
        for r, c in product(range(R), range(C)):
            if not grid[r][c]: continue
            amt[nodes[(r, c)]] = grid[r][c]
            adj[0].append(nodes[(r, c)])
            for nr, nc in neighborhood(r, c):
                if not in_bounds(nr, nc) or not grid[nr][nc]: continue
                adj[nodes[(r, c)]].append(nodes[(nr, nc)])
        ans = 0
        dp = Counter({(1, 0) : 0})
        while dp:
            ndp = Counter()
            for (mask, u), gold in dp.items():
                ans = max(ans, gold)
                for v in adj[u]:
                    if (mask >> v) & 1: continue
                    nmask = mask | (1 << v)
                    ndp[(nmask, v)] = max(ndp[(nmask, v)], gold + amt[v])
            dp = ndp
        return ans
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

##

### Solution 1:

```py

```