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

## 979. Distribute Coins in Binary Tree

### Solution 1:  dfs, tree traversal

```py
class Solution:
    def distributeCoins(self, root: Optional[TreeNode]) -> int:
        ans = 0
        def dfs(u):
            nonlocal ans
            if not u: return 0
            if not u.left and not u.right: return -1 if u.val == 0 else u.val - 1
            lcoins, rcoins = dfs(u.left), dfs(u.right)
            ans = ans + abs(lcoins) + abs(rcoins)
            u.val = u.val + lcoins + rcoins
            return u.val - 1
        dfs(root)
        return ans
```

## 1863. Sum of All Subset XOR Totals

### Solution 1:  dynamic programming, bitmask, subset sum

```py
class Solution:
    def subsetXORSum(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        m = 2 ** 5
        dp = [0] * (m + 1)
        dp[0] = 1
        for num in nums:
            ndp = dp[:]
            for v in range(m + 1):
                if v ^ num > m: continue
                ndp[v ^ num] += dp[v]
            dp = ndp
        return sum(i * v for i, v in enumerate(dp))
```

## 1255. Maximum Score Words Formed by Letters

### Solution 1:  enumerate bitmask, frequency array

```py
class Solution:
    def maxScoreWords(self, words: List[str], letters: List[str], score: List[int]) -> int:
        n, m = len(words), len(letters)
        unicode = lambda ch: ord(ch) - ord("a")
        freq = [0] * 26
        for ch in letters:
            freq[unicode(ch)] += 1
        ans = 0
        for mask in range(1 << n):
            mfreq = [0] * 26
            for i in range(n):
                if (mask >> i) & 1:
                    for ch in words[i]:
                        mfreq[unicode(ch)] += 1
            if any(f1 > f2 for f1, f2 in zip(mfreq, freq)): continue
            ans = max(ans, sum(score[i] * mfreq[i] for i in range(26)))
        return ans
```

## 552. Student Attendance Record II

### Solution 1:  dynamic programming, counting

```py
class Solution:
    def checkRecord(self, n: int) -> int:
        MOD = int(1e9) + 7
        dp = [[0] * 3 for _ in range(2)] # dp[A][L]
        dp[0][0] = 1
        for _ in range(n):
            ndp = [[0] * 3 for _ in range(2)]
            for i in range(3):
                # add A
                ndp[1][0] = (ndp[1][0] + dp[0][i]) % MOD
                for j in range(2):
                    # add P
                    ndp[j][0] = (ndp[j][0] + dp[j][i]) % MOD
                    # add L
                    if i > 0: ndp[j][i] = (ndp[j][i] + dp[j][i - 1]) % MOD
            dp = ndp
        return sum(sum(row) % MOD for row in dp) % MOD
```

## 1404. Number of Steps to Reduce a Number in Binary Representation to One

### Solution 1:  loop, addition modulo 2

```py
class Solution:
    def numSteps(self, s: str) -> int:
        carry = ans = 0
        n = len(s)
        for i in reversed(range(1, n)):
            cur = int(s[i]) + carry
            if cur > 0: carry = 1
            ans = ans + 1 + cur % 2
        return ans + carry
```

## 260. Single Number III

### Solution 1:  bit manipulation, bitwise xor sum

You end with xor = x ^ y, where x and y are the integers you are trying to find.  
You know where the xor has a bit equal to 1 that indicates that the x and y disagreed at that bit.
So x may have had that bit set to 1 and y had it set to 0.  So then split all the numbers based on if they have that bit
This will lead to getting x and also y.

```py
import operator
class Solution:
    def singleNumber(self, nums: List[int]) -> List[int]:
        xsum = reduce(operator.xor, nums, 0)
        ans = [0] * 2
        i = 0
        while i < 32:
            if (xsum >> i) & 1: break
            i += 1
        for num in nums:
            if (num >> i) & 1:
                ans[0] ^= num
            else:
                ans[1] ^= num
        return ans
```

## 846. Hand of Straights

### Solution 1:  compression, frequency array, two pointers 

```cpp
class Solution {
public:
    bool isNStraightHand(vector<int>& hand, int groupSize) {
        sort(hand.begin(), hand.end());
        vector<int> freq;
        freq.push_back(1);
        for (int i = 1; i < hand.size(); i++) {
            if (hand[i] == hand[i - 1]) {
                freq.end()[-1]++;
            } else {
                freq.push_back(1);
            }
        }
        hand.erase(unique(hand.begin(), hand.end()), hand.end());
        int N = hand.size();
        int i = 0;
        while (i < N) {
            freq[i]--;
            for (int j = 1; j < groupSize; j++) {
                if (i + j == N) return false;
                if (hand[i + j] != hand[i + j - 1] + 1 || freq[i + j] == 0) return false;
                freq[i + j]--;
            }
            while (i < N && freq[i] == 0) i++;
        }
        return true;
    }
};
```

## 523. Continuous Subarray Sum

### Solution 1:  hash table, modular arithmetic, prefix sum under modulo

```cpp
class Solution {
public:
    bool checkSubarraySum(vector<int>& nums, int k) {
        set<int> vis;
        long long psum = 0, ppsum = 0;
        for (long long num : nums) {
            psum = (psum + num) % k;
            if (vis.find(psum) != vis.end()) return true;
            vis.insert(ppsum);
            ppsum = psum;
        }
        return false;
    }
};
```

## 1122. Relative Sort Array

### Solution 1:  rank, custom sorting

```cpp
class Solution {
public:
    vector<int> relativeSortArray(vector<int>& arr1, vector<int>& arr2) {
        const int N = 1e3 + 5;
        vector<int> rank(N, N);
        int n1 = arr1.size(), n2 = arr2.size();
        for (int i = 0; i < n2; i++) {
            rank[arr2[i]] = i;
        }
        sort(arr1.begin(), arr1.end(), [&](const int& a, const int& b) {
            if (rank[a] != rank[b]) return rank[a] < rank[b];
            return a < b;
        });
        return arr1;
    }
};
```

## 945. Minimum Increment to Make Array Unique

### Solution 1:  sort, two pointers

```cpp
class Solution {
public:
    int minIncrementForUnique(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        int freq = 0, ans = 0, n = nums.size(), j = 0;
        const int N = 2e5;
        for (int i = 0; i <= N; i++) {
            ans += freq;
            for (; j < n && nums[j] == i; j++) freq++;
            if (freq > 0) freq--;
        }
        return ans;
    }
};
```

## 330. Patching Array

### Solution 1:  two pointers, reach

```cpp
class Solution {
public:
    int minPatches(vector<int>& nums, int n) {
        int ans = 0;
        long long reach = 0;
        for (int i = 0; reach < n;) {
            if (i < nums.size() && nums[i] <= reach + 1) {
                reach += nums[i++];
            } else {
                reach += reach + 1;
                ans++;
            }
        }
        return ans;
    }
};
```

## 633. Sum of Square Numbers

### Solution 1:  hash table

```cpp
class Solution {
public:
    bool judgeSquareSum(int c) {
        set<int> vis;
        for (long long i = 0; i * i <= c; i++) {
            int b2 = c - i * i;
            if (i * i + i * i == c) return true;
            if (vis.find(b2) != vis.end()) return true;
            vis.insert(i * i);
        }
        return false;
    }
};
```

## 826. Most Profit Assigning Work

### Solution 1:  sorting, prefix max, two pointers

```cpp
class Solution {
public:
    int maxProfitAssignment(vector<int>& difficulty, vector<int>& profit, vector<int>& worker) {
        int pmax = 0, n = profit.size(), ans = 0;
        sort(worker.begin(), worker.end());
        vector<pair<int, int>> queries;
        queries.reserve(n);
        for (int i = 0; i < n; i++) {
            queries.emplace_back(difficulty[i], profit[i]);
        }
        sort(queries.begin(), queries.end());
        int j = 0;
        for (int w : worker) {
            while (j < n && queries[j].first <= w) {
                auto [d, p] = queries[j];
                j++;
                pmax = max(pmax, p);
            }
            ans += pmax;
        }
        return ans;
    }
};
```

## 1482. Minimum Number of Days to Make m Bouquets

### Solution 1:  greedy, binary search

```cpp
class Solution {
public:
    int minDays(vector<int>& bloomDay, int m, int k) {
        function<bool(int)> possible = [&](int target) {
            int cur = 0, cnt = 0;
            for (int day : bloomDay) {
                if (day <= target) {
                    cur++;
                } else {
                    cur = 0;
                }
                if (cur == k) {
                    cnt++;
                    cur = 0;
                }
            }
            return cnt < m;
        };
        const int INF = 1e9 + 5;
        int lo = -1, hi = INF;
        while (lo < hi) {
            int mi = lo + (hi - lo) / 2;
            if (possible(mi)) {
                lo = mi + 1;
            } else {
                hi = mi;
            }
        }
        return lo < INF ? lo : -1;
    }
};
```

## 1052. Grumpy Bookstore Owner

### Solution 1:  fixed size sliding window

```cpp
class Solution {
public:
    int maxSatisfied(vector<int>& customers, vector<int>& grumpy, int minutes) {
        int n = customers.size();
        int sum = 0, wsum = 0, ans = 0;
        for (int i = 0; i < n; i++) {
            if (!grumpy[i]) sum += customers[i];
            else wsum += customers[i];
            if (i >= minutes) {
                if (grumpy[i - minutes]) wsum -= customers[i - minutes];
            }
            ans = max(wsum, ans);
        }
        return sum + ans;
    }
};
```

## 1248. Count Number of Nice Subarrays

### Solution 1:  preprocessed trick, calculating the prefix sum of count of even integers before every odd integer

```cpp
class Solution {
public:
    int numberOfSubarrays(vector<int>& nums, int k) {
        vector<int> preprocessed;
        int cnt = 0;
        for (int num : nums) {
            if (num % 2 == 0) cnt++;
            else {
                preprocessed.push_back(cnt);
                cnt = 0;
            }
        }
        preprocessed.push_back(cnt);
        int n = preprocessed.size(), ans = 0;
        for (int i = 0; i < n - k; i++) {
            ans += (preprocessed[i] + 1) * (preprocessed[i + k] + 1);

        }
        return ans;
    }
};
```

### Solution 2:  Calculate number of subarrays with at most k odd integers, Take the difference to find how many subarrays with exactly k odd integers

```cpp
class Solution {
public:
    int numberOfSubarrays(vector<int>& nums, int k) {
        function<int(int)> atMost = [&nums](int k) {
            int cnt = 0, res = 0;
            for (int l = 0, r = 0; r < nums.size(); r++) {
                cnt += nums[r] % 2;
                while (cnt > k) {
                    cnt -= nums[l] % 2;
                    l++;
                }
                res += r - l + 1;
            }
            return res;
        };
        return atMost(k) - atMost(k - 1);
    }
};
```

## 1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit

### Solution 1:  monotonic deques, min deque, max deque, two pointers

```cpp
class Solution {
public:
    int longestSubarray(vector<int>& nums, int limit) {
        int n = nums.size(), ans = 0, j = 0;
        deque<int> minq, maxq;
        for (int i = 0; i < n; i++) {
            while (!minq.empty() && nums[i] < nums[minq.back()]) minq.pop_back();
            while (!maxq.empty() && nums[i] > nums[maxq.back()]) maxq.pop_back();
            minq.push_back(i); maxq.push_back(i);
            while (nums[maxq.front()] - nums[minq.front()] > limit) {
                if (minq.front() < maxq.front()) {
                    j = minq.front() + 1;
                    minq.pop_front();
                }
                else {
                    j = maxq.front() + 1;
                    maxq.pop_front();
                }
            }
            ans = max(ans, i - j + 1);
        }
        return ans;
    }
};
```

## 995. Minimum Number of K Consecutive Bit Flips

### Solution 1:  deque, prefix count, start backwards, with target, greedily flip when it is required

```cpp
class Solution {
public:
    int minKBitFlips(vector<int>& nums, int k) {
        deque<int> dq;
        int parity = 0, N = nums.size(), ans = 0;
        for (int i = 0; i < N; i++) {
            if (parity == nums[i]) {
                parity ^= 1;
                ans++;
                dq.push_back(i + k - 1);
            }
            if (!dq.empty() && i == dq.front()) {
                dq.pop_front();
                parity ^= 1;
            }
        }
        return dq.empty() ? ans : -1;
    }   
};
```

## 1038. Binary Search Tree to Greater Sum Tree

### Solution 1:  recursion, inorder tree traversal, binary search tree

```cpp
class Solution {
public:
    int sum = 0;
    TreeNode* bstToGst(TreeNode* root) {
        if (root == nullptr) return nullptr;
        bstToGst(root -> right);
        sum += root -> val;
        root -> val = sum;
        bstToGst(root -> left);
        return root;
    }
};
```

## 1791. Find Center of Star Graph

### Solution 1:  Node that appears in any two pair of edges is center

```cpp
class Solution {
public:
    int findCenter(vector<vector<int>>& edges) {
        int u1 = edges[0][0], v1 = edges[0][1], u2 = edges[1][0], v2 = edges[1][1];
        if (u1 == u2 || u1 == v2) return u1;
        return v1;
    }
};
```

##

### Solution 1:

```cpp

```

##

### Solution 1:

```cpp

```

##

### Solution 1:

```cpp

```

##

### Solution 1:

```cpp

```

##

### Solution 1:

```cpp

```
