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

### Solution 2:  math + combinatorics + inclusion exclusion principle + modular inverse + fermat's little theorem + precompute factorial and inverse factorials

![image](images/number_of_music_playlists.png)

```py
mod = int(1e9) + 7

def mod_inverse(v):
    return pow(v, mod - 2, mod)

def factorials(n):
    fact, inv_fact = [1] * (n + 1), [0] * (n + 1)
    for i in range(2, n + 1):
        fact[i] = (fact[i - 1] * i) % mod
    inv_fact[-1] = mod_inverse(fact[-1])
    for i in reversed(range(n)):
        inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % mod
    return fact, inv_fact

class Solution:
    def numMusicPlaylists(self, n: int, goal: int, k: int) -> int:
        fact, inv_fact = factorials(n)
        f = lambda x: pow(x - k, goal - k, mod) * inv_fact[n - x] * inv_fact[x - k]
        res = 0
        for i in range(k, n + 1):
            res = (res + (1 if (n - i) % 2 == 0 else -1) * f(i)) % mod
        return (res * fact[n]) % mod
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

## 1615. Maximal Network Rank

### Solution 1:  graph theory + degrees + adjacency matrix

if there are multiple with maximum degree than only need to look through those.

```py
class Solution:
    def maximalNetworkRank(self, n: int, roads: List[List[int]]) -> int:
        degrees = [0] * n
        adj_mat = [[0] * n for _ in range(n)]
        for u, v in roads:
            adj_mat[u][v], adj_mat[v][u] = 1, 1
            degrees[u] += 1
            degrees[v] += 1
        max_deg = max(degrees)
        max_ind = [i for i in range(n) if degrees[i] == max_deg]
        if len(max_ind) == 1:
            u = max_ind[0]
            return max(degrees[u] + degrees[v] - adj_mat[u][v] for v in range(n) if v != u)
        return max(degrees[u] + degrees[v] - adj_mat[u][v] for u, v in product(max_ind, repeat = 2) if u != v)
```

## 459. Repeated Substring Pattern

### Solution 1:  modulus + divisors + prefix + time complexity $O(n\sqrt{n})$

There may be $\sqrt{n}$ divisors for n, for each one check if the repeated substring matches the entire string s. 
You can do this with modulus, so just mod by the current divisor m, and that way it will keep wrapping around and matching the prefix with the string.  But need to iterate through entire string which is where the O(n) operations come from. 

```py
class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        n = len(s)
        def matches(m):
            for i in range(n):
                if s[i % m] != s[i]: return False
            return True
        for i in range(1, n // 2 + 1):
            if n % i == 0 and matches(i): return True
        return False
```

### Solution 2:  string is rotation of itself + concatenation of string + boyer's moore algorithm

python uses boyer's moore algorithm to test if pattern is in string.  Which is average time complexity of O(n) so it isn't too bad.  

This uses the fact that the string will be a roration of itself if it contains a repeated substring pattern. 

You have to remove the first and last characters, cause otherwise it will match the entire string which is like a trivial case.  That is string s is trivially a substring of itself. 

![images](images/string_with_repeated_substrings.png)

```py
class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        n = len(s)
        t = s[1:] + s[:-1]
        return s in t
```

### Solution 3:  using Z-algorithm

The pattern you are searching for is string s within the string s + s, with first and last character removed.  So just need to encode it for the z algorithm by putting pattern # string, and it will find if the pattern is a susbstring of the string.

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
    def repeatedSubstringPattern(self, s: str) -> bool:
        n = len(s)
        t = s + "#" + s[1:] + s[:-1]
        z_array = z_algorithm(t)
        return any(z == n for z in z_array)
```

## 723. Candy Crush

### Solution 1:  matrix loop + inplace + lowest zero pointer + bottom to top and treat columns independent for drop phase + use absolute value to detect finding crushable sets in place

```py
class Solution:
    def candyCrush(self, board: List[List[int]]) -> List[List[int]]:
        R, C = len(board), len(board[0])
        def find():
            found = False
            for r, c in product(range(R), range(C)):
                # mark horizontal
                if 0 < c < C - 1 and board[r][c] != 0 and abs(board[r][c - 1]) == abs(board[r][c]) == abs(board[r][c + 1]):
                    for i in range(c - 1, c + 2):
                        board[r][i] = -abs(board[r][i])
                    found = True
                # mark vertical
                if 0 < r < R - 1 and board[r][c] != 0 and abs(board[r - 1][c]) == abs(board[r][c]) == abs(board[r + 1][c]):
                    for i in range(r - 1, r + 2):
                        board[i][c] = -abs(board[i][c])
                    found = True
            for r, c in product(range(R), range(C)):
                if board[r][c] < 0: board[r][c] = 0 # mark as empty
            return found
        def drop():
            for c in range(C):
                lowest_zero = -1
                for r in reversed(range(R)):
                    if board[r][c] == 0:
                        lowest_zero = max(lowest_zero, r)
                    elif lowest_zero >= 0 and board[r][c] > 0:
                        board[lowest_zero][c], board[r][c] = board[r][c], board[lowest_zero][c]
                        lowest_zero -= 1
        while find():
            drop()
        return board
```

## 168. Excel Sheet Column Title

### Solution 1:  base 26 + subtract by one because all characters are shifted by one

That is A is 1, Z is 26, but I want them to be 0 and 25, so shift to the left by 1.  To make the coefficients set so that 0 represents A and 25 represent Z you need to add one to each of the coefficients, and since we are converting to base 26, we need to subtract the 1.  Cause it had the added 1 in the base 10 representation.  

$V = (c_{n} + 1) * b^{n} + ... + (c_{2} + 1) * b^{2} + (c_{1} + 1) * b^{1} + (c_{0} + 1) * b^{0}$
Do this while V > 0
1. $V = V - 1$
1. $c_{i} = V \% b$
1. $V = \lfloor \frac{V}{b} \rfloor$

```py
class Solution:
    def convertToTitle(self, n: int) -> str:
        res = []
        while n > 0:
            n -= 1
            ch = chr(n % 26 + ord("A"))
            res.append(ch)
            n //= 26
        return "".join(reversed(res))
```

## 68. Text Justification

### Solution 1:  string + greedy + left justified

```py
class Solution:
    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        res = []
        cur = words[0]
        num = 0
        for word in words[1:]:
            if len(word) + 1 <= maxWidth - len(cur):
                cur += " "
                cur += word
                num += 1
            else:
                if not num:
                    res.append(cur.ljust(maxWidth, " "))
                else:
                    num_spaces = maxWidth - len(cur) + num
                    each = num_spaces // num
                    extra = num_spaces % num
                    s = ""
                    for w in cur.split():
                        s += w
                        s += " " * (each + int(extra > 0))
                        extra -= 1
                    res.append(s.rstrip(" "))
                cur = word
                num = 0
        res.append(cur.ljust(maxWidth, " "))
        return res
```

## 725. Split Linked List in Parts

### Solution 1:  linked list

```py
class Solution:
    def splitListToParts(self, head: Optional[ListNode], k: int) -> List[Optional[ListNode]]:
        def size(head):
            n = 0
            while head:
                head = head.next
                n += 1
            return n
        sz = size(head)
        res = []
        while k > 0:
            cnt = math.ceil(sz / k)
            sub_head = head
            for _ in range(cnt - 1):
                head = head.next
            if head:
                head.next, head = None, head.next
            res.append(sub_head)
            sz -= cnt
            k -= 1
        return res
```

## 92. Reverse Linked List II

### Solution 1:  multiple pointers for linked list + reverse linked list

```py
class Solution:
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        dummy = ListNode(next = head)
        s = dummy
        for _ in range(left - 1):
            s = s.next
        left_node = s.next
        right_node = None
        s.next = None
        e = left_node
        for _ in range(right - left + 1):
            nxt = e.next
            e.next = right_node
            right_node = e
            e = nxt
        s.next, left_node.next = right_node, e
        return dummy.next
```

## 1282. Group the People Given the Group Size They Belong To

### Solution 1:  offline query + sort

```py
class Solution:
    def groupThePeople(self, groupSizes: List[int]) -> List[List[int]]:
        groups = sorted([(g, i) for i, g in enumerate(groupSizes)])
        res = [[]]
        for g, i in groups:
            res[-1].append(i)
            if len(res[-1]) == g:
                res.append([])
        return res[:-1]
```

## 358. Rearrange String k Distance Apart

### Solution 1:  max heap + queue to store up to k characters + greedy

Best to use the characters with highest frequency, once used keep a queue of characters that are currently blocked. And once the length of queue is >= k that means the character can be reused, add back into max heap.  

```py
class Solution:
    def rearrangeString(self, s: str, k: int) -> str:
        queue = deque()
        heapify(max_heap := [(-s.count(ch), ch) for ch in string.ascii_lowercase])
        res = []
        while max_heap:
            cnt, ch = heappop(max_heap)
            cnt = abs(cnt)
            if cnt == 0: continue
            cnt -= 1
            res.append(ch)
            queue.append((cnt, ch))
            if len(queue) >= k:
                cnt, ch = queue.popleft()
                heappush(max_heap, (-cnt, ch))
        return "".join(res) if len(res) == len(s) else ""
```

## 332. Reconstruct Itinerary

### Solution 1:  dfs + stack + Hierholzer's algorithm + Eulerian path + directed graph + greedy + sort

```py
class Solution:
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        tickets.sort(key = lambda x: x[-1], reverse = True)
        adj_list = defaultdict(list)
        for u, v in tickets:
            adj_list[u].append(v)
        stack = []
        def dfs(u):
            while adj_list[u]:
                dfs(adj_list[u].pop())
            stack.append(u)
        dfs("JFK")
        return stack[::-1]
```

## 1359. Count All Valid Pickup and Delivery Options

### Solution 1:  dynamic programming + space optimized + math

fixing P_i and then letting D_i have it's possible locations

```py
class Solution:
    def countOrders(self, n: int) -> int:
        mod = int(1e9) + 7
        f = lambda x: x * (2 * x - 1)
        dp = 1
        for i in range(1, n + 1):
            dp = (dp * f(i)) % mod
        return dp
```

### Solution 2:  permutations + combinatorics + math

```py
class Solution:
    def countOrders(self, n: int) -> int:
        mod = int(1e9) + 7
        res = fact = 1
        for i in range(2, n + 1):
            res = (res * (2 * i - 1)) % mod
            fact = (fact * i)
        return (res * fact) % mod
```

### Solution 3:  probability + math

probability = favorable_outcomes / total_outcomes

total_outcomes = (2n)! => total number of permutations
probability for correct order for P_i and D_i is 1/2
probability = (1/2)^n, probability that all are in correct order
favorable_outcomes = probability * total_outcomes

```py
class Solution:
    def countOrders(self, n: int) -> int:
        mod = int(1e9) + 7
        res = 1
        for i in range(2, 2 * n + 1):
            res *= i
            if i % 2 == 0: res //= 2
            res %= mod
        return res
```

## 847. Shortest Path Visiting All Nodes

### Solution: BFS + bitmask + dynamic programming

The reason to use BFS is because we have an unweighted undirected graph, so I can imagine all weights are equal to 1.  So BFS will explore it optimally to return the shortest path.  I do need to store the (node, mask) , else it will continue visiting the same node with the same set of nodes visited in the path.  This is obviously already been computed.  So save the shortest path to reach some node aftering visitin g a set of nodes.  

TC: O(N*2^N)

```c++
int shortestPathLength(vector<vector<int>>& graph) {
    int n = graph.size();
    queue<vector<int>> q;
    int endMask = (1<<n)-1;
    vector<vector<bool>> vis(n, vector<bool>(1<<n, false));
    for (int i = 0;i<n;i++) {
        vis[i][(1<<i)]=true;
        q.push({i,(1<<i),0});
    }
    while (!q.empty()) {
        auto v = q.front();
        int i = v[0], mask = v[1], path = v[2];
        q.pop();
        if (mask==endMask) return path;
        for (int nei : graph[i]) {
            int nmask = mask|(1<<nei);
            if (vis[nei][nmask]) continue;
            vis[nei][nmask] = true;
            q.push({nei, nmask, path+1});
        }
    }
    return -1;
}
```

```py
class Solution:
    def shortestPathLength(self, graph: List[List[int]]) -> int:
        n = len(graph)
        dq = deque()
        end_mask = (1 << n) - 1
        steps = 0
        vis = [[0] * (1 << n) for _ in range(n)]
        for i in range(n):
            vis[i][1 << i] = 1
            dq.append((i, 1 << i))
        while dq:
            for _ in range(len(dq)):
                u, mask = dq.popleft()
                for v in graph[u]:
                    nmask = mask | (1 << v)
                    if nmask == end_mask: return steps + 1
                    if vis[v][nmask]: continue
                    vis[v][nmask] = 1
                    dq.append((v, nmask))
            steps += 1
        return 0
```

## 1063. Number of Valid Subarrays

### Solution 1:  monotonically increasing stack

```py
class Solution:
    def validSubarrays(self, nums: List[int]) -> int:
        stack = []
        res = 0
        for num in nums:
            while stack and stack[-1] > num:
                stack.pop()
            stack.append(num)
            res += len(stack)
        return res
```

### Solution 2:  RMQ + sparse tables + binary search

```py
class Solution:
    def validSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        lg = [0] * (n + 1)
        for i in range(2, n + 1):
            lg[i] = lg[i // 2] + 1
        LOG = 16
        st = [[math.inf] * n for _ in range(LOG)]
        st[0] = nums[:]
        for i in range(1, LOG):
            j = 0
            while (j + (1 << (i - 1))) < n:
                st[i][j] = min(st[i - 1][j], st[i - 1][j + (1 << (i - 1))])
                j += 1
        def query(left, right):
            length = right - left + 1
            i = lg[length]
            return min(st[i][left], st[i][right - (1 << i) + 1])
        res = 0
        for i in range(n):
            left, right = i, n - 1
            while left < right:
                mid = (left + right + 1) >> 1
                if query(left, mid) >= nums[i]:
                    left = mid
                else:
                    right = mid - 1
            res += left - i + 1
        return res
```

## 389. Find the Difference

### Solution 1: hashmap with array of size 26

```py
class Solution:
    def findTheDifference(self, s: str, t: str) -> str:
        cnt = [0]*26
        for ch in t:
            cnt[ord(ch)-ord('a')]+=1
        for ch in s:
            cnt[ord(ch)-ord('a')]-=1
        for i, ct in enumerate(cnt):
            if ct:
                return chr(i+ord('a'))
        return ""
```

### Solution 2: Hashmap with dictionary in python, not best space optimization

```py
class Solution:
    def findTheDifference(self, s: str, t: str) -> str:
        return list(Counter(t) - Counter(s)).pop()
```

### Solution 3: mapreduce algorithm xor with last character

```py
class Solution:
    def findTheDifference(self, s: str, t: str) -> str:
        return chr(reduce(xor, map(ord, s+t)))
```

## 880. Decoded String at Index

### Solution 1:

```py
class Solution:
    def decodeAtIndex(self, s: str, k: int) -> str:
        s += "1"
        k -= 1
        arr = re.split(r'(\d{1})', s)
        arr.pop()
        stack = []
        p = 0
        for i in range(1, len(arr), 2):
            chars = arr[i - 1]
            cnt = int(arr[i])
            p = cnt * (len(chars) + p)
            stack.append((chars, cnt))
        while stack:
            chars, cnt = stack.pop()
            p = p // cnt - len(chars)
            if p <= k:
                k %= p + len(chars)
                i = k - p
                if i >= 0:
                    return chars[i]
        return ""
```

```py
class Solution:
    def decodeAtIndex(self, s: str, k: int) -> str:
        sz = 0
        for ch in s:
            if ch.isdigit():
                sz *= int(ch)
            else:
                sz += 1
        for ch in reversed(s):
            k %= sz
            if k == 0 and ch.isalpha(): return ch
            if ch.isdigit():
                sz //= int(ch)
            else:
                sz -= 1
```


```cpp
#define ll long long
class Solution {
public:
    string decodeAtIndex(string s, ll k) {
        ll sz = 0;
        for (auto ch : s) {
            if (isdigit(ch)) sz *= (ch - '0');
            else sz++;
        }
        reverse(s.begin(), s.end());
        for (auto ch: s) {
            k %= sz;
            if (k == 0 and !isdigit(ch)) {
                string s(1, ch);
                return s;
            }
            if (isdigit(ch)) sz /= (ch - '0');
            else sz--;
        }
        return "";
    }
};
```

## 557. Reverse Words in a String III

```py
class Solution:
    def reverseWords(self, s: str) -> str:
        res = " ".join(s.split()[::-1])[::-1]
        return res
```

## 1804. Implement Trie II (Prefix Tree)

```py
class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.prefix_count = self.word_count = 0

class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            node = node.children[ch]
            node.prefix_count += 1
        node.word_count += 1

    def countWordsEqualTo(self, word: str) -> int:
        node = self.root
        for ch in word:
            node = node.children[ch]
        return node.word_count

    def countWordsStartingWith(self, prefix: str) -> int:
        node = self.root
        for ch in prefix:
            node = node.children[ch]
        return node.prefix_count

    def erase(self, word: str) -> None:
        node = self.root
        for ch in word:
            node = node.children[ch]
            node.prefix_count -= 1
        node.word_count -= 1
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