## 490. The Maze

### Solution 1:  bfs + memoization

```py
class Solution:
    def hasPath(self, maze: List[List[int]], start: List[int], destination: List[int]) -> bool:
        empty, wall, visited = 0, 1, 2
        R, C = len(maze), len(maze[0])
        queue = deque([tuple(start)])
        maze[start[0]][start[1]] = visited
        is_ball_moving = lambda r, c: 0<=r<R and 0<=c<C and maze[r][c] != wall
        while queue:
            r, c = queue.popleft()
            if [r,c] == destination: return True
            for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                nr, nc = r+dr, c+dc
                while is_ball_moving(nr,nc):
                    nr += dr
                    nc += dc
                nr -= dr
                nc -= dc
                if maze[nr][nc] != visited:
                    queue.append((nr,nc))
                    maze[nr][nc] = visited
        return False
```

## 505. The Maze II

### Solution 1:  minheap + dijkstra + generator

```py
class Solution:
    def shortestDistance(self, maze: List[List[int]], start: List[int], destination: List[int]) -> int:
        empty, wall = 0, 1
        R, C = len(maze), len(maze[0])
        minheap = [(0, *start)]
        min_dist = [[inf]*C for _ in range(R)]
        min_dist[start[0]][start[1]] = 0
        is_ball_moving = lambda r, c: 0<=r<R and 0<=c<C and maze[r][c] != wall
        def neighbors(r, c):
            for dr, dc in [(1,0),(0,1),(-1,0),(0,-1)]:
                nr, nc, gap = r + dr, c + dc, 1
                while is_ball_moving(nr,nc):
                    nr, nc, gap = nr + dr, nc + dc, gap + 1
                yield (nr-dr,nc-dc,gap-1)
        while minheap:
            dist, r, c = heappop(minheap)
            if [r,c] == destination: return dist
            for nr, nc, gap in neighbors(r,c):
                ndist = dist + gap
                if ndist < min_dist[nr][nc]:
                    min_dist[nr][nc] = ndist
                    heappush(minheap, (ndist, nr, nc))
        return -1
```

## 499. The Maze III

### Solution 1: minheap variation + dijkstra algorithm + shortest path + seen set + continual movement

```py
class Solution:
    def findShortestWay(self, maze: List[List[int]], ball: List[int], hole: List[int]) -> str:
        empty, wall = 0, 1
        R, C = len(maze), len(maze[0])
        is_ball_moving = lambda r,c: 0<=r<R and 0<=c<C and maze[r][c] != wall
        def neighbors(r,c):
            for di, (dr, dc) in zip('durl', [(1,0),(-1,0),(0,1),(0,-1)]):
                nr, nc, gap = r+dr,c+dc,1
                while is_ball_moving(nr,nc) and [nr,nc] != hole:
                    nr,nc,gap = nr+dr,nc+dc,gap+1
                if [nr,nc]!=hole:
                    nr, nc, gap = nr-dr,nc-dc, gap-1
                yield (nr,nc,gap, di)
        minheap = [(0, '', *ball)]
        seen = set()
        while minheap:
            dist, path, r, c = heappop(minheap)
            if [r,c]==hole: 
                return path
            if (r,c) in seen: continue
            seen.add((r,c))
            for nr,nc,gap,di in neighbors(r,c):
                ndist = dist + gap
                heappush(minheap,(ndist,path+di,nr,nc))
        return 'impossible'
```

## 1368. Minimum Cost to Make at Least One Valid Path in a Grid

### Solution 1:  minheap + dijkstra algo

```py
class Solution:
    def minCost(self, grid: List[List[int]]) -> int:
        right, left, down, up = 1, 2, 3, 4
        R, C = len(grid), len(grid[0])
        minheap = [(0,0,0)]
        min_dist = [[inf]*C for _ in range(R)]
        min_dist[0][0] = 0
        in_bounds = lambda r,c: 0<=r<R and 0<=c<C
        while minheap:
            dist, r, c = heappop(minheap)
            if [r,c] == [R-1,C-1]: return dist
            for di, (nr,nc) in zip([1,2,3,4],[(r,c+1),(r,c-1),(r+1,c),(r-1,c)]):
                if not in_bounds(nr,nc): continue
                ndist = dist + int(grid[r][c]!=di)
                if ndist < min_dist[nr][nc]:
                    min_dist[nr][nc] = ndist
                    heappush(minheap, (ndist,nr,nc))
        return -1
```

## 1928. Minimum Cost to Reach Destination in Time

### Solution 1: minheap + dijkstra algo + memoization for time and node

```py
class Solution:
    def minCost(self, maxTime: int, edges: List[List[int]], passingFees: List[int]) -> int:
        n = len(passingFees)
        min_cost = [[inf]*(maxTime+1) for _ in range(n)] #(node,time)
        min_cost[0][0] = passingFees[0]
        # CONSTRUCT THE GRAPH
        graph = [[] for _ in range(n)]
        for u, v, t in edges:
            graph[u].append((v,t))
            graph[v].append((u,t))
        minheap = [(passingFees[0], 0, 0)]
        while minheap:
            cost, time, node = heappop(minheap)
            if node == n-1: return cost
            for nei, wei in graph[node]:
                ncost = cost + passingFees[nei]
                ntime = time + wei
                if ntime > maxTime: continue
                if ncost < min_cost[nei][ntime]:
                    min_cost[nei][ntime] = ncost
                    heappush(minheap,(ncost,ntime,nei))
        return -1
```

## 1514. Path with Maximum Probability

### Solution 1:

```py

```

## 126. Word Ladder II

### Solution 1:

```py
class Solution:
    def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        wildcard = '*'
        word_len = len(beginWord)
        wordList = set([beginWord, *wordList])
        if endWord not in wordList: return []
        get_pattern = lambda word: word[:i]+wildcard+word[i+1:]
        # CONSTRUCT THE ADJACENCY LIST THAT IS FOR A PATTERN WHAT WOULD BE THE NEXT NODES FROM IT
        adjList = defaultdict(list)
        for word in wordList:
            for i in range(word_len):
                pattern = get_pattern(word)
                adjList[pattern].append(word)
        wordList -= set([beginWord])
        graph = defaultdict(list)
        # BFS AND BUILD A DIRECTED GRAPH THAT IS BACKWARDS, THAT IS CREATING BACKEDGES FROM NODE TO PARENT NODE
        queue = deque([beginWord])
        while queue:
            sz = len(queue)
            cur_level_visited = set()
            for _ in range(sz):
                node = queue.popleft()
                for i in range(word_len):
                    pattern = get_pattern(node)                   
                    for nnode in adjList[pattern]:
                        if nnode not in wordList: continue
                        if nnode not in cur_level_visited:
                            cur_level_visited.add(nnode)
                            queue.append(nnode)
                        graph[nnode].append(node)
            wordList -= cur_level_visited
        # PERFORM BFS ON THE BACKWARDS GRAPH
        layer = [[endWord]]
        found_begin = False
        while layer:
            nlayer = []
            for node_list in layer:
                node = node_list[0]
                for nei in graph[node]:
                    nlayer.append([nei] + node_list)
                    found_begin |= (nei == beginWord)
            layer = nlayer
            if found_begin: break
        return layer
```

## 1570. Dot Product of Two Sparse Vectors

### Solution 1:  two pointers + reduce space by creating list of tuples

```py
class SparseVector:
    def __init__(self, nums: List[int]):
        self.vec = [(i, num) for i, num in enumerate(nums) if num != 0]
        
    # Return the dotProduct of two sparse vectors
    def dotProduct(self, vec: 'SparseVector') -> int:
        v1, v2 = self.vec, vec.vec
        n, m = len(v1), len(v2)
        if n > m:
            v1, v2 = v2, v1
            n, m = m, n
        j = 0
        result = 0
        for i, vi in v1:
            while j < m and v2[j][0] < i:
                j += 1
            if j == m: break
            if i == v2[j][0]:
                result += vi*v2[j][1]
        return result
```

## 387. First Unique Character in a String

### Solution 1:  counter + hash table

```py
class Solution:
    def firstUniqChar(self, s: str) -> int:
        cnts = Counter(s)
        for i, ch in enumerate(s):
            if cnts[ch] == 1: return i
        return -1
```

```py
class Solution:
    def firstUniqChar(self, s: str) -> int:
        return min((s.index(ch) for ch in string.ascii_lowercase if s.count(ch) == 1), default = -1)
```

## 871. Minimum Number of Refueling Stops

### Solution 1:  dynamic programming 

```py
class Solution:
    def minRefuelStops(self, target: int, startFuel: int, stations: List[List[int]]) -> int:
        n = len(stations)
        maxPos = [startFuel] + [0]*n
        for i, (pos, fuel) in enumerate(stations, start=1):
            for j in reversed(range(i)):
                if pos <= maxPos[j]:
                    maxPos[j+1] = max(maxPos[j+1], maxPos[j]+fuel)
        for i, dist in enumerate(maxPos):
            if dist >= target: return i
        return -1
```

## 659. Split Array into Consecutive Subsequences

### Solution 1: deque + doubly linked list

```py
class Solution:
    def isPossible(self, nums: List[int]) -> bool:
        queue = deque()
        for num in nums:
            while queue and queue[0][0] < num - 1: 
                _, cnt = queue.popleft()
                if cnt < 3: return False 
            if queue and queue[0][0]+1 == num:
                _, cnt = queue.popleft()
                queue.append((num, cnt+1))
            else:
                queue.appendleft((num, 1))
        return not any(cnt < 3 for _, cnt in queue)
```

## 1338. Reduce Array Size to The Half

### Solution 1:  custom sort comparator sort + greedy

```py
class Solution:
    def minSetSize(self, arr: List[int]) -> int:
        cnts = Counter(arr)
        n = len(arr)
        removed_cnt = 0
        for i, k in enumerate(sorted(cnts,key=lambda x: cnts[x], reverse=True), start=1):
            removed_cnt += cnts[k]
            if removed_cnt >= n//2:
                return i
        return len(cnts)
```

## 804. Unique Morse Code Words

### Solution 1:  string parsing + map

```py
class Solution:
    def uniqueMorseRepresentations(self, words: List[str]) -> int:
        morse_to_alpha = [".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
        return len({''.join(map(lambda ch: morse_to_alpha[ord(ch)-ord('a')], word)) for word in words})

```

## 2373. Largest Local Values in a Matrix

### Solution 1:  find max for neighbors

```py
class Solution:
    def largestLocal(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        result = [[0]*(n-2) for _ in range(n-2)]
        for r,c in product(range(1,n-1),repeat=2):
            result[r-1][c-1] = max(grid[nr][nc] for nr,nc in [(r-1,c),(r+1,c),(r,c+1),(r,c-1),(r-1,c-1),(r-1,c+1),(r+1,c-1),(r+1,c+1),(r,c)])
        return result
```

## 2374. Node With Highest Edge Score

### Solution 1:  array + custom max function

```py
class Solution:
    def edgeScore(self, edges: List[int]) -> int:
        n = len(edges)
        scores = [0]*n
        for out, in_ in enumerate(edges):
            scores[in_] += out
        return max(range(n), key=lambda i: scores[i])
```

## 2375. Construct Smallest Number From DI String

### Solution 1:  backtracking + visited set

```py
class Solution:
    def smallestNumber(self, pattern: str) -> str:
        digits = '123456789'
        self.cur_num = []
        n = len(pattern)
        self.ans = '9'*(n+1)
        used = set()
        def backtrack(i):
            if i == n:
                self.ans = min(self.ans, ''.join(self.cur_num))
                return
            for dig in digits:
                if dig in used: continue
                if pattern[i] == 'I' and dig > self.cur_num[-1]:
                    self.cur_num.append(dig)
                    used.add(dig)
                    backtrack(i+1)
                    self.cur_num.pop()
                    used.remove(dig)
                elif pattern[i] == 'D' and dig < self.cur_num[-1]:
                    self.cur_num.append(dig)
                    used.add(dig)
                    backtrack(i+1)
                    self.cur_num.pop()
                    used.remove(dig)
        for dig in digits:
            self.cur_num.append(dig)
            used.add(dig)
            backtrack(0)
            self.cur_num.pop()
            used.remove(dig)
        return self.ans
```

## 2379. Minimum Recolors to Get K Consecutive Black Blocks

### Solution 1:  sliding window

```py
class Solution:
    def minimumRecolors(self, blocks: str, k: int) -> int:
        n = len(blocks)
        cnt_black = 0
        max_black = 0
        for i in range(n):
            cnt_black += (blocks[i] == 'B')
            if i >= k:
                cnt_black -= (blocks[i-k] == 'B')     
            max_black = max(max_black, cnt_black)
        return k-max_black
```

## 2380. Time Needed to Rearrange a Binary String

### Solution 1:  brute force + find substring in string + string replace

```py
class Solution:
    def secondsToRemoveOccurrences(self, s: str) -> int:
        time = 0
        while '01' in s:
            s = s.replace('01', '10')
            time += 1
        return time
```

### Solution 2:  dynammic programming + prefix sum

```py
class Solution:
    def secondsToRemoveOccurrences(self, s: str) -> int:
        prefix = 0
        cnt_zeros = 0
        for ch in s:
            cnt_zeros += (ch=='0')
            if ch == '1' and cnt_zeros > 0:
                prefix = max(cnt_zeros, prefix+1)
        return prefix
```

## 2381. Shifting Letters II

### Solution 1:  mark start and end of each shift + prefix sum over the shifts + mod + line sweep

```py
class Solution:
    def shiftingLetters(self, s: str, shifts: List[List[int]]) -> str:
        mod = 26
        n = len(s)
        offsets = [0]*(n+1)
        for start, end, dir in shifts:
            if dir == 1:
                offsets[start] += 1
                offsets[end+1] -= 1
            else:
                offsets[start] -= 1
                offsets[end+1] += 1
        result = ['a']*n
        rolling_shift = 0
        for i, ch in enumerate(s):
            rolling_shift += offsets[i]
            val = (ord(ch) - ord('a') + rolling_shift + mod)%mod + ord('a')
            result[i] = chr(val)
        return ''.join(result)
```

## 2382. Maximum Segment Sum After Removals

### Solution 1:  hash table + backwards + merge segments with a hash table

```py
class Solution:
    def maximumSegmentSum(self, nums: List[int], removeQueries: List[int]) -> List[int]:
        n = len(nums)
        Segment = namedtuple('Segment', ['val', 'len'])
        seg_dict = defaultdict(lambda: Segment(0,0))
        arr = []
        maxSegmentVal = 0
        for i in reversed(removeQueries):
            arr.append(maxSegmentVal)
            left_segment, right_segment = seg_dict[i-1], seg_dict[i+1]
            len_ = left_segment.len + right_segment.len + 1
            segment_val = left_segment.val + right_segment.val + nums[i]
            maxSegmentVal = max(maxSegmentVal, segment_val)
            seg_dict[i-left_segment.len] = Segment(segment_val, len_)
            seg_dict[i+right_segment.len] = Segment(segment_val, len_)
        return reversed(arr)
```

## 549. Binary Tree Longest Consecutive Sequence II

### Solution 1:  dfs + dataclasses + binary tree

```py
from dataclasses import make_dataclass
class Solution:
    def longestConsecutive(self, root: Optional[TreeNode]) -> int:
        Delta = make_dataclass('Delta', [('incr', int), ('decr', int)], frozen=True)
        self.ans = 0
        def dfs(node):
            if not node: return Delta(0,0)
            incr = decr = 1
            val = node.val
            if node.left:
                left = dfs(node.left)
                if node.left.val == val - 1:
                    decr += left.decr
                elif node.left.val == val + 1:
                    incr += left.incr
            if node.right:
                right = dfs(node.right)
                if node.right.val == val - 1:
                    decr = max(decr, 1 + right.decr)
                if node.right.val == val + 1:
                    incr = max(incr, 1 + right.incr)
            self.ans = max(self.ans, incr+decr-1)
            return Delta(incr, decr)
        dfs(root)
        return self.ans
```

## 342. Power of Four

### Solution 1:  bit manipulation + hexadecimal + find it is power of two + find it is even bit with hexadecimal bitmask

```py
class Solution:
    def isPowerOfFour(self, n: int) -> bool:
        return n > 0 and n & (n-1) == 0 and 0xaaaaaaaa&n == 0
```

## 2376. Count Special Integers

### Solution 1:  digit dynamic programming + math + permutations + statistics + bitmask

![image](images/count_special_integers.png)

```py
class Solution:
    def countSpecialNumbers(self, n: int) -> int:
        # math.perm(n,r) total objects is n, and can pick r objects
        snum = str(n)
        num_digits = len(snum)
        num_objects = 9
        result = 0
        # PART 1 permutations
        for num_spaces in range(num_digits-1):
            result += math.perm(num_objects,num_spaces)
        result *= 9
        # PART 2
        bitmask = 0
        num_digits -= 1
        is_last_digit = lambda cur_dig, upper_dig, num_digits: cur_dig == upper_dig and num_digits > 0
        is_leading_zero = lambda num_digits, cur_dig: num_digits == len(snum)-1 and cur_dig == 0
        is_marked_digit = lambda bitmask, cur_dig: (bitmask>>cur_dig)&1
        for dig in map(int, snum):
            cur_perm = math.perm(num_objects, num_digits)
            for cur_dig in range(dig+1):
                if is_last_digit(cur_dig, dig, num_digits) or is_leading_zero(num_digits, cur_dig) or is_marked_digit(bitmask, cur_dig): continue 
                result += cur_perm
            if is_marked_digit(bitmask, dig): break
            bitmask |= (1<<dig)
            num_objects -= 1
            num_digits -= 1
        return result 
```

## 383. Ransom Note

### Solution 1:  two hashmaps + all

```py
class Solution:
    def count_arr(self, s: str) -> List[int]:
        cnter = [0]*26
        get_unicode = lambda ch: ord(ch) - ord('a')
        for ch in s:
            cnter[get_unicode(ch)] += 1
        return cnter
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        ranCnt, magCnt = self.count_arr(ransomNote), self.count_arr(magazine)
        return all(cnt1>=cnt2 for cnt1, cnt2 in zip(magCnt, ranCnt))
```

### Solution 2:  one hashmap 

```py
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        magCnt = Counter(magazine)
        for ch in ransomNote:
            magCnt[ch] -= 1
            if magCnt[ch] < 0: return False
        return True
```

### Solution 3:  counter subtraction + set difference

```py
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        return not Counter(ransomNote) - Counter(magazine)
```

## 326. Power of Three

### Solution 1:  numpy convert to base 10 (decimal) to base 3 

```py
import numpy as np
class Solution:
    def isPowerOfThree(self, n: int) -> bool:
        return n > 0 and sum(map(int, np.base_repr(n, base=3))) == 1
```

### Solution 2:  prime integers find the if the largest value is divisible by n

```py
class Solution:
    def isPowerOfThree(self, n: int) -> bool:
        return n > 0 and 3**19 % n == 0
```

## 342. Power of Four

### Solution 1:  bit manipulation + bitmask + find that it is power of 2 by trick + find it is 1 at an odd position in binary

```py
class Solution:
    def isPowerOfFour(self, n: int) -> bool:
        return n > 0 and n & (n-1) == 0 and 0xaaaaaaaa&n == 0
```

## 234. Palindrome Linked List

### Solution 1:  slow and fast pointer get middle + reversed linked list with (prev, cur, next) pointer 

```py
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        # SLOW AND FAST POINTER
        middle = self.get_middle(head)
        right = self.reversed_list(middle.next)
        while right:
            if head.val != right.val: return False
            head = head.next
            right = right.next
        return True
    
    def get_middle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow = fast = head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        return slow

    def reversed_list(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        cur = head
        while cur:
            next_ = cur.next
            cur.next = prev
            prev = cur
            cur = next_
        return prev
```

## 1857. Largest Color Value in a Directed Graph

### Solution 1:  topological sort + store count for each path + deque

```py
class Solution:
    def largestPathValue(self, colors: str, edges: List[List[int]]) -> int:
        n = len(colors)
        # CONSTRUCT GRAPH
        graph = [[] for _ in range(n)]
        indegrees = [0]*n
        for node_out, node_in in edges:
            graph[node_out].append(node_in)
            indegrees[node_in] += 1
        queue = deque()
        counters = [[0]*26 for _ in range(n)]
        get_unicode = lambda ch: ord(ch) - ord('a')
        for i in range(n):
            if indegrees[i] == 0:
                queue.append(i)
        largest_path = processed_nodes = 0
        while queue:
            node = queue.popleft()
            color = get_unicode(colors[node])
            counters[node][color] += 1
            largest_path = max(largest_path, counters[node][color])
            processed_nodes += 1
            for nei in graph[node]:
                indegrees[nei] -= 1
                for i in range(26):
                    counters[nei][i] = max(counters[nei][i], counters[node][i])
                if indegrees[nei] == 0:
                    queue.append(nei)
        return largest_path if processed_nodes == n else -1
```

## 1203. Sort Items by Groups Respecting Dependencies

### Solution 1:  topological sort + topological sort of groups then topological sort of nodes within each group + preprocess data

![images](images/sort_items_by_dependencies.png)

```py
class Solution:
    def sortItems(self, n: int, m: int, group: List[int], beforeItems: List[List[int]]) -> List[int]:
        """
        Creates a topological ordering of nodes
        n = number of nodes
        nodes = iterable of nodes in directed graph
        adj_list = dictionary of lists adjacency list representation of directed graph for the nodes given

        returns: topological ordering of nodes if possible
        """
        def topological_ordering(n, nodes, adj_list):
            indegrees = Counter()
            queue = deque()
            for neis in adj_list.values():
                for nei in neis: indegrees[nei] += 1
            for node in nodes:
                if indegrees[node] == 0: queue.append(node)
            topo_order = []
            while queue:
                node = queue.popleft()
                topo_order.append(node)
                for nei in adj_list[node]:
                    indegrees[nei] -= 1
                    if indegrees[nei] == 0: queue.append(nei)
            return topo_order if len(topo_order) == n else []
        """
        preprocess the data into directed graphs for topological_ordering function

        Regrouping all nodes to belong to a group can make the topological ordering of groups easier.
        sort by group, so same groups are together, than you can use itertools.groupby to split the groups.
        special case is when key = -1, each item belongs to own group in that case to loop through all the nodes and assign while incrementing the m
        Else just assign m
        m will be equal to the number of unique groups
        """
        m = 0
        for key, grp in itertools.groupby(sorted([(group[i], i) for i in range(n)]), key = lambda x: x[0]):
            for _, i in grp:
                group[i] = m
                m += key == -1
            m += key != -1

        """
        Create an adjacency lists for the group and the nodes in each group
        This is needed to perform topo ordering
        1. add edge to directed intergroup graph if u, v belong to different groups
        2. add edge to corresponding directed intragroup graph if u, v belong to the same group
        """
        inter_grp_adj_list = defaultdict(list)
        intra_grp_adj_lists = [defaultdict(list) for _ in range(m)]
        for v, neis in enumerate(beforeItems):
            # u -> v
            for u in neis:
                if group[u] != group[v]:
                    inter_grp_adj_list[group[u]].append(group[v])
                else:
                    intra_grp_adj_lists[group[u]][u].append(v)

        """
        Find the a valid topological ordering of the intergroup directed graph
        This is finding the valid order in which you can process the intragroup graphs for each group.
        This guarantees that you process the dependencies between groups in proper order
        
        For example, if node u, and node v belong to group i and j respectively.  But v depends on u that is u -> v then you
        will have that i -> j, that is group j depends on group i, because a node belonging to group j depends on a node belonging to group i
        """
        inter_group_topo_ordering = topological_ordering(m, range(m), inter_grp_adj_list)
        if not inter_group_topo_ordering: return []

        """
        Looping through the each group in topological ordering, which guarantees that any dependencies between groups is satisfied.

        1. Create a list of all nodes belonging to each group
        2. Find valid topolocail ordering of the nodes within a group
        """
        res = []
        group_nodes = [[] for _ in range(m)]
        for i in range(n):
            group_nodes[group[i]].append(i)
        for group_i in inter_group_topo_ordering:
            topo_order = topological_ordering(len(group_nodes[group_i]), group_nodes[group_i], intra_grp_adj_lists[group_i])
            if not topo_order: return []
            res.extend(topo_order)
        return res
```

## 1591. Strange Printer II

### Solution 1:  topological sort + dataclass + default value + graphlib library + detect cycle in graph

```py
from dataclasses import make_dataclass, field
from graphlib import TopologicalSorter
class Solution:
    def isPrintable(self, targetGrid: List[List[int]]) -> bool:
        Rectangle = make_dataclass('Rectangle', [('maxRow', int, field(default=-inf)), ('minCol', int, field(default=inf)), ('minRow', int, field(default=inf)), ('maxCol', int, field(default=-inf))])
        R, C = len(targetGrid), len(targetGrid[0])
        rect_dict = defaultdict(Rectangle)
        for r, c in product(range(R), range(C)):
            color = targetGrid[r][c]
            rect = rect_dict[color]
            rect.maxRow = max(rect.maxRow, r)
            rect.minRow = min(rect.minRow, r)
            rect.maxCol = max(rect.maxCol, c)
            rect.minCol = min(rect.minCol, c)
        in_bounds = lambda r, c, rect: rect.minRow<=r<=rect.maxRow and rect.minCol<=c<=rect.maxCol
        indegrees = Counter()
        graph = defaultdict(set)
        for r, c in product(range(R), range(C)):
            color_out = targetGrid[r][c]
            for color_in, rect in rect_dict.items():
                if color_out == color_in or not in_bounds(r,c,rect): continue
                graph[color_in].add(color_out)
        ts = TopologicalSorter(graph)
        try:
            ts.prepare()
            return True
        except:
            return False
```

### Solution 2:  topological sort + deque

```py
from dataclasses import make_dataclass, field
class Solution:
    def isPrintable(self, targetGrid: List[List[int]]) -> bool:
        Rectangle = make_dataclass('Rectangle', [('maxRow', int, field(default=-inf)), ('minCol', int, field(default=inf)), ('minRow', int, field(default=inf)), ('maxCol', int, field(default=-inf))])
        R, C = len(targetGrid), len(targetGrid[0])
        rect_dict = defaultdict(Rectangle)
        for r, c in product(range(R), range(C)):
            color = targetGrid[r][c]
            rect = rect_dict[color]
            rect.maxRow = max(rect.maxRow, r)
            rect.minRow = min(rect.minRow, r)
            rect.maxCol = max(rect.maxCol, c)
            rect.minCol = min(rect.minCol, c)
        in_bounds = lambda r, c, rect: rect.minRow<=r<=rect.maxRow and rect.minCol<=c<=rect.maxCol
        indegrees = Counter()
        graph = defaultdict(set)
        for r, c in product(range(R), range(C)):
            color_out = targetGrid[r][c]
            for color_in, rect in rect_dict.items():
                if color_out == color_in or not in_bounds(r,c,rect): continue
                graph[color_in].add(color_out)
        for node_ins in graph.values():
            for node_in in node_ins:
                indegrees[node_in] += 1
        queue = deque()
        for color in rect_dict.keys():
            if indegrees[color] == 0:
                queue.append(color)
        while queue:
            node = queue.popleft()
            for nei in graph[node]:
                indegrees[nei] -= 1
                if indegrees[nei] == 0:
                    queue.append(nei)
        return sum(indegrees.values()) == 0
```

## 869. Reordered Power of 2

### Solution 1:  permutations + backtracking + filter + map + any

```py
class Solution:
    def reorderedPowerOf2(self, n: int) -> bool:
        perm_gen = permutations(str(n))
        perm_filter = filter(lambda p: p[0]!='0', perm_gen)
        perm_map = map(lambda p: int(''.join(p)), perm_filter)
        return any(v&(v-1) == 0 for v in perm_map)
```

### Solution 2:  counter + generate every possible counter for power of 2, and check if it is equal to current number

```py
class Solution:
    def reorderedPowerOf2(self, n: int) -> bool:
        counts = Counter(str(n))
        return any(counts == Counter(str(1<<i)) for i in range(30))
```

## 363. Max Sum of Rectangle No Larger Than K

### Solution 1:  sortedset + transpose + binary search + 2d prefix sum + convert 2d to 1d array

```py
from sortedcontainers import SortedSet
import numpy as np
class Solution:
    def maxSumSubmatrix(self, matrix: List[List[int]], k: int) -> int:
        R, C = len(matrix), len(matrix[0])
        # tranpose so that the shorter dimension is in the nested loop
        if R > C:
            matrix = np.transpose(matrix)
            R, C = C, R
        self.result = -inf
        def colRangeSum(rowSum):
            seen = SortedSet([0])
            for s in accumulate(rowSum):
                idx = seen.bisect_left(s-k)
                if idx < len(seen):
                    self.result = max(self.result, s-seen[idx])
                seen.add(s)
        for r1 in range(R):
            rowSum = [0]*C
            for r2 in range(r1,R):
                for c in range(C):
                    rowSum[c] += matrix[r2][c]
                colRangeSum(rowSum)
                if self.result == k: return self.result
        return self.result
                    
```

## 1567. Maximum Length of Subarray With Positive Product

### Solution 1:  sliding window

```py
class Solution:
    def getMaxLen(self, nums: List[int]) -> int:
        left = maxLen = negativeCount = 0
        leftMostNegative = None
        n = len(nums)
        for right in range(n):
            if nums[right] == 0:
                left = right + 1
                negativeCount = 0
            elif nums[right] < 0:
                if negativeCount == 0:
                    leftMostNegative = right
                negativeCount += 1
            if negativeCount%2 == 0:
                maxLen = max(maxLen, right-left+1)
            else:
                maxLen = max(maxLen, right-leftMostNegative)
        return maxLen 
```

### Solution 2:  dynamic programming

```py

```

## 2389. Longest Subsequence With Limited Sum

### Solution 1: sort + offline queries

```py
class Solution:
    def answerQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        n, m = len(nums), len(queries)
        nums.sort()
        answer = [0]*m
        j = psum = 0
        for q, i in sorted([(q, i) for i, q in enumerate(queries)]):
            while j < n and psum <= q:
                psum += nums[j]
                j += 1
            answer[i] = j
            if psum > q:
                answer[i] -= 1
        return answer
```

### Solution 2: sort + online query + binary search + O(mlogn + nlogn + n) = O(mlogn + nlogn) time

```py
class Solution:
    def answerQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        n, m = len(nums), len(queries)
        psum = [0]*(n+1)
        nums.sort()
        for i in range(n):
            psum[i+1] = psum[i] + nums[i]
        answer = [0]*m
        for i in range(m):
            answer[i] = bisect.bisect_right(psum, queries[i]) - 1
        return answer
```

## 2390. Removing Stars From a String

### Solution 1:  stack 

```py
class Solution:
    def removeStars(self, s: str) -> str:
        star = '*'
        stack = []
        for ch in s:
            if ch == star:
                if stack:
                    stack.pop()
            else:
                stack.append(ch)
        return ''.join(stack)
```

## 2391. Minimum Amount of Time to Collect Garbage

### Solution 1:  greedy + iterate over each garbage material

```py
class Solution:
    def garbageCollection(self, garbage: List[str], travel: List[int]) -> int:
        n = len(travel)
        glass, paper, metal = 'G', 'P', 'M'
        def compute_time(material: str) -> int:
            cnt = sum(g.count(material) for g in garbage)
            time = 0
            for i, garb in enumerate(garbage):
                cur_count = garb.count(material)
                time += cur_count
                cnt -= cur_count
                if cnt == 0: break
                time += travel[i]
            return time
        return compute_time(glass) + compute_time(paper) + compute_time(metal)
```

## 2392. Build a Matrix With Conditions

### Solution 1: topological sort for row and col + hash map + cycle detection + queue + set

```py
class Solution:
    def topoSort(self, k: int, conditions: List[List[int]]) -> List[int]:
        condSet = set([tuple(edge) for edge in conditions])
        topoList = []
        graph = [[] for _ in range(k+1)]
        indegrees = [0]*(k+1)
        for u, v in condSet:
            graph[u].append(v)
            indegrees[v] += 1
        queue = deque()
        for i in range(1,k+1):
            if indegrees[i] == 0:
                queue.append(i)
        while queue:
            node = queue.popleft()
            topoList.append(node)
            for nei_node in graph[node]:
                indegrees[nei_node] -= 1
                if indegrees[nei_node] == 0:
                    queue.append(nei_node)
        return topoList
    def buildMatrix(self, k: int, rowConditions: List[List[int]], colConditions: List[List[int]]) -> List[List[int]]:
        row = self.topoSort(k, rowConditions)
        if len(row) < k: return []
        col = self.topoSort(k, colConditions)
        if len(col) < k: return []
        col_mapper = {v:i for i,v in enumerate(col)}
        matrix = [[0]*k for _ in range(k)]    
        for r, rv in enumerate(row):
            c = col_mapper[rv]
            matrix[r][c] = rv
        return matrix
```

## 1329. Sort the Matrix Diagonally

### Solution 1:  heap + hash map

```py
class Solution:
    def diagonalSort(self, mat: List[List[int]]) -> List[List[int]]:
        R, C = len(mat), len(mat[0])
        diagonals = defaultdict(list)
        for r, c in product(range(R), range(C)):
            diagonals[r-c].append(mat[r][c])
        for diagonal in diagonals.values():
            heapify(diagonal)
        for r, c in product(range(R), range(C)):
            val = heappop(diagonals[r-c])
            mat[r][c] = val
        return mat
```

## 200. Number of Islands

### Solution 1:

```py
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        R, C = len(grid), len(grid[0])
        water, land = '0', '1'
        num_islands = 0
        def dfs(r,c):
            is_land = lambda r, c: 0<=r<R and 0<=c<C and grid[r][c] == land
            for nr, nc in filter(lambda x: is_land(x[0],x[1]), [(r+1,c),(r-1,c),(r,c+1),(r,c-1)]):
                grid[nr][nc] = water
                dfs(nr,nc)
        for r, c in product(range(R), range(C)):
            if grid[r][c] == land:
                dfs(r,c)
                num_islands += 1
        return num_islands
```

## 2398. Maximum Number of Robots Within Budget

### Solution 1:  two sliding window algorithms for running cost and max charge time + monotonic sliding window + deque

```py
class Solution:
    def maximumRobots(self, chargeTimes: List[int], runningCosts: List[int], budget: int) -> int:
        n = len(chargeTimes)
        maxRobots = 0
        cur_cost = 0
        charge_window = deque()
        window = deque()
        for i in range(n):
            charge = chargeTimes[i]
            cost = runningCosts[i]
            cur_cost += cost
            window.append(i)
            while charge_window and charge_window[-1][0] < charge:
                charge_window.pop()
            charge_window.append((charge, i))
            while window and charge_window[0][0] + len(window)*cur_cost > budget:
                j = window.popleft()
                prev_cost = runningCosts[j]
                while charge_window and charge_window[0][1]<=j:
                    charge_window.popleft()
                cur_cost -= prev_cost
            maxRobots = max(maxRobots, len(window))
        return maxRobots
```

## 2397. Maximum Rows Covered by Columns

### Solution 1: bitmask + array to store count for each row to quickly check if it is covered

```py
class Solution:
    def maximumRows(self, mat: List[List[int]], cols: int) -> int:
        maxCover = 0
        R, C = len(mat), len(mat[0])
        count_rows = [0]*R
        for r, c in product(range(R), range(C)):
            count_rows[r] += mat[r][c]
        for i in range(1<<C):
            if i.bit_count() != cols: continue
            cur_rows = [0]*R
            for j in range(C):
                if (i>>j)&1:
                    for r in range(R):
                        cur_rows[r] += mat[r][j]
            maxCover = max(maxCover, sum(cnt1==cnt2 for cnt1,cnt2 in zip(count_rows, cur_rows)))
        return maxCover
```

## 2396. Strictly Palindromic Number

### Solution 1:  loop + convert to base 10 number to arbitrary base + reverse to check it is palindrome

```py
class Solution:
    def isStrictlyPalindromic(self, n: int) -> bool:
        for base in range(2,n-1):
            arr = []
            num = n
            while num > 0:
                digit = int(num%base)
                arr.append(digit)
                num //= base
            if arr[::-1] != arr:
                return False
        return True
```

## 2395. Find Subarrays With Equal Sum

### Solution 1:  loop + zip + dictionary

```py
class Solution:
    def findSubarrays(self, nums: List[int]) -> bool:
        n = len(nums)
        seen = set()
        for x,y in zip(nums,nums[1:]):
            if x+y in seen: return True
            seen.add(x+y)
        return False
                
```

## 967. Numbers With Same Consecutive Differences

### Solution 1:  recursion + backtrack

```py
class Solution:
    def numsSameConsecDiff(self, n: int, k: int) -> List[int]:
        result = []
        def backtrack(i):
            if i == n:
                result.append(int(''.join(map(str,number))))
                return
            for j in range(10):
                if abs(number[-1]-j) == k:
                    number.append(j)
                    backtrack(i+1)
                    number.pop()
        number = []
        for i in range(1,10):
            number.append(i)
            backtrack(1)
            number.pop()
        return result
```

## 924. Minimize Malware Spread

### Solution 1:  color connected components with dfs + intial nodes are only interesting if they are only infected belonging to connected component

![minimize malware spread](images/minimize_malware_spread.png)

```py
class Solution:
    def minMalwareSpread(self, graph: List[List[int]], initial: List[int]) -> int:
        n = len(graph)
        index = sorted(initial)[0]
        initial = set(initial)
        colorSize = Counter()
        infected_color = defaultdict(list)
        visited = [0]*n
        def dfs(i: int) -> int:
            cnt = 1
            if i in initial:
                infected_color[color].append(i)
            for j in range(n):
                if i == j or graph[i][j]==0 or visited[j]: continue
                visited[j] = 1
                cnt += dfs(j)
            return cnt
        color = 0
        for i in range(n):
            if visited[i]: continue
            visited[i] = 1
            colorSize[color] = dfs(i)
            color += 1
        maxSize = 0
        for color in colorSize.keys():
            if len(infected_color[color]) == 1:
                node = infected_color[color][0]
                if colorSize[color] > maxSize:
                    maxSize = colorSize[color]
                    index = node
                elif colorSize[color] == maxSize and node < index:
                    index = node
        return index
```

## 928. Minimize Malware Spread II

### Solution 1:  dfs from each infected node + dictionary to store what normal nodes can be reached from infected nodes

```py
class Solution:
    def minMalwareSpread(self, graph: List[List[int]], initial: List[int]) -> int:
        n = len(graph)
        malware_reached = defaultdict(list)
        initialSet = set(initial)
        def dfs(node: int) -> None:
            for nei_node, is_nei in enumerate(graph[node]):
                if not is_nei or nei_node in visited or nei_node in initialSet: continue
                visited.add(nei_node)
                malware_reached[nei_node].append(infected_node)
                dfs(nei_node)
        for infected_node in initial:
            visited = set([infected_node])
            dfs(infected_node)
        count_singly_infected = Counter()
        for infected_nodes in malware_reached.values():
            if len(infected_nodes) > 1: continue
            count_singly_infected[infected_nodes[0]] += 1
        smallest_index, maxSaved = 0, -1
        for infected in sorted(initial):
            if count_singly_infected[infected] > maxSaved:
                smallest_index = infected
                maxSaved = count_singly_infected[infected]
        return smallest_index
```

## 1697. Checking Existence of Edge Length Limited Paths

### Solution 1:  union find + offline query + sort edge and query lists

```py
class UnionFind:
    def __init__(self, n: int):
        self.size = [1]*n
        self.parent = list(range(n))
    
    def find(self,i: int) -> int:
        while i != self.parent[i]:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    """
    returns true if the nodes were not union prior. 
    """
    def union(self,i: int,j: int) -> bool:
        i, j = self.find(i), self.find(j)
        if i!=j:
            if self.size[i] < self.size[j]:
                i,j=j,i
            self.parent[j] = i
            self.size[i] += self.size[j]
            return True
        return False
    
    @property
    def root_count(self):
        return sum(node == self.find(node) for node in range(len(self.parent)))
    def single_connected_component(self) -> bool:
        return self.size[self.find(0)] == len(self.parent)
    def is_same_connected_components(self, i: int, j: int) -> bool:
        return self.find(i) == self.find(j)
    def num_connected_components(self) -> int:
        return len(set(map(self.find, self.parent)))
    def __repr__(self) -> str:
        return f'parents: {[(i, parent) for i, parent in enumerate(self.parent)]}, sizes: {self.size}'

class Solution:
    def distanceLimitedPathsExist(self, n: int, edgeList: List[List[int]], queries: List[List[int]]) -> List[bool]:
        dsu = UnionFind(n)
        edgeList.sort(key = lambda edge: edge[2])
        m = len(queries)
        queries = [(u, v, lim, i) for i, (u, v, lim) in enumerate(queries)]
        res = [False]*m
        i = 0
        for u, v, lim, idx in sorted(queries, key = lambda q: q[2]):
            while i < len(edgeList) and edgeList[i][2] < lim:
                dsu.union(edgeList[i][0], edgeList[i][1])
                i += 1
            res[idx] = dsu.is_same_connected_components(u, v)
        return res
```

## 1724. Checking Existence of Edge Length Limited Paths II

### Solution 1:

```py
class UnionFind:
    def __init__(self,n: int):
        self.size = [1]*n
        self.parent = list(range(n))
    
    def find(self,i: int) -> int:
        if i==self.parent[i]:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self,i: int,j: int) -> bool:
        i, j = self.find(i), self.find(j)
        if i!=j:
            if self.size[i] < self.size[j]:
                i,j=j,i
            self.parent[j] = i
            self.size[i] += self.size[j]
            return True
        return False
    
class BinaryLift:
    """
    This binary lift function works on any undirected graph that is composed of
    an adjacency list defined by graph
    """
    def __init__(self, node_count: int, graph: List[List[int]]):
        self.size = node_count
        self.graph = graph # pass in an adjacency list to represent the graph
        self.depth = [0]*node_count
        self.parents = [-1]*node_count
        self.parentsWeight = [0]*node_count
        self.visited = [False]*node_count
        # ITERATE THROUGH EACH POSSIBLE TREE
        for node in range(node_count):
            if self.visited[node]: continue
            self.visited[node] = True
            self.get_parent_depth(node)
        self.maxAncestor = 18 # set it so that only up to 2^18th ancestor can exist for this example
        self.jump = [[-1]*self.maxAncestor for _ in range(self.size)]
        self.maxJumpWeight = [[0]*self.maxAncestor for _ in range(self.size)]
        self.build_sparse_table()
        
    def build_sparse_table(self) -> None:
        """
        builds the jump and maxWeightJump sparse arrays for computing the 2^jth ancestor of ith node in any given query
        """
        for j in range(self.maxAncestor):
            for i in range(self.size):
                if j == 0:
                    self.jump[i][j] = self.parents[i]
                    self.maxJumpWeight[i][j] = self.parentsWeight[i]
                elif self.jump[i][j-1] != -1:
                    prev_ancestor = self.jump[i][j-1]
                    self.jump[i][j] = self.jump[prev_ancestor][j-1]
                    current_jump_weight = self.maxJumpWeight[i][j-1]
                    prev_max_weight = self.maxJumpWeight[prev_ancestor][j-1]
                    if prev_max_weight == 0: continue 
                    self.maxJumpWeight[i][j] = max(current_jump_weight, prev_max_weight)
                    
    def get_parent_depth(self, node: int, parent_node: int = -1, weight: int = 0, depth: int = 0) -> None:
        """
        Fills out the depth array for each node and the parent array for each node
        """
        self.parents[node] = parent_node
        self.parentsWeight[node] = weight
        self.depth[node] = depth
        for nei_node, wei in self.graph[node]:
            if self.visited[nei_node]: continue
            self.visited[nei_node] = True
            self.get_parent_depth(nei_node, node, wei, depth+1)
            
    def max_weight_lca(self, p: int, q: int) -> int:
        self.maxWeight = 0
        # ASSUME NODE P IS DEEPER THAN NODE Q   
        if self.depth[p] < self.depth[q]:
            p, q = q, p
        k = self.depth[p] - self.depth[q]
        p = self.kthAncestor(p, k)
        if p == q: return self.maxWeight
        for j in range(self.maxAncestor)[::-1]:
            if self.jump[p][j] != self.jump[q][j]:
                self.maxWeight = max(self.maxWeight, self.maxJumpWeight[p][j], self.maxJumpWeight[q][j])
                p, q = self.jump[p][j], self.jump[q][j] # jump to 2^jth ancestor nodes
        self.maxWeight = max(self.maxWeight, self.maxJumpWeight[p][0], self.maxJumpWeight[q][0])
        return self.maxWeight
    
    def kthAncestor(self, node: int, k: int) -> int:
        while node != -1 and k > 0:
            j = int(math.log2(k))
            self.maxWeight = max(self.maxWeight, self.maxJumpWeight[node][j])
            node = self.jump[node][j]
            k -= (1<<j)
        return node
        
class DistanceLimitedPathsExist:

    def __init__(self, n: int, edgeList: List[List[int]]):
        # CONSTRUCT THE MINIMUM SPANNING TREE 
        edgeList.sort(key=lambda edge: edge[2])
        self.dsu = UnionFind(n)
        graph = [[] for _ in range(n)]
        weight = defaultdict(int)
        for u, v, w in edgeList:
            if self.dsu.union(u,v):
                graph[u].append((v,w))
                graph[v].append((u,w))
        self.binary_lift = BinaryLift(n, graph)

    def query(self, p: int, q: int, limit: int) -> bool:
        # CHECK THAT BOTH P AND Q BELONG TO SAME MINIMUM SPANNING TREE, THAT IS ARE IN THE SAME DISJOINT SET OF NODES
        if self.dsu.find(p) != self.dsu.find(q): return False
        # COMPUTE THE MAX WEIGHT WHILE FINDING THE LOWEST COMMON ANCESTOR
        maxWeight = self.binary_lift.max_weight_lca(p,q)
        return maxWeight < limit
```

## 1627. Graph Connectivity With Threshold

### Solution 1:  union find + prime sieve + memoization

```py
class UnionFind:
    def __init__(self,n):
        self.size = [1]*n
        self.parent = list(range(n))
    
    def find(self,i):
        if i==self.parent[i]:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self,i,j):
        i, j = self.find(i), self.find(j)
        if i!=j:
            if self.size[i] < self.size[j]:
                i,j=j,i
            self.parent[j] = i
            self.size[i] += self.size[j]
            return True
        return False
    def __repr__(self):
        return f'parents: {[(i, parent) for i, parent in enumerate(self.parent)]}, sizes: {self.size}'
    
class Solution:
    def areConnected(self, n: int, threshold: int, queries: List[List[int]]) -> List[bool]:
        if threshold == 0: return [True]*len(queries)
        def prime_sieve(lim):
            sieve, primes = [[] for _ in range(lim)], []
            for integer in range(2,lim):
                if not len(sieve[integer]):
                    primes.append(integer)
                    for possibly_divisible_integer in range(integer,lim,integer):
                        current_integer = possibly_divisible_integer
                        while not current_integer%integer:
                            sieve[possibly_divisible_integer].append(integer)
                            current_integer //= integer
            return primes
        dsu = UnionFind(n+1)
        queue = deque(prime_sieve(n+1))
        visited = [False]*(n+1)
        while queue:
            integer = queue.popleft()
            visited[integer] = True
            for i in range(integer+integer, n+1, integer):
                if integer > threshold:
                    dsu.union(integer, i)
                if not visited[i]:
                    queue.append(i)
                    visited[i] = True
        return [dsu.find(x)==dsu.find(y) for x,y in queries]
```

## 2399. Check Distances Between Same Letters

### Solution 1:  string + re.finditer

```py
class Solution:
    def checkDistances(self, s: str, distance: List[int]) -> bool:
        for ch in string.ascii_lowercase:
            v = ord(ch)-ord('a')
            indices = [i.start() for i in re.finditer(ch, s)]
            if indices and indices[1]-indices[0]-1 != distance[v]: return False
        return True
```

## 2400. Number of Ways to Reach a Position After Exactly k Steps

### Solution 1:  iterative dynamic programming + set of previous positions + memoization

```py
class Solution:
    def numberOfWays(self, startPos: int, endPos: int, k: int) -> int:
        mod = int(1e9)+7
        memo = defaultdict(int)
        prev_positions = set([startPos])
        memo[(startPos,0)] = 1
        for i in range(k):
            next_positions = set()
            for pos in prev_positions:
                if k-i < abs(pos-endPos): continue
                for step in [-1,1]:
                    state = (pos+step, i+1)
                    memo[state] = (memo[state]+memo[(pos,i)])%mod
                    next_positions.add(pos+step)
            prev_positions = next_positions
        return memo[(endPos, k)]
```

## 2401. Longest Nice Subarray

### Solution 1:  sliding window + bit manipulation + count of bits must be less than or equal to 1 for subarray to work

```py
class Solution:
    def longestNiceSubarray(self, nums: List[int]) -> int:
        left = 0
        bit_counts = [0]*32
        maxWindow = 0
        over = 0
        for right, num in enumerate(nums):
            for i in range(32):
                if (num>>i)&1:
                    bit_counts[i] += 1
                    if bit_counts[i] > 1:
                        over += 1
            while over:
                pnum = nums[left]
                left += 1
                for i in range(32):
                    if (pnum>>i)&1:
                        bit_counts[i] -= 1
                        if bit_counts[i] == 1:
                            over -= 1
            maxWindow = max(maxWindow, right-left+1)
        return maxWindow
```

## 2402. Meeting Rooms III

### Solution 1:  two minheap + minheap for room indices + minheap for (endtime, room_index)

```py
class Solution:
    def mostBooked(self, n: int, meetings: List[List[int]]) -> int:
        m = len(meetings)
        rooms = list(range(n))
        heapify(rooms)
        waiting = []
        counts = [0] * n
        for s, e in sorted(meetings):
            while waiting and waiting[0][0] <= s:
                _, i = heappop(waiting)
                heappush(rooms, i)
            time = s
            if not rooms:
                time, i = heappop(waiting)
                heappush(rooms, i)
            i = heappop(rooms)
            counts[i] += 1
            heappush(waiting, (time - s + e, i))
        return max(range(n), key = lambda i: counts[i])
```

## 987. Vertical Order Traversal of a Binary Tree

### Solution 1:  bfs + deque + sort + hash table

```py
class Solution:
    def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:
        cols = defaultdict(list)
        cols[0].append(root.val)
        queue = deque([(root, 0)])
        while queue:
            sz = len(queue)
            row = defaultdict(list)
            for _ in range(sz):
                node, col = queue.popleft()
                for nei, nc in zip([node.left, node.right], [col-1, col+1]):
                    if not nei: continue
                    queue.append((nei, nc))
                    row[nc].append(nei.val)
            for c in row: cols[c].extend(sorted(row[c]))
        return [cols[c] for c in sorted(cols)]
```

## 2393. Count Strictly Increasing Subarrays

### Solution 1:  math + space optimized dyanmic programming

```py
class Solution:
    def countSubarrays(self, nums: List[int]) -> int:
        delta = pnum = result = 0
        for num in nums:
            if num > pnum:
                delta += 1
            else:
                delta = 1
            result += delta
            pnum = num
        return result
```

## 429. N-ary Tree Level Order Traversal

### Solution 1:  bfs + deque

```py
class Solution:
    def levelOrder(self, root: 'Node') -> List[List[int]]:
        result = []
        if not root: return result
        queue = deque([root])
        while queue:
            sz = len(queue)
            level = []
            for _ in range(sz):
                node = queue.popleft()
                level.append(node.val)
                for child in node.children:
                    queue.append(child)
            result.append(level)
        return result
```

## 814. Binary Tree Pruning

### Solution 1:  recursion + postorder traversal to update binary tree

```py
class Solution:
    def pruneTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        def containsOne(node: TreeNode) -> bool:
            if not node: return False
            left_contains_one = containsOne(node.left)
            right_contains_one = containsOne(node.right)
            if not left_contains_one:
                node.left = None
            if not right_contains_one:
                node.right = None
            return node.val or left_contains_one or right_contains_one
        containsOne(root)
        return root if containsOne(root) else None
```

## 2394. Employees With Deductions

### Solution 1:  ifnull + left join + group by + timestampdiff

```sql
WITH work_tbl AS (
    SELECT
        l.employee_id,
        IFNULL(SUM(CEIL(TIMESTAMPDIFF(SECOND, r.in_time, r.out_time)/60)), 0) AS work_time,
        l.needed_hours*60 AS needed_minutes
    FROM Employees l
    LEFT JOIN Logs r
    ON l.employee_id = r.employee_id
    GROUP BY employee_id
),
deduct_tbl AS (
    SELECT employee_id
    FROM work_tbl
    WHERE work_time < needed_minutes
)
SELECT *
FROM deduct_tbl
```

## 305. Number of Islands II

### Solution 1:  union find + set + disjoint connected components

```py
class UnionFind:
    def __init__(self,n: int):
        self.size = [1]*n
        self.parent = list(range(n))
    
    def find(self,i: int) -> int:
        if i==self.parent[i]:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self,i: int,j: int) -> bool:
        i, j = self.find(i), self.find(j)
        if i!=j:
            if self.size[i] < self.size[j]:
                i,j=j,i
            self.parent[j] = i
            self.size[i] += self.size[j]
            return True
        return False
    
    def __repr__(self) -> str:
        return f'parents: {[(i, parent) for i, parent in enumerate(self.parent)]}, sizes: {self.size}'
    
class Solution:
    def numIslands2(self, m: int, n: int, positions: List[List[int]]) -> List[int]:
        get_index = lambda row, col: row*n+col
        dsu = UnionFind(m*n)
        k = len(positions)
        answer = [0]*k
        connected_components = 0
        processed = set()
        def neighbors(r: int, c: int) -> Iterable[Tuple]:
            in_bounds = lambda r, c: 0<=r<m and 0<=c<n
            for nr, nc in [(r+1,c), (r-1,c), (r,c+1), (r,c-1)]:
                if not in_bounds(nr,nc): continue
                yield (nr, nc)
        for i, (r,c) in enumerate(positions):
            index = get_index(r,c)
            if index in processed: 
                answer[i] = connected_components
                continue
            surrounding_components = set()
            for nr, nc in neighbors(r,c):
                ni = get_index(nr,nc)
                if ni not in processed: continue
                surrounding_components.add(dsu.find(ni))
            processed.add(index)
            for si in surrounding_components:
                dsu.union(index,si)
            if len(surrounding_components) > 0:
                connected_components -= (len(surrounding_components)-1)
            else:
                connected_components += 1
            answer[i] = connected_components
        return answer
```

### Solution 2:  compact union find + union find with dictionary and sets + space optimized union find

```py
class UnionFind:
    def __init__(self):
        self.size = dict()
        self.parent = dict()
        self.connected_components = set()
        self.count_connected_components = 0
        
    def add(self, i: int) -> None:
        if i not in self.connected_components:
            self.connected_components.add(i)
            self.parent[i] = i
            self.size[i] = 1
        self.count_connected_components += 1
    
    def find(self,i: int) -> int:
        if i==self.parent[i]:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self,i: int,j: int) -> bool:
        # FIND THE REPRESENTATIVE NODE FOR THESE NODES IF THEY ARE ALREADY BELONGING TO CONNECTED COMPONENTS
        i, j = self.find(i), self.find(j)
        if i!=j:
            if self.size[i] < self.size[j]:
                i,j=j,i
            self.parent[j] = i
            self.size[i] += self.size[j]
            self.count_connected_components -= 1
            return True
        return False
    
    def __repr__(self) -> str:
        return f'parents: {self.parent}, sizes: {self.size}, connected_components: {self.connected_components}'
    
class Solution:
    def numIslands2(self, m: int, n: int, positions: List[List[int]]) -> List[int]:
        get_index = lambda row, col: row*n+col
        dsu = UnionFind()
        k = len(positions)
        answer = [0]*k
        def neighbors(r: int, c: int) -> Iterable[Tuple]:
            in_bounds = lambda r, c: 0<=r<m and 0<=c<n
            for nr, nc in [(r+1,c), (r-1,c), (r,c+1), (r,c-1)]:
                if not in_bounds(nr,nc): continue
                yield (nr, nc)
        for i, (r,c) in enumerate(positions):
            index = get_index(r,c)
            # POSITION THAT IS ALREADY LAND
            if index in dsu.connected_components: 
                answer[i] = dsu.count_connected_components
                continue
            # ADD CURRENT NODE TO CONNECTED COMPONENTS
            dsu.add(index)
            for nr, nc in neighbors(r,c):
                ni = get_index(nr,nc)
                if ni not in dsu.connected_components: continue
                # FIND THE ROOT NODE AND ADD IT, (ROOT NODE IS ALSO CALLED REPRESENTATIVE NODE IN UNION FIND)
                dsu.union(index, ni)
            # UPDATING THE TOTAL COUNT OF CONNECTED COMPONENTS, IF THERE ARE MULTIPLE SURROUNDING COMPONENTS, THEY GET MERGED SO IT NEEDS TO DECREASE
            answer[i] = dsu.count_connected_components
        return answer
            
```

## 362. Design Hit Counter

### Solution 1:  Scales with regards to the time range + queue

```py
class HitCounter:

    def __init__(self):
        self.queue = deque()
        self.hit_counts = 0
        self.time_range = 300

    def hit(self, timestamp: int) -> None:
        if self.queue and self.queue[-1][0] == timestamp:
            self.queue[-1][1] += 1
        else:
            self.queue.append([timestamp, 1])
        self.hit_counts += 1

    def getHits(self, timestamp: int) -> int:
        while self.queue and self.queue[0][0] <= timestamp-self.time_range:
            _, count = self.queue.popleft()
            self.hit_counts -= count
        return self.hit_counts
```

### Solution 2:  mutable dataclass for hit data

```py
from dataclasses import make_dataclass, field
class HitCounter:

    def __init__(self):
        self.queue = deque()
        self.hit_counts = 0
        self.time_range = 300
        self.HitData = make_dataclass('HitData', [('timestamp', int), ('count', int, field(default=1))])

    def hit(self, timestamp: int) -> None:
        if self.queue and self.queue[-1].timestamp == timestamp:
            self.queue[-1].count += 1
        else:
            self.queue.append(self.HitData(timestamp))
        self.hit_counts += 1

    def getHits(self, timestamp: int) -> int:
        while self.queue and self.queue[0].timestamp <= timestamp-self.time_range:
            self.hit_counts -= self.queue.popleft().count
        return self.hit_counts
```

## 1996. The Number of Weak Characters in the Game

### Solution 1:  greedy + group by attack in descending order + store maxDefence

![weak characters](images/weak_character.png)

```py
class Solution:
    def numberOfWeakCharacters(self, properties: List[List[int]]) -> int:
        attack_dict = defaultdict(list)
        for attack, defence in properties:
            attack_dict[attack].append(defence)
        maxDefence = 0
        result = 0
        for attack in sorted(attack_dict, reverse=True):
            tmp_maxDefence = 0
            for defence in attack_dict[attack]:
                result += (defence<maxDefence)
                tmp_maxDefence = max(tmp_maxDefence, defence)
            maxDefence = max(maxDefence, tmp_maxDefence)
        return result
```

## 352. Data Stream as Disjoint Intervals

### Solution 1:  dictionary + set + look at the left and right intervals to find the start and end point

![data stream](images/data_stream_as_disjoint_intervals.png)

```py
class SummaryRanges:

    def __init__(self):
        self.start = {}
        self.end = {}
        self.seen = set()

    def addNum(self, val: int) -> None:
        if val in self.seen: return
        self.seen.add(val)
        start = end = val
        if val+1 in self.start:
            end = self.start[val+1]
            self.start.pop(val+1)
        if val-1 in self.end:
            start = self.end[val-1]
            self.end.pop(val-1)
        self.start[start] = end
        self.end[end] = start
        
    def getIntervals(self) -> List[List[int]]:
        return sorted([interval for interval in self.start.items()])
```

## 1179. Reformat Department Table

### Solution 1: pivot table with sum and if statements + group by

```sql
SELECT 
    id,
    sum(if(month='Jan', revenue, null)) AS Jan_Revenue,
    sum(if(month='Feb', revenue, null)) AS Feb_Revenue,
    sum(if(month='Mar', revenue, null)) AS Mar_Revenue,
    sum(if(month='Apr', revenue, null)) AS Apr_Revenue,
    sum(if(month='May', revenue, null)) AS May_Revenue,
    sum(if(month='Jun', revenue, null)) AS Jun_Revenue,
    sum(if(month='Jul', revenue, null)) AS Jul_Revenue,
    sum(if(month='Aug', revenue, null)) AS Aug_Revenue,
    sum(if(month='Sep', revenue, null)) AS Sep_Revenue,
    sum(if(month='Oct', revenue, null)) AS Oct_Revenue,
    sum(if(month='Nov', revenue, null)) AS Nov_Revenue,
    sum(if(month='Dec', revenue, null)) AS Dec_Revenue
FROM Department
GROUP BY id
```

## 2404. Most Frequent Even Element

### Solution 1:

```py
class Solution:
    def mostFrequentEven(self, nums: List[int]) -> int:
        m = Counter(nums)
        maxCount = 0
        val = -1
        for key in sorted(m.keys()):
            if key%2==0:
                if m[key]>maxCount:
                    maxCount = m[key]
                    val = key
        return val
```

## 2405. Optimal Partition of String

### Solution 1:

```py
class Solution:
    def partitionString(self, s: str) -> int:
        result = 1
        cnt = [0]*26
        get_unicode = lambda ch: ord(ch)-ord('a')
        for ch in s:
            val = get_unicode(ch)
            cnt[val] += 1
            if cnt[val] > 1:
                cnt = [0]*26
                cnt[val] += 1
                result += 1
        return result
```

```py
class Solution:
    def partitionString(self, s: str) -> int:
        last_seen = [-1]*26
        res, start = 1, 0
        unicode = lambda ch: ord(ch) - ord('a')
        for i, v in enumerate(map(unicode, s)):
            if last_seen[v] >= start:
                res += 1
                start = i
            last_seen[v] = i
        return res
```

## 2406. Divide Intervals Into Minimum Number of Groups

### Solution 1: line sweep + sort

```py
class Solution:
    def minGroups(self, intervals: List[List[int]]) -> int:
        events = []
        for start, end in intervals:
            events.append((start, 1))
            events.append((end+1,-1))
        events.sort()
        count = maxCount = 0
        for event, delta in events:
            count += delta
            maxCount = max(maxCount, count)
        return maxCount
```

## 2407. Longest Increasing Subsequence II

### Solution 1:  max segment tree + max range queries + LIS ending at each element

![lis](images/lis2.png)

```py
class SegmentTree:
    def __init__(self, n: int, neutral: int, func: Callable[[int, int], int], is_count: bool = False):
        self.neutral = neutral
        self.size = 1
        self.is_count = is_count
        self.func = func
        self.n = n
        while self.size<n:
            self.size*=2
        self.tree = [0 for _ in range(self.size*2)]
        
    def update(self, idx: int, val: int) -> None:
        idx += self.size - 1
        self.tree[idx] = self.tree[idx] + val if self.is_count else val
        while idx > 0:
            idx -= 1
            idx >>= 1
            self.tree[idx] = self.func(self.tree[2*idx+1], self.tree[2*idx+2])
            
    def query(self, l: int, r: int) -> int:
        stack = [(0, self.size, 0)]
        result = 0
        while stack:
            # BOUNDS FOR CURRENT INTERVAL and idx for tree
            left_bound, right_bound, idx = stack.pop()
            # OUT OF BOUNDS
            if left_bound >= r or right_bound <= l: continue
            # CHECK IF CURRENT BOUNDS ARE WITHIN THE l and r
            if left_bound >= l and right_bound <= r:
                result = self.func(result, self.tree[idx])
                continue
            mid = (left_bound + right_bound)>>1
            stack.extend([(left_bound, mid, 2*idx+1), (mid, right_bound, 2*idx+2)])
        return result
    
    def __repr__(self) -> str:
        return f"array: {self.tree}"

class Solution:
    def lengthOfLIS(self, nums: List[int], k: int) -> int:
        max_func = lambda x, y: x if x > y else y
        n, ans = max(nums), 1
        maxSeg = SegmentTree(n+1, -inf, max_func)
        for num in nums:
            premax = maxSeg.query(max(0, num - k), num)
            if premax + 1 > ans:
                ans = premax+1
            maxSeg.update(num, premax + 1)
        return ans
```

## 948. Bag of Tokens

### Solution 1:  sort + greedy + place face up cards with lowest power consumption + place face down cards with highest power consumption

```py
class Solution:
    def bagOfTokensScore(self, tokens: List[int], power: int) -> int:
        tokens.sort()
        n = len(tokens)
        ans = score = i = 0
        for j in reversed(range(n)):
            while i <= j and tokens[i] <= power:
                power -= tokens[i]
                i += 1
                score += 1
            ans = max(ans, score)
            power += tokens[j]
            score -= 1
            if score < 0: break
        return ans
```

## 1383. Maximum Performance of a Team

### Solution 1:  minheap + sort based on efficiency since taking minimum

```py
class Solution:
    def maxPerformance(self, n: int, speed: List[int], efficiency: List[int], k: int) -> int:
        mod = int(1e9)+7
        maxPerf = 0
        stats = sorted([(eff, spd) for eff, spd in zip(efficiency, speed)], reverse=True)
        speed_sum = 0
        minheap = []
        for eff, spd in stats:
            speed_sum += spd
            heappush(minheap, spd)
            if len(minheap) > k:
                prev_spd = heappop(minheap)
                speed_sum -= prev_spd
            maxPerf = max(maxPerf, eff*speed_sum)
        return maxPerf%mod
```

## 1457. Pseudo-Palindromic Paths in a Binary Tree

### Solution 1:  bitmask + queue + deque + bfs

```py
class Solution:
    def pseudoPalindromicPaths (self, root: Optional[TreeNode]) -> int:
        result = 0
        is_leaf = lambda node: not node.left and not node.right
        isppalindrome = lambda mask: mask.bit_count()<=1
        queue = deque([(root, 0)])
        while queue:
            node, bitmask = queue.popleft()
            bitmask ^= (1<<(node.val-1))
            if is_leaf(node):
                result += isppalindrome(bitmask)
            queue.extend([(child_node, bitmask) for child_node in filter(None, (node.left, node.right))])
        return result
```

## 2007. Find Original Array From Doubled Array

### Solution 1:  bfs + deque

```py
class Solution:
    def findOriginalArray(self, changed: List[int]) -> List[int]:
        queue = deque()
        result = []
        for num in sorted(changed):
            if queue and num == queue[0]:
                queue.popleft()
            else:
                queue.append(2*num)
                result.append(num)
        return result if len(queue) == 0 else []
```

## 1770. Maximum Score from Performing Multiplication Operations

### Solution 1:  iterative dynamic programming

![dp](images/maximum_score_from_performing_multiplication_operations.png)

```py
class Solution:
    def maximumScore(self, nums: List[int], multipliers: List[int]) -> int:
        n, m = len(nums), len(multipliers)
        memo = [[0]*(m+1) for _ in range(m+1)]
        for i in reversed(range(m)):
            for left in range(i+1):
                right = left+n-i-1
                take_left = multipliers[i]*nums[left]+memo[i+1][left+1]
                take_right = multipliers[i]*nums[right]+memo[i+1][left]
                memo[i][left] = max(take_left, take_right)
        return memo[0][0]
```

## 336. Palindrome Pairs

### Solution 1:  three cases to solve

![palindrome pairs](images/palindrome_pairs.png)

```py
class Solution:
    def is_palindrome(self, word: str) -> bool:
        left, right = 0, len(word)-1
        while left < right and word[left] == word[right]:
            left += 1
            right -= 1
        return left >= right
    def palindromePairs(self, words: List[str]) -> List[List[int]]:
        n = len(words)
        word_dict = {word: i for i, word in enumerate(words)}
        result = set()
        for i in range(n):
            for j in range(len(words[i])+1):
                prefix_word = words[i][:j]
                suffix_palindrome = words[i][j:]
                rev_word1 = prefix_word[::-1]
                if rev_word1 in word_dict and word_dict[rev_word1] != i and self.is_palindrome(suffix_palindrome):
                    result.add((i, word_dict[rev_word1]))
                prefix_palindrome = prefix_word
                suffix_word = suffix_palindrome
                rev_word2 = suffix_word[::-1]
                if rev_word2 in word_dict and word_dict[rev_word2] != i and self.is_palindrome(prefix_palindrome):
                    result.add((word_dict[rev_word2], i))
        return result
```

## 393. UTF-8 Validation

### Solution 1:  bit manipulation + bitmask

```py
class Solution:
    def validUtf8(self, data: List[int]) -> bool:
        filler_bitmask = int('10', 2)
        is_1byte = lambda mask: (mask>>7)==0
        def is_nbyte(n, cand):
            mask = ''
            for _ in range(n):
                mask += '1'
            mask += '0'
            shifts = 7-n
            return (cand>>shifts) == int(mask,2)
        n = len(data)
        i = 0
        while i < n:
            d = data[i]
            if is_1byte(d):
                i += 1
                continue
            b = None
            for j in range(2,5):
                if is_nbyte(j,d):
                    b = j
                    break
            if b is None: return False
            i += 1
            for _ in range(b-1):
                if i == n: return False
                if (data[i]>>6) != filler_bitmask: return False
                i += 1

        return True
```

## 609. Find Duplicate File in System

### Solution 1:  regex for pattern matching in strings + dictionary + filter

```py
class Solution:
    def findDuplicate(self, paths: List[str]) -> List[List[str]]:
        paths_dict = defaultdict(list)
        for path in paths:
            directory = path.split()[0]
            for file_name, file_content in zip(re.findall('\w*\.txt', path), re.findall('\(.*?\)', path)):
                file_path = f'{directory}/{file_name}'
                paths_dict[file_content].append(file_path)
        return filter(lambda lst: len(lst)>1, paths_dict.values())
```

## 2413. Smallest Even Multiple

### Solution 1:  greedy + simple cases

```py
class Solution:
    def smallestEvenMultiple(self, n: int) -> int:
        return n if n%2==0 else 2*n
```

## 2414. Length of the Longest Alphabetical Continuous Substring

### Solution 1:  two pointer

```py
class Solution:
    def longestContinuousSubstring(self, s: str) -> int:
        cur = result = 1
        n = len(s)
        for i in range(1,n):
            if ord(s[i]) == ord(s[i-1])+1:
                cur += 1
            else:
                cur = 1
            result = max(result, cur)
        return result
```

## 2415. Reverse Odd Levels of Binary Tree

### Solution 1:  dictionary + dfs

```py
class Solution:
    def reverseOddLevels(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        levels = defaultdict(list)
        def dfs(node: Optional[TreeNode], depth: int = 0) -> None:
            if not node: return
            levels[depth].append(node)
            dfs(node.left,depth+1)
            dfs(node.right,depth+1)
        dfs(root)
        for level in levels.keys():
            if level&1:
                row = levels[level]
                values = [node.val for node in row]
                for node, val in zip(row, reversed(values)):
                    node.val = val
        return root
```

## 2416. Sum of Prefix Scores of Strings

### Solution 1:  rolling hash + cannot use mod 1e9+7 because it somehow has a hash collision

```py
class Solution:
    def sumPrefixScores(self, words: List[str]) -> List[int]:
        p = 31
        coefficient = lambda x: ord(x) - ord('a') + 1
        n = len(words)
        prefix_dict = defaultdict(list)
        for i, word in enumerate(words):
            rolling_hash = 0
            for ch in word:
                rolling_hash = rolling_hash*p+coefficient(ch)
                prefix_dict[rolling_hash].append(i)
        answer = [0]*n
        for array in prefix_dict.values():
            cnt = len(array)
            for i in array:
                answer[i] += cnt
        return answer
```

### Solution 2:  trie datastructure

```py
class TrieNode:
    def __init__(self, count_: int = 0):
        self.children = defaultdict(TrieNode)
        self.count = count_
        
    def __repr__(self) -> str:
        return f'count: {self.count}, children: {self.children}'
        
class Solution:
    def sumPrefixScores(self, words: List[str]) -> List[int]:
        root = TrieNode()
        for word in words:
            node = root
            for ch in word:
                node.children[ch].count += 1
                node = node.children[ch]
        answer = []
        for word in words:
            cnt = 0
            node = root
            for ch in word:
                node = node.children[ch]
                cnt += node.count
            answer.append(cnt)
        return answer
```

## 839. Similar String Groups

### Solution 1:  union find + count number of disjoint connected components

```py
class UnionFind:
    def __init__(self,n: int):
        self.size = [1]*n
        self.parent = list(range(n))
    
    def find(self,i: int) -> int:
        if i==self.parent[i]:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self,i: int,j: int) -> bool:
        i, j = self.find(i), self.find(j)
        if i!=j:
            if self.size[i] < self.size[j]:
                i,j=j,i
            self.parent[j] = i
            self.size[i] += self.size[j]
            return True
        return False
    def num_connected_components(self) -> int:
        return len(set(map(self.find, self.parent)))
    def __repr__(self) -> str:
        return f'parents: {[(i, parent) for i, parent in enumerate(self.parent)]}, sizes: {self.size}'
class Solution:
    def numSimilarGroups(self, strs: List[str]) -> int:
        strs = list(set(strs))
        n = len(strs)
        dsu = UnionFind(n)
        edit_distance = lambda s1, s2: sum([1 for x, y in zip(s1, s2) if x != y])
        for i in range(n):
            for j in range(i):
                if edit_distance(strs[i], strs[j]) in (0, 2):
                    dsu.union(i, j)
        return dsu.num_connected_components()
```

## 269. Alien Dictionary

### Solution 1:  topological sort + cycle detection + dfs graph traversal with recursion

```py
class Solution:
    def alienOrder(self, words: List[str]) -> str:
        graph = defaultdict(list)
        indegrees = Counter()
        characters = set()
        for word in words:
            characters.update(word)
        def dfs(word_group: List[str]) -> bool:
            cur_chars = [cword[0] for cword in word_group]
            for x, y in zip(cur_chars, cur_chars[1:]):
                if x != y:
                    indegrees[y] += 1
                    graph[x].append(y)
            result = True
            for key, cand_next_batch in groupby(word_group, key=lambda word: word[0]):
                next_batch = []
                for word in cand_next_batch:
                    if len(word) == 1 and next_batch: return False
                    if len(word) > 1:
                        next_batch.append(word[1:])
                result &= dfs(next_batch)
            return result
        if not dfs(words): return ''
        queue = deque()
        for char in characters:
            if indegrees[char] == 0:
                queue.append(char)
        alien_alphabet = []
        while queue:
            char = queue.popleft()
            alien_alphabet.append(char)
            for nei_char in graph[char]:
                indegrees[nei_char] -= 1
                if indegrees[nei_char] == 0:
                    queue.append(nei_char)
        return '' if any(ind > 0 for ind in indegrees.values()) else ''.join(alien_alphabet)
```

## 1494. Parallel Courses II

### Solution 1:  topological sort using dependency masks + memoization to minimize semesters for any courses taken + bfs + reduce with using | bit operator

```py
class Solution:
    def minNumberOfSemesters(self, n: int, dependencies: List[List[int]], k: int) -> int:
        memo = [16]*(1<<n)
        dep_masks = [0]*n
        for u, v in dependencies:
            u -= 1
            v -= 1
            dep_masks[v] |= (1<<u)
        queue = deque([(0, 0)]) # (mask, semesters)
        end_mask = (1<<n)-1
        while queue:
            mask, semester = queue.popleft()
            if mask == end_mask: return semester
            semester += 1
            courses = []
            for course in range(n):
                if (mask>>course)&1: continue # course has already been taken
                if (dep_masks[course]&mask) != dep_masks[course]: continue # prerequisites not complete for course
                courses.append(course)
            if len(courses) <= k: # take all courses
                mask |= reduce(lambda x, y: x | (1<<y), courses, 0) # mark all courses as taken
                if semester < memo[mask]:
                    memo[mask] = semester
                    queue.append((mask, semester))
            else:
                for course_plan in combinations(courses, k):
                    course_plan = list(course_plan)
                    mask_plan = mask | reduce(lambda x, y: x | (1<<y), course_plan, 0)
                    if semester < memo[mask_plan]:
                        memo[mask_plan] = semester
                        queue.append((mask_plan, semester))
        return -1
```

## 685. Redundant Connection II

### Solution 1:  remove each edge and perform a topological sort on the rooted tree + filterfalse

```py
class Solution:
    def findRedundantDirectedConnection(self, edges: List[List[int]]) -> List[int]:
        n = len(edges)
        edges = edges[::-1]
        indegrees = [0]*(n+1)
        graph = defaultdict(list)
        for u, v in edges:
            indegrees[v] += 1
            graph[u].append(v)
        root_node = 0
        for i in range(1,n+1):
            if indegrees[i] == 0:
                root_node = i
        two_root_nodes = lambda node: indegrees[node] == 1 and root_node != 0
        no_root_nodes = lambda node: indegrees[node] > 1 and root_node == 0
        for u, v in filterfalse(lambda edge: two_root_nodes(edge[1]) or no_root_nodes(edge[1]), edges):
            temp_indegrees = indegrees[:]
            temp_indegrees[v] -= 1
            queue = deque()
            if root_node != 0:
                queue.append(root_node)
            if temp_indegrees[v] == 0:
                queue.append(v)
            cnt_nodes = 0
            while queue:
                node = queue.popleft()
                cnt_nodes += 1
                for nei_node in graph[node]:
                    if [node, nei_node] == [u,v]: continue # removed edge
                    temp_indegrees[nei_node] -= 1
                    if temp_indegrees[nei_node] == 0:
                        queue.append(nei_node)
            if cnt_nodes == n: return [u,v]
        return [0,0]
```

## 2421. Number of Good Paths

### Solution 1:  union find + undirected graph + sort

```py
class UnionFind:
    def __init__(self,n: int):
        self.size = [1]*n
        self.parent = list(range(n))
    
    def find(self,i: int) -> int:
        if i==self.parent[i]:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self,i: int,j: int) -> bool:
        i, j = self.find(i), self.find(j)
        if i!=j:
            if self.size[i] < self.size[j]:
                i,j=j,i
            self.parent[j] = i
            self.size[i] += self.size[j]
            return True
        return False
    def __repr__(self) -> str:
        return f'parents: {[(i, parent) for i, parent in enumerate(self.parent)]}, sizes: {self.size}'
class Solution:
    def numberOfGoodPaths(self, vals: List[int], edges: List[List[int]]) -> int:
        n = len(vals)
        adjList = [[] for _ in range(n)] # adjacency representation
        for u, v in edges:
            adjList[u].append(v)
            adjList[v].append(u)
        dsu = UnionFind(n)
        order_nodes = defaultdict(list)
        for i, v in enumerate(vals):
            order_nodes[v].append(i)
        result = 0
        for val in sorted(order_nodes.keys()):
            nodes_list = order_nodes[val]
            for node in nodes_list:
                for nei_nodes in adjList[node]:
                    if vals[nei_nodes] <= val:
                        dsu.union(node, nei_nodes)
            cnt = Counter()
            for node in nodes_list:
                root_node = dsu.find(node)
                cnt[root_node] += 1
                result += cnt[root_node]
        return result
```

## 2417. Closest Fair Integer

### Solution 1:  count parity of integer + brute force on special case + special state

```py
class Solution:
    def odd_len(self, len_: int) -> int:
        size = (len_+1)//2
        integer_list = [1]+[0]*(size)+[1]*(size-1)
        return int(''.join(map(str,integer_list)))
    def count_parity(self, integer: int) -> Tuple[int,int]:
        odd = sum(1 for dig in map(int, str(integer)) if dig&1)
        even = sum(1 for dig in map(int, str(integer)) if dig%2==0)
        return odd, even 
    def closestFair(self, n: int) -> int:
        odd, even = self.count_parity(n)
        while even != odd:
            len_n = len(str(n))
            n = self.odd_len(len_n) if len_n&1 else n + 1
            odd, even = self.count_parity(n)
        return n
```

## 2420. Find All Good Indices

### Solution 1:  set union + monotonic stack

```py
class Solution:
    def build(self, it: Iterable[int], nums: List[int], k: int) -> Iterable[int]:
        stack = []
        for i in it:
            if len(stack) >= k:
                yield i
            if stack and nums[stack[-1]] < nums[i]:
                stack.clear()
            stack.append(i)
    def goodIndices(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        g1 = self.build(range(n), nums, k)
        g2 = self.build(reversed(range(n)), nums, k)
        return sorted(list(set(g1)&set(g2)))
```

## 2419. Longest Subarray With Maximum Bitwise AND

### Solution 1:  maximum value will be the max bitwise and in the array, so just find longest group with that key

```py
class Solution:
    def longestSubarray(self, nums: List[int]) -> int:
        m = max(nums)
        longest = 0
        for k, g in groupby(nums):
            if k == m:
                longest = max(longest, len(list(g)))
        return longest
```

## 2418. Sort the People

### Solution 1:  sort + zip

```py
class Solution:
    def sortPeople(self, names: List[str], heights: List[int]) -> List[str]:
        return [names[i] for i in sorted(range(len(names)), key=lambda i: heights[i], reverse=True)]
```

## 622. Design Circular Queue

### Solution 1:  two pointers for front and rear + array

```py
class MyCircularQueue:

    def __init__(self, k: int):
        self.rear_ptr = -1
        self.front_ptr = self.cnt = 0
        self.cap = k
        self.arr = [0]*k

    def enQueue(self, value: int) -> bool:
        if self.cnt == self.cap: return False
        self.rear_ptr = (self.rear_ptr+1)%self.cap
        self.arr[self.rear_ptr] = value
        self.cnt += 1
        return True

    def deQueue(self) -> bool:
        if self.cnt == 0: return False
        self.front_ptr = (self.front_ptr+1)%self.cap
        self.cnt -= 1
        return True
        

    def Front(self) -> int:
        return self.arr[self.front_ptr] if self.cnt > 0 else -1

    def Rear(self) -> int:
        return self.arr[self.rear_ptr] if self.cnt > 0 else -1

    def isEmpty(self) -> bool:
        return self.cnt == 0

    def isFull(self) -> bool:
        return self.cnt == self.cap
```

## 913. Cat and Mouse

### Solution 1:

```py
class Solution:
    def catMouseGame(self, graph: List[List[int]]) -> int:
        mouse_wins, cat_wins, draw = 0, 2, 1
        mouse_start_pos, cat_start_pos, hole_pos, start_turn = 1, 2, 0, 0
        initial_node = (mouse_start_pos, cat_start_pos, start_turn)
        queue = deque([initial_node])
        visited = set([initial_node])
        previous_nodes = set()
        path = []
        reverse_adj_list = defaultdict(list)
        outdegrees = Counter()
        memo = dict()
        cnt = 0
        def get_neighbors(pos):
            for nei_pos in graph[pos]:
                yield nei_pos
        is_cat_turn = lambda turn: turn&1
        is_mouse_turn = lambda turn: turn%2==0
        while queue:
            node = queue.popleft()
            mouse_pos, cat_pos, turn = node
            if mouse_pos == cat_pos: 
                memo[node] = cat_wins
                continue
            if mouse_pos == hole_pos:
                memo[node] = mouse_wins
                continue
            if (mouse_pos, cat_pos, turn%2) in previous_nodes: 
                memo[node] = draw
                continue
            previous_nodes.add((mouse_pos, cat_pos, turn%2))
            path.append(node)
            neighbors = get_neighbors(mouse_pos) if is_mouse_turn(turn) else get_neighbors(cat_pos)
            for nei_pos in neighbors:
                if is_cat_turn(turn) and nei_pos == hole_pos: continue
                nei_node = (nei_pos, cat_pos, turn+1) if is_mouse_turn(turn) else (mouse_pos, nei_pos, turn+1)
                reverse_adj_list[nei_node].append(node)
                outdegrees[node] += 1
                if nei_node in visited: continue
                visited.add(nei_node)
                queue.append(nei_node)
        queue = deque(memo.keys())
        while queue:
            node = queue.popleft()
            _, _, turn = node
            if turn&1:
                for nei_node in reverse_adj_list[node]:
                    outdegrees[nei_node] -= 1
                    memo[nei_node] = min(memo.get(nei_node, inf), memo[node])
                    if outdegrees[nei_node] == 0:
                        queue.append(nei_node)
            else:
                for nei_node in reverse_adj_list[node]:
                    outdegrees[nei_node] -= 1
                    memo[nei_node] = max(memo.get(nei_node, -inf), memo[node])
                    if outdegrees[nei_node] == 0:
                        queue.append(nei_node)
        result = memo[(1,2,0)]
        return result if result == 2 else result^1
```

## 1192. Critical Connections in a Network

### Solution 1:  tarjan's algorithm + dfs

```py
class Solution:
    def criticalConnections(self, n: int, connections: List[List[int]]) -> List[List[int]]:
        adjList = [[] for _ in range(n)]
        for u, v in connections:
            adjList[u].append(v)
            adjList[v].append(u)
        unvisited = inf
        self.disc = [unvisited]*n
        self.low = [unvisited]*n
        self.bridges = []
        self.cnt = 0
        def dfs(node: int = 0, parent_node: Optional[int] = None) -> None:
            if self.disc[node] != unvisited:
                return
            self.disc[node] = self.low[node] = self.cnt
            self.cnt += 1
            for nei_node in adjList[node]:
                if nei_node == parent_node: continue
                dfs(nei_node, node)
                if self.disc[node] < self.low[nei_node]:
                    self.bridges.append([node, nei_node])
                self.low[node] = min(self.low[node], self.low[nei_node])
        dfs()
        return self.bridges
```

## 834. Sum of Distances in Tree

### Solution 1:  preorder dfs + postorder dfs + math + dp on tree + ancestor and descendent sum + rooting undirected tree

![image](images/Sum_of_distances_in_tree_1.png)
![image](images/Sum_of_distances_in_tree_2.png)

```py
class Solution:
    def sumOfDistancesInTree(self, n: int, edges: List[List[int]]) -> List[int]:
        adj_list = [[] for _ in range(n)]
        for u, v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
        sizes, ancestor_dist, descendent_dist = [1]*n, [0]*n, [0]*n
        def postorder(parent: int, node: int) -> None:
            for child in adj_list[node]:
                if child == parent: continue
                postorder(node, child)
                sizes[node] += sizes[child]
                descendent_dist[node] += descendent_dist[child] + sizes[child]
        def preorder(parent: int, node: int) -> None:
            for child in adj_list[node]:
                if child == parent: continue
                ancestor_dist[child] = ancestor_dist[node] + (descendent_dist[node] - descendent_dist[child] - sizes[child]) + (n - sizes[child])
                preorder(node, child)
        postorder(-1, 0) # choose 0 as arbitrary root of tree
        preorder(-1, 0)
        return [anc_dist + des_dist for anc_dist, des_dist in zip(ancestor_dist, descendent_dist)]
```

## 1345. Jump Game IV

### Solution 1:

```py
class Solution:
    def minJumps(self, arr: List[int]) -> int:
        n = len(arr)
        jump_locations = defaultdict(list)
        for i in range(n):
            jump_locations[arr[i]].append(i)
        queue = deque([(0, 0)])
        vis = [0]*n
        vis[0] = 1
        while queue:
            idx, jumps = queue.popleft()
            if idx == n - 1: return jumps
            if idx > 0 and not vis[idx - 1]:
                vis[idx - 1] = 1
                queue.append((idx - 1, jumps + 1))
            if idx < n - 1 and not vis[idx + 1]:
                vis[idx + 1] = 1
                queue.append((idx + 1, jumps + 1))
            for jump_idx in jump_locations[arr[idx]]:
                if vis[jump_idx]: continue
                vis[jump_idx] = 1
                queue.append((jump_idx, jumps + 1))
            jump_locations.pop(arr[idx])
        return -1
```

```cpp
int minJumps(vector<int>& arr) {
    int n = arr.size();
    vector<bool> vis(n,false);
    unordered_map<int,vector<int>> values;
    for (int i = 0;i<n;i++) {
        values[arr[i]].push_back(i);
    }
    queue<int> q;
    q.push(0);
    vis[0] = true;
    int steps = 0;
    auto check = [&](const int i) {
        return i>=0 && i<n && !vis[i];
    };
    while (!q.empty()) {
        queue<int> nq;
        int sz = q.size();
        while (sz--) {
            int i = q.front();
            q.pop();
            if (i==n-1) {
                return steps;
            }
            for (int nei : values[arr[i]]) {
                if (!vis[nei]) {
                    vis[nei]=true;
                    nq.push(nei);
                }
            }
            values[arr[i]].clear();
            for (int j : {i-1,i+1}) {
                if (check(j)) {
                    vis[j]=true;
                    nq.push(j);
                }
            }
        }
        steps++;
        swap(q,nq);
    }
    return -1;
}
```

## 2423. Remove Letter To Equalize Frequency

### Solution 1:  counter + set

```py
class Solution:
    def equalFrequency(self, word: str) -> bool:
        freq = Counter(word)
        for ch in set(word):
            freq[ch] -= 1
            if freq[ch] == 0:
                freq.pop(ch)
            if len(set(freq.values())) == 1: return True
            freq[ch] += 1
        return False
```

## 2424. Longest Uploaded Prefix

### Solution 1:  array + pointer

```py
class LUPrefix:

    def __init__(self, n: int):
        self.video_arr = [0]*(n+1)
        self.prefix_ptr = 0

    def upload(self, video: int) -> None:
        self.video_arr[video-1] = 1

    def longest(self) -> int:
        while self.video_arr[self.prefix_ptr] == 1:
            self.prefix_ptr += 1
        return self.prefix_ptr
```

## 2425. Bitwise XOR of All Pairings

### Solution 1:  boolean algebra + bit manipulation

```py
class Solution:
    def xorAllNums(self, nums1: List[int], nums2: List[int]) -> int:
        n1 = reduce(xor, nums1) if len(nums2)&1 else 0
        n2 = reduce(xor, nums2) if len(nums1)&1 else 0
        return n1^n2
```

## 2426. Number of Pairs Satisfying Inequality

### Solution 1:  sorted list + binary search + backwards iteration

```py
from sortedcontainers import SortedList
class Solution:
    def numberOfPairs(self, nums1: List[int], nums2: List[int], diff: int) -> int:
        seenList = SortedList()
        n = len(nums1)
        result = 0
        for i in reversed(range(n)):
            j = seenList.bisect_left(nums1[i]-nums2[i])
            delta = len(seenList)-j
            result += delta
            seenList.add(nums1[i]-nums2[i]+diff)
        return result
```

### Solution 2:  fenwick tree + binary indexed tree + find count of elements

```py
class FenwickTree:
    def __init__(self, N):
        self.sums = [0 for _ in range(N+1)]

    def update(self, i, delta):

        while i < len(self.sums):
            self.sums[i] += delta
            i += i & (-i)

    def query(self, i):
        res = 0
        while i > 0:
            res += self.sums[i]
            i -= i & (-i)
        return res

    def __repr__(self):
        return f"array: {self.sums}"
    
class Solution:
    def numberOfPairs(self, nums1: List[int], nums2: List[int], diff: int) -> int:
        n = len(nums1)
        minval, maxval = min(nums1+nums2), max(nums1+nums2)
        limit = 2*max(abs(minval), maxval)+1
        fenwick_tree = FenwickTree(2*limit)
        result = 0
        for i, j in enumerate(reversed(range(n))):
            query_val = min(2*limit-1, max(0, nums1[j]-nums2[j]-diff+limit-1))
            countPairs = fenwick_tree.query(query_val)
            delta = i-countPairs
            result += delta
            fenwick_tree.update(nums1[j]-nums2[j]+limit, 1)
        return result
```

## 531. Lonely Pixel I

### Solution 1:

```py
class Solution:
    def findLonelyPixel(self, picture: List[List[str]]) -> int:
        R, C = len(picture), len(picture[0])
        black = 'B'
        rows, cols = [0]*R, [0]*C
        for r, c in product(range(R), range(C)):
            pixel = picture[r][c]
            rows[r] += (pixel==black)
            cols[c] += (pixel==black)
        return sum(1 for r, c in product(range(R), range(C)) if rows[r]==cols[c]==1 and picture[r][c]==black)
```

## 2422. Merge Operations to Turn Array Into a Palindrome

### Solution 1:  greedy + two pointer

```py
class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)
        left, right = 0, n-1
        result = 0
        while left < right:
            if nums[left] == nums[right]: 
                left += 1
                right -= 1
            elif nums[left] < nums[right]:
                result += 1
                left += 1
                nums[left] += nums[left-1]
            else:
                result += 1
                right -= 1
                nums[right] += nums[right+1]
        return result
```

## 2427. Number of Common Factors

### Solution 1: math

```py
class Solution:
    def commonFactors(self, a: int, b: int) -> int:
        if a > b:
            a, b = b, a
        return sum(1 for i in range(1,a+1) if a%i==0 and b%i==0)
```

## 2428. Maximum Sum of an Hourglass

### Solution 1:  matrix

```py
class Solution:
    def maxSum(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        maxsum = 0
        def getHour(r: int, c: int) -> int:
            return grid[r-1][c-1]+grid[r-1][c]+grid[r-1][c+1]+grid[r][c]+grid[r+1][c-1]+grid[r+1][c]+grid[r+1][c+1]
        for r, c in product(range(1,R-1), range(1,C-1)):
            maxsum = max(maxsum, getHour(r,c))
        return maxsum
```

## 2429. Minimize XOR

### Solution 1:  greedy + bit manipulation

```py
class Solution:
    def minimizeXor(self, num1: int, num2: int) -> int:
        num_bits = num2.bit_count()
        result = 0
        for i in reversed(range(31)):
            result<<=1
            if (num1>>i)&1 and num_bits>0:
                result |= 1
                num_bits -= 1
            if num_bits > i:
                result |= 1
                num_bits -= 1
        return result
```

## 2430. Maximum Deletions on a String

### Solution 1: dynamic programming + string slicing + o(n^3) but still passes if you check the worst case when it is all just a single character

```py
class Solution:
    def deleteString(self, s: str) -> int:
        n = len(s)
        if len(set(s)) == 1: return n
        dp = [1]*n
        for i in range(n-2,-1,-1):
            for j in range(1,(n-i)//2+1):
                if s[i:i+j] == s[i+j:i+2*j]:
                    dp[i] = max(dp[i], dp[i+j]+1)
        return dp[0]
```

### Solution 2:  O(n^2) + two dynamic programming + longest common substring

```py
class Solution:
    def deleteString(self, s: str) -> int:
        n = len(s)
        if len(set(s)) == 1: return n
        dp = [1]*n
        lcs = [[0]*(n+1) for _ in range(n+1)]
        for i in range(n-1,-1,-1):
            for j in range(i+1,n):
                if s[i] == s[j]:
                    lcs[i][j] = lcs[i+1][j+1]+1
                if lcs[i][j] >= j-i:
                    dp[i] = max(dp[i], dp[j]+1)
        return dp[0]
```

### Solution 3: z array 

```cpp
class Solution {
public:
    vector<int> z_function(string s) {
    int n = (int) s.length();
    vector<int> z(n);
    for (int i = 1, l = 0, r = 0; i < n; ++i) {
        if (i <= r)
            z[i] = min (r - i + 1, z[i - l]);
        while (i + z[i] < n && s[z[i]] == s[i + z[i]])
            ++z[i];
        if (i + z[i] - 1 > r)
            l = i, r = i + z[i] - 1;
    }
    return z;
    }
    int deleteString(string s) {
        int n = s.size();
        string ns = "";
        vector<int> dp(n,1), z;
        for(int i=n-1;i>=0;i--){
            ns = s[i] + ns;
            z = z_function(ns);
            for(int j=i+1;j<n && ((i + 2*(j-i)) <= n);j++){
                if(z[j-i] >= (j-i))
                    dp[i] = max(dp[i], 1+dp[j]);
            }
        }
        return dp[0];
    }
};
```

## 1578. Minimum Time to Make Rope Colorful

### Solution 1:  greedy + groupby + remove the largest value

```py
class Solution:
    def minCost(self, colors: str, neededTime: List[int]) -> int:
        n = len(colors)
        color_groups = groupby(colors)
        result = 0
        i = 0
        while i < n:
            _, grp = next(color_groups)
            sz = len(list(grp))
            groupSum = groupMaxCost = 0
            for _ in range(sz):
                groupMaxCost = max(groupMaxCost, neededTime[i])
                groupSum += neededTime[i]
                i += 1
            if sz > 1:
                result += groupSum - groupMaxCost
        return result
```

```py
class Solution:
    def minCost(self, colors: str, neededTime: List[int]) -> int:
        ans, n = 0, len(colors)
        for key, grp in groupby(range(n), key = lambda i: colors[i]):
            grp = list(grp)
            if len(grp) == 1: continue
            total = largest = 0
            for i in grp:
                total += neededTime[i]
                largest = max(largest, neededTime[i])
            ans += total - largest
        return ans
```

## 112. Path Sum

### Solution 1:  dfs + stack implementation + iterative implementation + modify tree to keep accumulated sum along each path from root

```py
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if not root: return False
        frontier = [root]
        while frontier:
            node = frontier.pop()
            children = list(filter(None, (node.left, node.right)))
            if len(children) == 0 and node.val == targetSum: return True
            for child_node in children:
                child_node.val += node.val
                frontier.append(child_node)
        return False
```

## 623. Add One Row to Tree

### Solution 1:  bfs + deque + sentinel root node

```py
class Solution:
    def addOneRow(self, root: Optional[TreeNode], val: int, depth: int) -> Optional[TreeNode]:
        sentinel_root = TreeNode(0, root)
        queue = deque([(sentinel_root, 1)])
        while queue:
            node, level = queue.popleft()
            if level == depth:
                node.left, node.right = TreeNode(val, node.left), TreeNode(val, None, node.right)
                continue
            for child_node in filter(None, (node.left, node.right)):
                queue.append((child_node, level+1))
        return sentinel_root.left
```

## 732. My Calendar III

### Solution 1:  boundary count + sort + linear scan

```py
class MyCalendarThree:

    def __init__(self):
        self.delta = Counter()
        self.max_booking = 0
        
    def book(self, start: int, end: int) -> int:
        self.delta[start] += 1
        self.delta[end] -= 1
        active = 0
        for time in sorted(self.delta.keys()):
            active += self.delta[time]
            if active > self.max_booking:
                self.max_booking = active
        return self.max_booking
```

### Solution 2:  lazy segment tree datastructure

```py

```

## 16. 3Sum Closest

### Solution 1:  two pointer approach + reduce variables

```py
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        diff = inf
        n = len(nums)
        nums.sort()
        for i, num in enumerate(nums):
            if i > 0 and nums[i-1] == num: continue
            left, right = i+1, n-1
            while left < right:
                sum_ = num+nums[left]+nums[right]
                if abs(sum_-target) < abs(diff):
                    diff = sum_-target
                if sum_ < target:
                    left += 1
                else:
                    right -= 1
            if diff == 0: break
        return target + diff
```

## 1531. String Compression II

### Solution 1:  recursive dynammic programming

```py
class Solution:
    def getLengthOfOptimalCompression(self, s: str, k: int) -> int:
        n = len(s)
        @cache
        def dfs(index, last_char, last_char_cnt, deleted):
            if deleted > k: return math.inf
            if index == n: return 0
            if s[index] != last_char:
                take = dfs(index + 1, s[index], 1, deleted) + 1
            else:
                take = dfs(index + 1, s[index], last_char_cnt + 1, deleted) + (1 if last_char_cnt in (1, 9, 99) else 0)
            skip = dfs(index + 1, last_char, last_char_cnt, deleted + 1)
            return min(take, skip)
        return dfs(0, "#", 0, 0)
```

## 2440. Create Components With Same Value

### Solution 1:  factorization + math + bfs + topological sort + undirected tree

```py
class Solution:
    def componentValue(self, nums: List[int], edges: List[List[int]]) -> int:
        n = len(nums)
        if n == 1: return 0
        adj_list = [[] for _ in range(n)]
        degrees = [0]*n
        for u, v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
            degrees[u] += 1
            degrees[v] += 1
        # leaf nodes have indegree equal to 1
        default_deque = deque([i for i, d in enumerate(degrees) if d == 1])
        maxNum, sum_ = max(nums), sum(nums)
        def component_values_work(target: int) -> bool:
            values, deg, queue = nums[:], degrees[:], default_deque.copy()
            while queue:
                node = queue.popleft()
                if deg[node] == 0: continue
                deg[node] = 0
                if values[node] == target: # create new component
                    for nei_node in adj_list[node]:
                        if deg[nei_node] == 0: continue
                        deg[nei_node] -= 1
                        if deg[nei_node] == 0:
                            return values[nei_node] == target # parent node is the last node in the tree
                        elif deg[nei_node] == 1:
                            queue.append(nei_node)
                else:
                    for nei_node in adj_list[node]:
                        if deg[nei_node] == 0: continue # must be child node
                        deg[nei_node] -= 1
                        values[nei_node] += values[node]
                        if deg[nei_node] == 0:
                            return values[nei_node] == target
                        elif deg[nei_node] == 1:
                            queue.append(nei_node)
            return False
        for cand in range(maxNum,sum_):
            if sum_%cand==0 and component_values_work(cand):
                num_components = sum_//cand
                return num_components-1
        return 0
```

## 2438. Range Product Queries of Powers

### Solution 1:  prefix product + multiplicative modular inverse + bit manipulation

```py
class Solution:
    def productQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        powers = []
        mod = int(1e9)+7
        for i in range(31):
            if (n>>i)&1:
                powers.append((1<<i))
        prefixProduct = [1]
        for i, p in enumerate(powers):
            prefixProduct.append((prefixProduct[-1]*p)%mod)
        n = len(queries)
        answer = [0]*n
        for i, (left, right) in enumerate(queries):
            resp = (prefixProduct[right+1]*pow(prefixProduct[left], -1, mod))%mod
            answer[i] = resp
        return answer
```

## 2439. Minimize Maximum of Array

### Solution 1:  backwards traversal + prefix sum average

```py
class Solution:
    def minimizeArrayValue(self, nums: List[int]) -> int:
        remaining_sum = sum(nums)
        slots = len(nums)
        maxTarget = -inf
        for i in range(slots-1,0,-1):
            target = remaining_sum//slots
            nums[i-1] += max(0,nums[i]-target)
            nums[i] = min(nums[i], target)
            maxTarget = max(maxTarget, nums[i])
            remaining_sum -= nums[i]
            slots -= 1
        return max(maxTarget, nums[0])
```

```py

```

## 2437. Number of Valid Clock Times

### Solution 1:  regex full match + iterate through all hours and minutes + f string with 0 padding + the '.' means any character in regex

```py
class Solution:
    def countTime(self, time: str) -> int:
        pattern = time.replace('?', '.')
        return sum(re.fullmatch(pattern, f"{hour:02}:{minute:02}") is not None for hour in range(24) for minute in range(60))
```

## 2441. Largest Positive Integer That Exists With Its Negative

### Solution 1:  set + max with default

```py
class Solution:
    def findMaxK(self, nums: List[int]) -> int:
        seen = set(nums)
        return max([abs(num) for num in seen if -num in seen], default=-1)
```

## 2442. Count Number of Distinct Integers After Reverse Operations

### Solution 1:  set 

```py
class Solution:
    def countDistinctIntegers(self, nums: List[int]) -> int:
        s = set(nums)
        for num in nums:
            reversed_num = int(str(num)[::-1])
            s.add(reversed_num)
        return len(s)
```

## 2443. Sum of Number and Its Reverse

### Solution 1:  reverse string + any

```py
class Solution:
    def sumOfNumberAndReverse(self, num: int) -> bool:
        return any(i+int(str(i)[::-1]) == num for i in range(num+1))
```

## 2444. Count Subarrays With Fixed Bounds

### Solution 1:  3 pointers 

```py
class Solution:
    def countSubarrays(self, nums: List[int], minK: int, maxK: int) -> int:
        n = len(nums)
        lmin = lmax = -1
        ans = j = 0
        for i in range(n):
            if nums[i] > maxK or nums[i] < minK:
                j = i + 1
                lmin = lmax = -1
            if nums[i] == minK: lmin = i
            if nums[i] == maxK: lmax = i
            delta = max(0, min(lmin, lmax) - j + 1)
            ans += delta
        return ans
```

## 1335. Minimum Difficulty of a Job Schedule

### Solution 1:  recurisve dynamic programming + precompute the max value in every interval in the array

```py
class Solution:
    def minDifficulty(self, jobDifficulty: List[int], d: int) -> int:
        n = len(jobDifficulty)
        if d > n: return -1
        @cache
        def dfs(index: int, day: int) -> int:
            if index == n and day == 0: return 0
            if index == n and day < 0: return inf
            if day == 0: return inf
            best = inf
            max_diff_job = 0
            for j in range(index, n):
                max_diff_job = max(max_diff_job, jobDifficulty[j])
                best = min(best, max_diff_job + dfs(j+1,day-1))
            return best
        result = dfs(0,d)
        return result
```

### Solution 2:  recursive dynamic programming + binary options, at each step + take minimal value

```py
class Solution:
    def minDifficulty(self, jobDifficulty: List[int], d: int) -> int:
        n = len(jobDifficulty)
        if d > n: return -1
        @cache
        def dfs(index: int, day: int, maxCost: int = -1) -> int:
            if day == 0: return inf
            if index == n:
                return maxCost if day == 1 else inf
            newDayCost = maxCost + dfs(index+1,day-1,jobDifficulty[index]) if maxCost > -1 else inf
            sameDayCost = dfs(index+1,day,max(maxCost, jobDifficulty[index]))
            return min(newDayCost, sameDayCost)
        result = dfs(0,d)
        return result
```

## 800. Similar RGB Color

### Solution 1:  brute force + generator + convert hexadecimal to integer value + max function with custom comparator

```py
class Solution:
    def similarRGB(self, color: str) -> str:
        base = 16
        chars = string.digits + 'abcdef'
        color = color[1:]
        def getHexVal(c: string) -> Iterable[str]:
            for i in range(2,7,2):
                yield int(c[i-2:i], base)
        def similarity_eval(c1: str, c2: str) -> int:
            x1, x2, x3 = getHexVal(c1)
            y1, y2, y3 = getHexVal(c2)
            return -(x1-y1)*(x1-y1)-(x2-y2)*(x2-y2)-(x3-y3)*(x3-y3)
        similar_color = max((x*2+y*2+z*2 for x,y,z in product(chars, repeat=3)), key = lambda c: similarity_eval(color, c))
        return f'#{similar_color}'
```

### Solution 2:  Optimization on brute force + find closest hexadecimal value for each of the 3 segments + no longer 3 nested for loops

```py
class Solution:
    def similarRGB(self, color: str) -> str:
        base = 16
        chars = string.digits + 'abcdef'
        def getClosest(h: str) -> str:
            return min((ch+ch for ch in chars), key = lambda x: abs(int(h, base)-int(x, base)))
        result = [getClosest(color[i-2:i]) for i in range(3,8,2)]
        return '#' + ''.join(result)
```

## 2432. The Employee That Worked on the Longest Task

### Solution 1:  max

```py
class Solution:
    def hardestWorker(self, n: int, logs: List[List[int]]) -> int:
        longest_time = -inf
        longest_time_id = inf
        current_time = 0
        for id, leave_time in logs:
            delta_time = leave_time - current_time
            if delta_time > longest_time or delta_time==longest_time and id < longest_time_id:
                longest_time = delta_time
                longest_time_id = id
            current_time = leave_time
        return longest_time_id
```

## 2433. Find The Original Array of Prefix Xor

### Solution 1:  bit manipulation + math properties of xor + a = b^c, c = a^b, b = a^b

```py
class Solution:
    def findArray(self, pref: List[int]) -> List[int]:
        n = len(pref)
        arr = [0]*n
        prefixXor = pref[0]
        arr[0] = prefixXor
        for i in range(1,n):
            arr[i] = prefixXor^pref[i]
            prefixXor ^= arr[i]
        return arr
```

```py
class Solution:
    def findArray(self, pref: List[int]) -> List[int]:
        n = len(pref)
        arr = [0] * n
        for i in range(n):
            if i > 0:
                arr[i] = pref[i - 1]
            arr[i] ^= pref[i]
        return arr
```

## 2434. Using a Robot to Print the Lexicographically Smallest String

### Solution 1:  suffix min array + stack + greedy

```py
class Solution:
    def robotWithString(self, s: str) -> str:
        n = len(s)
        s = list(s)
        suffixMin = ['{']*(n+1)
        for i in reversed(range(n)):
            suffixMin[i] = min(suffixMin[i+1], s[i])
        p, t = [], []
        left = 0
        for i in range(1, n+1):
            if suffixMin[i] > suffixMin[i-1] or (s[i-1] == suffixMin[i-1] and s[i] > suffixMin[i]):
                t.extend(s[left:i])
                left = i
            while t and t[-1] <= suffixMin[i]:
                p.append(t.pop())
        return ''.join(p)
```

## 2435. Paths in Matrix Whose Sum Is Divisible by K

### Solution 1:  dynamic programming + state is (row, col, remainder) + memo with 3 dimensional list

```py
class Solution:
    def numberOfPaths(self, grid: List[List[int]], k: int) -> int:
        R, C = len(grid), len(grid[0])
        memo = [[[0]*k for _ in range(C)] for _ in range(R)]
        memo[0][0][grid[0][0]%k] = 1
        mod = int(1e9)+7
        for r, c in product(range(R),range(C)):
                if r == c == 0: continue
                for i in range(k):
                    if c > 0:
                        left_count = memo[r][c-1][i]
                        left_val = (grid[r][c] + i)%k
                        memo[r][c][left_val] = (memo[r][c][left_val]+left_count)%mod
                    if r > 0:
                        above_count = memo[r-1][c][i]
                        above_val = (grid[r][c] + i)%k
                        memo[r][c][above_val] = (memo[r][c][above_val]+above_count)%mod
        return memo[-1][-1][0] # number of paths to reach the bottom right corner cell with remainder of 0, divisible by k
```

### Solution 2: space optimized + state optimized

```py

```

## 2431. Maximize Total Tastiness of Purchased Fruits

### Solution 1:  recursive dynamic programming + state (index, amount_remain, coupons_remain) + maximize on tastiness + 3 choices at each state

```py
class Solution:
    def maxTastiness(self, price: List[int], tastiness: List[int], maxAmount: int, maxCoupons: int) -> int:
        @cache
        def dfs(index: int, amt: int, coup: int) -> int:
            if amt < 0: return -inf
            if index == len(price): return 0 
            buyFruitCoupon = dfs(index+1,amt-(price[index]//2),coup-1)+tastiness[index] if coup > 0 else -inf
            buyFruit = dfs(index+1,amt-price[index],coup)+tastiness[index]
            skipFruit = dfs(index+1,amt,coup)
            return max(buyFruitCoupon, buyFruit, skipFruit)
        return dfs(0,maxAmount,maxCoupons)
```

## 1832. Check if the Sentence Is Pangram

### Solution 1:  set + string module

```py
class Solution:
    def checkIfPangram(self, sentence: str) -> bool:
        return len(set(sentence)) == len(string.ascii_lowercase)
```

## 2436. Minimum Split Into Subarrays With GCD Greater Than One

### Solution 1:  math + gcd + sliding window

```py
class Solution:
    def minimumSplits(self, nums: List[int]) -> int:
        neutral = 0
        cur_gcd = neutral
        numSplits = neutral
        for num in nums:
            cur_gcd = gcd(cur_gcd, num)
            if cur_gcd == 1:
                cur_gcd = num
                numSplits += 1
        return numSplits + 1
```

## 76. Minimum Window Substring

### Solution 1:  sliding window with counter for frequency

```py
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        freq = Counter(t)
        unmatched = len(t)
        n = len(s)
        ans = "A" * (n + 1)
        l = 0
        for r in range(n):
            if freq[s[r]] > 0: unmatched -= 1
            freq[s[r]] -= 1
            while unmatched == 0:
                freq[s[l]] += 1
                if freq[s[l]] > 0: unmatched += 1
                if unmatched == 1 and r - l + 1 < len(ans): ans = s[l : r + 1]
                l += 1
        return ans if len(ans) <= n else ""
```

## 645. Set Mismatch

### Solution 1:  bit manipulation + xor + in place modification of array to find duplicate element

Quick explanation on this one. 

Since you are considering integers from 1 to n, to find the missing integer, you just need to xor everything with 1 to n and the elements in array, cause all will cancel because you are xoring every value that exists in array twice.  Except for the duplicated element, which would be xored 3 times, so you need to xor that one more time.  The only value that is xored once is the missing integer.  

You can find the duplicate integer, by storing negative values at index that corresponds with the integer. Cause the only time you should see a negative at that index already means you already set it to negative, and thus this is the duplicate element. 

```py
class Solution:
    def findErrorNums(self, nums: List[int]) -> List[int]:
        n = len(nums)
        miss = dupe = 0
        for i in range(n):
            num = abs(nums[i])
            miss ^= (i + 1) ^ num
            if nums[num - 1] < 0: dupe = num
            nums[num - 1] *= -1
        return [dupe, miss ^ dupe]
```

## 217. Contains Duplicate

### Solution 1:  set

```py
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        seen = set()
        for num in nums:
            if num in seen: return True
            seen.add(num)
        return False
```

### Solution 2: sort + bit manipulation + inverse of xor 

```py
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        nums.sort()
        for i in range(1,len(nums)):
            if (nums[i]^nums[i-1])==0: return True
        return False
```

## 2445. Number of Nodes With Value One

### Solution 1:  counter + bit manipulation + loop through and find parent nodes and what was the value of parents

```py
class Solution:
    def numberOfNodes(self, n: int, queries: List[int]) -> int:
        queries_counter = Counter()
        for qu in queries:
            queries_counter[qu] ^= 1
        res = 0
        value_nodes = Counter()
        for node in range(1, n+1):
            parent_node = node//2
            val_parent = value_nodes[parent_node]
            if queries_counter[node]:
                val_parent ^= 1
            res += val_parent
            value_nodes[node] = val_parent
        return res
```

### Solution 1:  prefix xor + bit manipulation

```py
class Solution:
    def numberOfNodes(self, n: int, queries: List[int]) -> int:
        prefix_flips = [0]*(n+1)
        for i in queries:
            prefix_flips[i] ^= 1
        for i in range(1,n+1):
            parent = i//2
            prefix_flips[i] = prefix_flips[parent]^prefix_flips[i]
        return sum(prefix_flips)
```

## 2446. Determine if Two Events Have Conflict

### Solution 1:  intersection for two intervals

```py
class Solution:
    def haveConflict(self, event1: List[str], event2: List[str]) -> bool:
        def get_minutes(event):
            ev = event.split(':')
            return int(ev[0])*60+int(ev[1])
        s1, e1, s2, e2 = get_minutes(event1[0]), get_minutes(event1[1]), get_minutes(event2[0]), get_minutes(event2[1])
        return min(e1,e2) >= max(s1,s2)
```

## 2447. Number of Subarrays With GCD Equal to K

### Solution 1:  brute force solution 

```py
class Solution:
    def subarrayGCD(self, nums: List[int], k: int) -> int:
        n = len(nums)
        res = 0
        for i in range(n):
            g = 0
            for j in range(i,n):
                g = gcd(g, nums[j])
                res += (g==k)
                if g < k:
                    break
        return res
```

### Solution 2:  Count greatest common denominators

```py
class Solution:
    def subarrayGCD(self, nums: List[int], k: int) -> int:
        gcds = Counter()
        res = 0
        for num in nums:
            next_gcds = Counter()
            if num%k==0:
                gcds[num] += 1
            for prev_gcd, cnt in gcds.items():
                next_gcds[gcd(prev_gcd,num)] += cnt
            gcds = next_gcds
            res += gcds[k]
        return res
```

## 2448. Minimum Cost to Make Array Equal

### Solution 1:  prefix sum + suffix sum + sort + counter

```py
class Solution:
    def minCost(self, nums: List[int], cost: List[int]) -> int:
        unique_nums = sorted(list(set(nums)))
        cost_cnt = Counter()
        for i, num in enumerate(nums):
            cost_cnt[num] += cost[i]
        mincost = inf
        prefixCost = suffixCost = initial_cost = prev_num = 0
        for num, val in cost_cnt.items():
            initial_cost += num*val
            suffixCost += val
        mincost = initial_cost
        for val in unique_nums:
            delta = val - prev_num
            initial_cost = (initial_cost + delta*prefixCost - delta*suffixCost)
            mincost = min(mincost, initial_cost)
            prefixCost += cost_cnt[val]
            suffixCost -= cost_cnt[val]
            prev_num = val
            if prefixCost > suffixCost: break # pass the global maximum value, on the downward slope
        return mincost
```

### Solution 2:  zip + sort + prefix sum + suffix sum

```py
class Solution:
    def minCost(self, nums: List[int], cost: List[int]) -> int:
        mincost = inf
        prefixCost = suffixCost = totalCost = prev_num = 0
        for num, val in zip(nums, cost):
            totalCost += num*val
            suffixCost += val
        for n, c in sorted(zip(nums, cost)):
            delta = n - prev_num
            totalCost = (totalCost + delta*prefixCost - delta*suffixCost)
            prefixCost += c
            suffixCost -= c
            mincost = min(mincost, totalCost)
            if prefixCost > suffixCost: break
            prev_num = n
        return mincost
```

### Solution 3:  binary search + quadratic function + positive and negative slope

```py
class Solution:
    def minCost(self, nums: List[int], cost: List[int]) -> int:
        mincost = inf
        def getCost(target: int) -> int:
            return sum([c*abs(target-x) for x, c in zip(nums, cost)])
        left, right = min(nums), max(nums)
        while left < right:
            mid = (left+right)>>1
            if getCost(mid) < getCost(mid+1):
                right = mid
            else:
                left = mid+1
        return getCost(left)
```

```py
class Solution:
    def minCost(self, nums: List[int], cost: List[int]) -> int:
        cost_counter = Counter()
        for x, y in zip(nums, cost):
            cost_counter[x] += y
        prev = min(nums)
        prefix, suffix = 0, sum((x - prev) * y for x, y in cost_counter.items())
        prefix_delta, suffix_delta = 0, sum(cost)
        res = prefix + suffix
        for num in sorted(cost_counter.keys()):
            delta = num - prev
            prefix += delta * prefix_delta
            suffix -= delta * suffix_delta
            prefix_delta += cost_counter[num]
            suffix_delta -= cost_counter[num]
            prev = num
            res = min(res, prefix + suffix)
        return res
```

## 2449. Minimum Number of Operations to Make Arrays Similar

### Solution 1:  greedy + split into odds and evens, since you are increment/decrement by 2 + sort

```py
class Solution:
    def makeSimilar(self, nums: List[int], target: List[int]) -> int:
        odd_nums, even_nums = sorted([x for x in nums if x&1]), sorted([x for x in nums if x%2==0])
        odd_tar, even_tar = sorted([x for x in target if x&1]), sorted([x for x in target if x%2==0])
        return (sum(x-y for x,y in zip(odd_nums, odd_tar) if x>y) + sum(x-y for x,y in zip(even_nums, even_tar) if x>y))//2
```

## 487. Max Consecutive Ones II

### Solution 1:  sliding window with two variables + track count of consecutive ones to the left and right from 0 as pivot point

```py
class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        left_count = right_count = result = 0
        for num in nums:
            right_count += 1
            if num == 0:
                left_count, right_count = right_count, 0
            result = max(result, left_count + right_count)
        return result
```

## 2450. Number of Distinct Binary Strings After Applying Operations

### Solution 1:  greedy + math + result is independent of s characters + depends on length of s and k + can either flip or not flip at each index + binary tree + 2^(number of flips)

```py
class Solution:
    def countDistinctStrings(self, s: str, k: int) -> int:
        n = len(s)
        mod = int(1e9) + 7
        return (2**(n-k+1))%mod
```

## 2451. Odd String Difference

### Solution 1:  dictionary + tuple

```py
class Solution:
    def oddString(self, words: List[str]) -> str:
        n = len(words[0])
        diffArray = [[0]*(n-1) for _ in range(len(words))]
        for i, word in enumerate(words):
            for j in range(1,n):
                diffArray[i][j-1] = ord(word[j])-ord(word[j-1])
        diffDict = Counter()
        dd = {}
        for i in range(len(words)):
            key = tuple(diffArray[i])
            diffDict[key] += 1
            dd[key] = words[i]
        for key, cnt in diffDict.items():
            if cnt == 1:
                return dd[key]
        return ''
```

## 2452. Words Within Two Edits of Dictionary

### Solution 1:  levenshtein distance + dictionary + filter

```py
class Solution:
    def twoEditWords(self, queries: List[str], dictionary: List[str]) -> List[str]:
        minEditDistance = defaultdict(lambda: inf)
        def edit_distance(w1, w2):
            return sum(c1 != c2 for c1, c2 in zip(w1, w2))
        for word in queries:
            for dword in dictionary:
                dist = edit_distance(word, dword)
                minEditDistance[word] = min(minEditDistance[word], dist)
        return filter(lambda word: minEditDistance[word] <= 2, queries)
```
 
## 2453. Destroy Sequential Targets

### Solution 1:  bucket + place each integer into correct bucket it belongs to

```py
class Solution:
    def destroyTargets(self, nums: List[int], space: int) -> int:
        bucket = Counter()
        minInt = defaultdict(lambda: inf)
        for num in map(lambda x: x-1, nums):
            bucket[num%space] += 1
            minInt[num%space] = min(minInt[num%space], num + 1)
        maxDestroy, minIndex = 0, inf
        for key, cnt in bucket.items():
            if cnt > maxDestroy or (cnt == maxDestroy and minInt[key] < minIndex):
                maxDestroy = cnt
                minIndex = minInt[key]
        return minIndex
```

## 2454. Next Greater Element IV

### Solution 1:  sortedlist + greedy + binary search + set

```py
from sortedcontainers import SortedList
class Solution:
    def secondGreaterElement(self, nums: List[int]) -> List[int]:
        sl = SortedList()
        seen = set()
        n = len(nums)
        default = -1
        answer = [default]*n
        for i, num in  enumerate(nums):
            j = sl.bisect_left((num,)) - 1
            while j >= 0:
                index = sl[j][1]
                if index in seen:
                    answer[index] = num
                    sl.pop(j)
                seen.add(index)
                j -= 1
            sl.add((num, i))
        return answer
```

### Solution 2:  two stacks + temporary list 

```py
class Solution:
    def secondGreaterElement(self, nums: List[int]) -> List[int]:
        n = len(nums)
        stack, first, second = [], [], [-1]*n
        for i, num in enumerate(nums):
            while first and nums[first[-1]] < num:
                j = first.pop()
                second[j] = num
            temp = []
            while stack and nums[stack[-1]] < num:
                j = stack.pop()
                temp.append(j)
            stack.append(i)
            first.extend(temp[::-1])
        return second
```

## 1293. Shortest Path in a Grid with Obstacles Elimination

### Solution 1:  best first search/informed search/A-star Search + memoization with set + estimated cost function is manhattan distance from target

```py
class Solution:
    def shortestPath(self, grid: List[List[int]], k: int) -> int:
        R, C = len(grid), len(grid[0])
        obstacle = 1
        state = (0, 0, 0)
        visited = set()
        neighbors = lambda r, c: ((r+1,c),(r-1,c),(r,c+1),(r,c-1))
        in_bounds = lambda r, c: 0 <= r < R and 0 <= c < C
        manhattan_distance = lambda r, c: R - 1 - r + C - 1 - c
        minheap = [(manhattan_distance(0, 0), 0, state)]
        while minheap:
            estimated_cost, steps, (r, c, removals) = heappop(minheap)
            if r == R-1 and c == C-1: return steps
            for nr, nc in neighbors(r, c):
                if not in_bounds(nr, nc): continue
                nremovals = removals + grid[nr][nc]
                nstate = (nr, nc, nremovals)
                if nremovals > k or nstate in visited: continue
                nsteps = steps + 1
                ncost = nsteps + manhattan_distance(nr, nc)
                heappush(minheap, (ncost, nsteps, nstate))
                visited.add(nstate)
        return -1
```

## 433. Minimum Genetic Mutation

### Solution 1:  queue + bfs + hamming distance

```py
class Solution:
    def minMutation(self, start: str, end: str, bank: List[str]) -> int:
        queue = deque([start])
        visited = set([start])
        mutations = 0
        def get_neighbors(gene: str) -> Iterable[str]:
            hamming_distance = lambda g1, g2: sum(1 for x, y in zip(g1, g2) if x != y)
            for nei_gene in bank:
                if nei_gene in visited or hamming_distance(gene, nei_gene) != 1: continue
                yield nei_gene
        while queue:
            sz = len(queue)
            for _ in range(sz):
                gene = queue.popleft()
                if gene == end: return mutations
                for neighbor in get_neighbors(gene):
                    visited.add(neighbor)
                    queue.append(neighbor)
            mutations += 1
        return -1
                
```

## 1198. Find Smallest Common Element in All Rows

### Solution 1:  counter + first element to count equal to number of rows

```py
class Solution:
    def smallestCommonElement(self, mat: List[List[int]]) -> int:
        cnt = Counter()
        R, C = len(mat), len(mat[0])
        for r, c in product(range(R), range(C)):
            v = mat[r][c]
            cnt[v] += 1
            if cnt[v] == R: return v
        return -1
```

## 1706. Where Will the Ball Fall

### Solution 1:  simulation

```py
class Solution:
    def findBall(self, grid: List[List[int]]) -> List[int]:
        R, C = len(grid), len(grid[0])
        ans = [-1]*C
        is_vShaped = lambda r, c: (c < C-1 and grid[r][c] == 1 and grid[r][c+1] == -1) or (c > 0 and grid[r][c] == -1 and grid[r][c-1] == 1)
        in_bounds = lambda c: 0 <= c < C
        for i in range(C):
            hor_pos = i
            reach_bottom = True
            for vert_pos in range(R):
                board = grid[vert_pos][hor_pos]
                if is_vShaped(vert_pos, hor_pos): 
                    reach_bottom = False
                    break
                hor_pos += (board == 1)
                hor_pos -= (board == -1)
                if not in_bounds(hor_pos): 
                    reach_bottom = False
                    break
            if reach_bottom:
                ans[i] = hor_pos
        return ans
```

## 70. Climbing Stairs

### Solution 1: iterative dp + space optimized + depends on previous two states + fibonacci sequence

```py
class Solution:
    def climbStairs(self, n: int) -> int:
        cur, prev = 1, 0
        for _ in range(n):
            cur, prev = cur + prev, cur
        return cur
```

## 1137. N-th Tribonacci Number

### Solution 1:  memory optimized dynamic programming

```py
class Solution:
    def tribonacci(self, n: int) -> int:
        t0, t1, t2 = 0, 1, 1
        if n == 0: return t0
        for _ in range(n-2):
            t2, t1, t0 = t0 + t1 + t2, t2, t1
        return t2
```

### Solution 2:  matrix exponentiation + linear algebra

```py
import numpy as np
class Solution:
    def tribonacci(self, n: int) -> int:
        transition_matrix = np.array([[1,1,1],[1,0,0],[0,1,0]])
        result = np.identity(3, dtype=np.int64)
        while n > 0:
            if n&1:
                result  = np.matmul(result, transition_matrix)
            transition_matrix = np.matmul(transition_matrix, transition_matrix)
            n >>= 1
        return result[0][-1]
```

## 212. Word Search II

### Solution 1:  trie data structure + backtracking recursion

```py
class TrieNode:
    def __init__(self, word: str = '$'):
        self.children = defaultdict(TrieNode)
        self.word = word
    def __repr__(self):
        return f'word: {self.word}, children: {self.children}'
class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        visited = '$'
        words = set(words)
        trie = TrieNode()
        # CONSTRUCT TRIE TREE
        for word in words:
            node = trie
            for ch in word:
                node = node.children[ch]
            node.word = word
        R, C = len(board), len(board[0])
        self.result = set()      
        in_bounds = lambda r, c: 0 <= r < R and 0 <= c < C
        neighborhood = lambda r, c: [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]
        # BACKTRACK IN BOARD
        def backtrack(r, c):
            # GET CURRENT CHARACTER
            unvisited = board[r][c]
            # CHECK IF THERE IS NO PREFIX MATCHING IN THE TRIE
            if unvisited not in self.node.children: return # early termination when there is no word with current prefix
            prev_node = self.node
            self.node = self.node.children[unvisited]
            # IF THIS PREFIX MATCHES A WORD
            if self.node.word != visited: 
                self.result.add(self.node.word)
            # MARK AS VISITED SO THIS PATH DOESN'T REUSE THE SAME CELL
            board[r][c] = visited
            # FIND THE NEIGHBORS TO CONTINUE PATH IF IT IS NOT VISITED IN THE CURRENT PATH
            for nr, nc in neighborhood(r, c):
                if not in_bounds(nr, nc) or board[nr][nc] == visited: continue
                backtrack(nr, nc)
            # UNDO CHANGES, SO MARK THIS CELL AS UNVISITED SO THAT ANOTHER PATH CAN VISIT IT
            board[r][c] = unvisited
            # IF AT A TERMINAL NODE IN TRIE, REMOVE THE NODE TO PREVENT REVISITING A COMPLETELY MATCHED PREFIX
            if len(self.node.children) == 0:
                prev_node.children.pop(unvisited)
            self.node = prev_node
        for r, c in product(range(R), range(C)):
            self.node = trie # initialize node to the root of trie tree
            backtrack(r,c)
        return self.result
```

### Solution 2:  trie node inheriting from defaultdict class

```py
class TrieNode(defaultdict):
    def __init__(self):
        super().__init__(TrieNode)
        self.word = '$'
class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        visited = '$'
        words = set(words)
        trie = TrieNode()
        # CONSTRUCT TRIE TREE
        for word in words:
            node = trie
            for ch in word:
                node = node[ch]
            node.word = word
        R, C = len(board), len(board[0])
        self.result = set()      
        in_bounds = lambda r, c: 0 <= r < R and 0 <= c < C
        neighborhood = lambda r, c: [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]
        # BACKTRACK IN BOARD
        def backtrack(r, c):
            # GET CURRENT CHARACTER
            unvisited = board[r][c]
            # CHECK IF THERE IS NO PREFIX MATCHING IN THE TRIE
            if unvisited not in self.node: return # early termination when there is no word with current prefix
            prev_node = self.node
            self.node = self.node[unvisited]
            # IF THIS PREFIX MATCHES A WORD
            if self.node.word != visited: 
                self.result.add(self.node.word)
            # MARK AS VISITED SO THIS PATH DOESN'T REUSE THE SAME CELL
            board[r][c] = visited
            # FIND THE NEIGHBORS TO CONTINUE PATH IF IT IS NOT VISITED IN THE CURRENT PATH
            for nr, nc in neighborhood(r, c):
                if not in_bounds(nr, nc) or board[nr][nc] == visited: continue
                backtrack(nr, nc)
            # UNDO CHANGES, SO MARK THIS CELL AS UNVISITED SO THAT ANOTHER PATH CAN VISIT IT
            board[r][c] = unvisited
            # IF AT A TERMINAL NODE IN TRIE, REMOVE THE NODE TO PREVENT REVISITING A COMPLETELY MATCHED PREFIX
            if len(self.node) == 0:
                prev_node.pop(unvisited)
            self.node = prev_node
        for r, c in product(range(R), range(C)):
            self.node = trie # initialize node to the root of trie tree
            backtrack(r,c)
        return self.result
```

## 45. Jump Game II

### Solution 1:  greedy + dynamic programming with constant space + always keep track of farthest can jump for the next jump while working way to need to jump. 

```py
class Solution:
    def jump(self, nums: List[int]) -> int:
        max_reach = jumps = reach = 0
        n = len(nums)
        for i in range(n - 1):
            max_reach = max(max_reach, i + nums[i])
            if i == reach: # MUST JUMP NOW OR SOME TIME BEFORE
                jumps += 1
                reach = max_reach
        return jumps
```

## 918. Maximum Sum Circular Subarray

### Solution 1:  max sum subarray + prefix and suffix max

```py
class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        n = len(nums)
        prefixMax, suffixMax = [-inf], [-inf]
        prefixSum = suffixSum = 0
        maxSum = sum_ = -inf
        for i in range(n):
            prefixSum += nums[i]
            suffixSum += nums[n-i-1]
            sum_ = max(nums[i], sum_ + nums[i])
            maxSum = max(maxSum, sum_)
            prefixMax.append(max(prefixMax[-1], prefixSum))
            suffixMax.append(max(suffixMax[-1], suffixSum))
        return max(maxSum, max((pre+suf for pre, suf in zip(prefixMax, reversed(suffixMax)))))
```

### Solution 2:  max sum subarray + nonempty min sum subarray (max sum subarray forming prefix and suffix is total_sum - min_sum) + find minimum sum subarray to remove

```py
class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        n = len(nums)
        maxSum = sum_ = -inf
        total = 0
        for num in nums:
            sum_ = max(num, sum_ + num)
            maxSum = max(maxSum, sum_)
            total += num
        minSum = sum_ = inf
        for num in nums[1:]:
            sum_ = min(num, sum_ + num)
            minSum = min(minSum, sum_)
        maxSum = max(maxSum, total - minSum)
        minSum = sum_ = inf
        for num in nums[:-1]:
            sum_ = min(num, sum_ + num)
            minSum = min(minSum, sum_)
        maxSum = max(maxSum, total - minSum)
        return maxSum
```

```py
class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        max_sum, min_sum, cur_min, cur_max, sum_ = -math.inf, math.inf, 0, 0, 0
        for num in nums:
            cur_min = min(cur_min + num, num)
            cur_max = max(cur_max + num, num)
            max_sum = max(max_sum, cur_max)
            min_sum = min(min_sum, cur_min)
            sum_ += num
        return max_sum if min_sum == sum_ else max(max_sum, sum_ - min_sum)
```

## 899. Orderly Queue

### Solution 1:  suffix array + lexicographically smallest suffix with string s+s + make that suffix the prefix for the result

```py

class Solution:
    def radix_sort(self, p: List[int], c: List[int]) -> List[int]:
        n = len(p)
        cnt = [0]*n
        next_p = [0]*n
        for cls_ in c:
            cnt[cls_] += 1
        pos = [0]*n
        for i in range(1,n):
            pos[i] = pos[i-1] + cnt[i-1]
        for pi in p:
            cls_i = c[pi]
            next_p[pos[cls_i]] = pi
            pos[cls_i] += 1
        return next_p

    def suffix_array(self, s: str) -> str:
        n = len(s)
        p, c = [0]*n, [0]*n
        arr = [None]*n
        for i, ch in enumerate(s):
            arr[i] = (ch, i)
        arr.sort()
        for i, (_, j) in enumerate(arr):
            p[i] = j
        c[p[0]] = 0
        for i in range(1,n):
            c[p[i]] = c[p[i-1]] + (arr[i][0] != arr[i-1][0])
        k = 1
        is_finished = False
        while k < n and not is_finished:
            for i in range(n):
                p[i] = (p[i] - k + n)%n
            p = self.radix_sort(p, c)
            next_c = [0]*n
            next_c[p[0]] = 0
            is_finished = True
            for i in range(1,n):
                prev_segments = (c[p[i-1]], c[(p[i-1]+k)%n])
                current_segments = (c[p[i]], c[(p[i]+k)%n])
                next_c[p[i]] = next_c[p[i-1]] + (prev_segments != current_segments)
                is_finished &= (next_c[p[i]] != next_c[p[i-1]])
            k <<= 1
            c = next_c
        return p[0]
    
    def orderlyQueue(self, s: str, k: int) -> str:
        if k > 1: 
            counts = Counter(s)
            return ''.join([char*counts[char] for char in string.ascii_lowercase])
        n = len(s)
        # suffix index for the lexicographically smallest suffix in the string s
        suffix_index = self.suffix_array(s+s)%n
        return s[suffix_index:] + s[:suffix_index]
```

### Solution 2: brute force + find the minimum of every possible rotated string + optimized sort of limited character search space by using bucket sort = O(n+k) + O(n^2)

```py
class Solution:
    def orderlyQueue(self, s: str, k: int) -> str:
        if k > 1: 
            counts = Counter(s)
            return ''.join([char*counts[char] for char in string.ascii_lowercase])
        result = s
        n = len(s)
        for i in range(n):
            cand_s = s[i:] + s[:i]
            result = min(cand_s, result)
        return result
```

### Solution 3:  Tournament Algorithm + Dual Elimination + Finds the champion of the tournament + O(nlogn)

```py
class Solution:
    def orderlyQueue(self, s: str, k: int) -> str:
        if k > 1: 
            counts = Counter(s)
            return ''.join([char*counts[char] for char in string.ascii_lowercase])
        min_char = min(s)
        s_len = len(s)
        advance = lambda index: (index + 1)%s_len
        champions = deque()
        for i, ch in enumerate(s):
            if ch == min_char:
                champions.append(i)
        # DUAL ELIMINATION UNTIL ONE CHAMPION REMAINS
        while len(champions) > 1:
            champion1 = champions.popleft()
            champion2 = champions.popleft()
            # ASSUME CHAMPION1 IS SMALLER INDEX
            if champion2 < champion1:
                champion1, champion2 = champion2, champion1
            # length of substring for champions is champion2-champion1
            # abcdefg
            # ^  ^
            # substring should be abc for champion 1, and def for champion 2
            current_champion = champion1
            left_champion, right_champion = champion1, champion2
            for _ in range(champion2 - champion1):
                if s[left_champion] < s[right_champion]: break
                if s[left_champion] > s[right_champion]:
                    current_champion = champion2
                    break
                left_champion = advance(left_champion)
                right_champion = advance(right_champion)
            champions.append(current_champion)
        champion_index = champions.pop()
        return s[champion_index:] + s[:champion_index]
```

## 1323. Maximum 69 Number

### Solution 1:  find first element equal to 6 in string representation of number + linear in terms of the number of digits in the number

```py
class Solution:
    def maximum69Number (self, num: int) -> int:
        snum = str(num)
        for i in range(len(snum)):
            if snum[i] == '6':
                return int(snum[:i]+'9'+snum[i+1:])
        return num
```

### Solution 2: string replace with occurrence of 1

```py
class Solution:
    def maximum69Number (self, num: int) -> int:
        snum = str(num)
        return int(snum.replace('6', '9', 1))
```

## 2460. Apply Operations to an Array

### Solution 1:  two pointers

```py
class Solution:
    def applyOperations(self, A: List[int]) -> List[int]:
        nonzero_front = 0
        for i in range(len(A)):
            if i < len(A) - 1 and A[i] == A[i+1]:
                A[i] *= 2
                A[i+1] = 0
            if A[i]:
                A[i], A[nonzero_front] = A[nonzero_front], A[i]
                nonzero_front += 1
        return A
```

## 2461. Maximum Sum of Distinct Subarrays With Length K

### Solution 1:  keep pointer to the last duplicat integer + store position of last index of each integer + sliding window of fixed size

```py
class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        m = max(nums) + 1
        pos = [-1]*m
        n = len(nums)
        cur = result = 0
        last_dupe = -1
        for i in range(n):
            cur += nums[i]
            if i>=k:    cur -= nums[i-k]
            if i-pos[nums[i]] < k:  last_dupe = max(last_dupe, pos[nums[i]])
            if i-last_dupe >= k:    result = max(result, cur)
            pos[nums[i]] = i
        return result
```

## 2462. Total Cost to Hire K Workers

### Solution 1:  minheap + deque

```py
class Solution:
    def totalCost(self, costs: List[int], k: int, candidates: int) -> int:
        minheap = []
        total_cost = 0
        first, last = 0, 1
        costs = deque([(cost, index) for index, cost in enumerate(costs)])
        for _ in range(candidates):
            if not costs: break
            cost, index = costs.popleft()
            heappush(minheap, (cost, index, first))
            if not costs: break
            cost, index = costs.pop()
            heappush(minheap, (cost, index, last))
        for _ in range(k):
            cost, index, pos = heappop(minheap)
            total_cost += cost
            if not costs: continue
            if pos == first:
                cost, index = costs.popleft()
                heappush(minheap, (cost, index, first))
            else:
                cost, index = costs.pop()
                heappush(minheap, (cost, index, last))
        return total_cost
```

## 2463. Minimum Total Distance Traveled

### Solution 1:  recursive dynamic programming + two states, repair robot at current factory or skip + O(nmk)

```py
class Solution:
    def minimumTotalDistance(self, robot: List[int], factory: List[List[int]]) -> int:
        robot.sort()
        factory.sort()
        @cache
        def dp(i: int, j: int, k: int) -> int:
            if i == len(robot): return 0
            if j == len(factory): return inf
            # repair robot at current factory
            current = dp(i+1, j, k-1) + abs(robot[i] - factory[j][0]) if k > 0 else inf
            # skip repair robot at current factory
            if j < len(factory)-1:
                current = min(current, dp(i, j+1, factory[j+1][1]))
            return current
        return dp(0,0,factory[0][1])
```

### Solution 2:  iterative dynamic programming + flatten factories + find all robots that can be placed at current factory and find the cheapest cost to place the robot at that location.

```py
class Solution:
    def minimumTotalDistance(self, rob, factory):
        rob.sort()
        factory.sort()
        fac = [f for f, lim in factory for _ in range(lim)]
        n, m = len(rob), len(fac)
        dp = [0] + [inf]*n
        for j, f in enumerate(fac):
            left, right = max(n - (m-j), 0), min(j, n-1)
            for i in reversed(range(left, right+1)):
                dp[i+1] = min(abs(f-rob[i])+dp[i], dp[i+1])
        return dp[-1]
```

## 1047. Remove All Adjacent Duplicates In String

### Solution 1:  stack

```py
class Solution:
    def removeDuplicates(self, s: str) -> str:
        stack = []
        for ch in s:
            if stack and stack[-1] == ch:
                stack.pop()
            else:
                stack.append(ch)
        return ''.join(stack)
```

## 339. Nested List Weight Sum

### Solution 1:  sum + recursion

```py
class Solution:
    def depthSum(self, nestedList: List[NestedInteger], depth: int = 1) -> int:
        return sum([depth*nestInt.getInteger() if nestInt.isInteger() else self.depthSum(nestInt.getList(), depth+1) for nestInt in nestedList])
```

## 901. Online Stock Span

### Solution 1:  stack

```py
class StockSpanner:

    def __init__(self):
        self.stack = [(inf, 1)]

    def next(self, price: int) -> int:
        span = 1
        while self.stack[-1][0] <= price:
            _, prev_span = self.stack.pop()
            span += prev_span
        self.stack.append((price, span))
        return span
```

## 1014. Best Sightseeing Pair

### Solution 1:  iterative dp + prefix max

```py
class Solution:
    def maxScoreSightseeingPair(self, values: List[int]) -> int:
        prefixMax = values[0]
        maxScore = -inf
        for j, val in enumerate(values[1:], start = 1):
            score = prefixMax + val - j
            maxScore = max(maxScore, score)
            prefixMax = max(prefixMax, val + j)
        return maxScore
```

## 122. Best Time to Buy and Sell Stock II

### Solution 1:  recursive dp

```py
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        buy, sell = 0, 1
        @cache
        def stock(i, state):
            if i == len(prices): return 0
            # action
            transaction = stock(i+1, state^1) + (prices[i] if state == sell else -prices[i])
            # nothin
            skip = stock(i + 1, state)
            return max(transaction, skip)
        return stock(0, buy)
```

## 309. Best Time to Buy and Sell Stock with Cooldown

### Solution 1:  recursive dp

```py
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        buy, sell = 0, 1
        @cache
        def stock(i, state):
            if i >= len(prices): return 0
            if state == buy:
                transaction = stock(i+1, state^1) - prices[i]
            else:
                transaction = stock(i+2, state^1) + prices[i]
            skip = stock(i + 1, state)
            return max(transaction, skip)
        return stock(0, buy)
```

## 714. Best Time to Buy and Sell Stock with Transaction Fee

### Solution 1:  recursive dp

```py
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        buy, sell = 0, 1
        @cache
        def stock(i, state):
            if i >= len(prices): return 0
            if state == buy:
                transaction = stock(i+1,state^1) - prices[i]
            else:
                transaction = stock(i+1,state^1) + prices[i] - fee
            skip = stock(i+1,state)
            return max(skip, transaction)
        return stock(0, buy)
```

```py
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        # s1 own stock, s2 do not own stock
        s1, s2 = -math.inf, 0
        for price in prices:
            ns1, ns2 = s1, s2
            # sell a stock
            ns2 = max(ns2, ns1 + price)
            # hold or buy a stock
            ns1 = max(ns1, ns2 - fee - price)
            s1, s2 = ns1, ns2
        return max(s1, s2)
```

## 26. Remove Duplicates from Sorted Array

### Solution 1:  two pointers + inplace

```py
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        left = 1
        n = len(nums)
        for right in range(1, n):
            if nums[right] != nums[right-1]:
                nums[left] = nums[right]
                left += 1
        return left
```

## 947. Most Stones Removed with Same Row or Column

### Solution 1:  union find + size of each disjoint connected component

```py
class UnionFind:
    def __init__(self,n: int):
        self.size = [1]*n
        self.parent = list(range(n))
    
    def find(self,i: int) -> int:
        if i==self.parent[i]:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self,i: int,j: int) -> bool:
        i, j = self.find(i), self.find(j)
        if i!=j:
            if self.size[i] < self.size[j]:
                i,j=j,i
            self.parent[j] = i
            self.size[i] += self.size[j]
            return True
        return False
    
    @property
    def root_count(self):
        return sum(node == self.find(node) for node in range(len(self.parent)))

    def __repr__(self) -> str:
        return f'parents: {[(i, parent) for i, parent in enumerate(self.parent)]}, sizes: {self.size}'
    
class Solution:
    def removeStones(self, stones: List[List[int]]) -> int:
        n = len(stones)
        dsu = UnionFind(n)
        rows, cols = defaultdict(list), defaultdict(list)
        for i, (r, c) in enumerate(stones):
            rows[r].append(i)
            cols[c].append(i)
        for row in rows.values():
            for r in range(1, len(row)):
                dsu.union(row[r-1], row[r])
        for col in cols.values():
            for c in range(1, len(col)):
                dsu.union(col[c-1], col[c])
        return len(stones) - dsu.root_count
```

### solution 2:  iterative union find + dictionary

```py
class UnionFind:
    def __init__(self):
        self.size = dict()
        self.parent = dict()
    
    def find(self,i: int) -> int:
        if i not in self.parent:
            self.size[i] = 1
            self.parent[i] = i
        while i != self.parent[i]:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self,i: int,j: int) -> bool:
        i, j = self.find(i), self.find(j)
        if i!=j:
            if self.size[i] < self.size[j]:
                i,j=j,i
            self.parent[j] = i
            self.size[i] += self.size[j]
            return True
        return False
    
    @property
    def root_count(self):
        return sum(node == self.find(node) for node in self.parent)

    def __repr__(self) -> str:
        return f'parents: {[(i, parent) for i, parent in enumerate(self.parent)]}, sizes: {self.size}'
    
class Solution:
    def removeStones(self, stones: List[List[int]]) -> int:
        dsu = UnionFind()
        for x, y in stones:
            dsu.union(f'x: {x}', f'y: {y}')
        return len(stones) - dsu.root_count
```

## 151. Reverse Words in a String

### Solution 1:  reverse + split with delimiter as whitespace

```py
class Solution:
    def reverseWords(self, s: str) -> str:
        return ' '.join(reversed(s.split()))
```

## 62. Unique Paths 

### Solution 1:  space optimized iterative dp + O(nm) time and O(n) space

```py
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        layer = [1]*n
        for _, j in product(range(1, m), range(1, n)):
            layer[j] += layer[j-1]
        return layer[-1]
```

### Solution 2:  navigate grid with combinations of down and right moves + math + combinations

![combinations](images/unique_paths_combinations.png)

```py
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        return math.comb(n+m-2, m-1)
```

## 222. Count Complete Tree Nodes

### Solution 1:  math + O(log^2(n)) time

```py
class Solution:
    def height(self, root: Optional[TreeNode]) -> int:
        h = -1
        while root:
            root = root.left
            h += 1
        return h
    def countNodes(self, root: Optional[TreeNode]) -> int:
        result = 0
        while root:
            h_left, h_right = self.height(root), self.height(root.right) + 1
            result += (1 << h_right)
            root = root.left if h_left != h_right else root.right
        return result
```

## 63. Unique Paths II

### Solution 1:  space optimized iterative dp + O(nm) time and O(n) space

```py
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        R, C = len(obstacleGrid), len(obstacleGrid[0])
        layer = [0]*C
        for c in range(C):
            if obstacleGrid[0][c] == 1: break
            layer[c] = 1
        for r, c in product(range(1, R), range(C)):
            if c > 0:
                layer[c] += layer[c-1]
            if obstacleGrid[r][c] == 1:
                layer[c] = 0
        return layer[-1]
```

## 304. Range Sum Query 2D - Immutable

### Solution 1:  2 dimensional prefix sum + O(1) queries

```py
class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        R, C = len(matrix), len(matrix[0])
        self.psum = [[0]*(C+1) for _ in range(R+1)]
        for r, c in product(range(R), range(C)):
            self.psum[r+1][c+1] = self.psum[r][c+1] + self.psum[r+1][c] + matrix[r][c] - self.psum[r][c]

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        return self.psum[row2+1][col2+1] - self.psum[row2+1][col1] - self.psum[row1][col2+1] + self.psum[row1][col1]
```

## 1314. Matrix Block Sum

### Solution 1:  2 dimensional prefix sum

```py
class Solution:
    def matrixBlockSum(self, mat: List[List[int]], k: int) -> List[List[int]]:
        R, C = len(mat), len(mat[0])
        ans = [[0]*C for _ in range(R)]
        # CONSTRUCT 2 DIMENSIONAL PREFIX SUM
        psum = [[0]*(C+1) for _ in range(R+1)]
        for r, c in product(range(R), range(C)):
            psum[r+1][c+1] = psum[r][c+1] + psum[r+1][c] + mat[r][c] - psum[r][c]
        # 2 DIMENSIONAL RANGE QUERIES
        for r, c in product(range(R), range(C)):
            # UPPER LEFT CORNER
            ul_row, ul_col = max(0, r-k), max(0, c-k)
            # BOTTOM RIGHT CORNER
            br_row, br_col = min(R-1, r+k), min(C-1, c+k)
            ans[r][c] = psum[br_row+1][br_col+1] - psum[br_row+1][ul_col] - psum[ul_row][br_col+1] + psum[ul_row][ul_col]
        return ans
```

## 2465. Number of Distinct Averages

### Solution 1:  set + two pointers + tilde i trick for when right pointer moves with left pointer each time + increment equals decrement at each iteration

```py
class Solution:
    def distinctAverages(self, nums: List[int]) -> int:
        n = len(nums)
        nums.sort()
        return len({nums[i] + nums[~i] for i in range(n//2)})
```

## 2466. Count Ways To Build Good Strings

### Solution 1:  iterative dp + python list

dp[i] is the number of ways to construct strings of length i. 

![example](images/good_strings.png)

```py
class Solution:
    def countGoodStrings(self, low: int, high: int, zero: int, one: int) -> int:
        memo = [0]*(high+1)
        memo[0], mod = 1, int(1e9) + 7
        result = 0
        for i in range(1, high + 1):
            if i >= zero:
                memo[i] = (memo[i] + memo[i-zero])%mod
            if i >= one:
                memo[i] = (memo[i] + memo[i-one])%mod
            if i >= low:
                result = (result + memo[i])%mod
        return result
```

### Solution 2: counter

```py
class Solution:
    def countGoodStrings(self, low: int, high: int, zero: int, one: int) -> int:
        memo = Counter({0:1})
        mod = int(1e9) + 7
        result = 0
        for i in range(1, high + 1):
            memo[i] = (memo[i] + memo[i-zero] + memo[i-one])%mod
            if i >= low:
                result = (result + memo[i])%mod
        return result
```

## 2467. Most Profitable Path in a Tree

### Solution 1:  rooted tree represented with undirected graph + dfs to find path from root to leaf + construct parent map + bfs for alice to leaf nodes

```py
class Solution:
    def mostProfitablePath(self, edges: List[List[int]], bob: int, amount: List[int]) -> int:
        n = len(amount)
        adj_list = defaultdict(list)
        for u, v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
        parent = [-1]*n
        dist_from_root = [inf]*n
        def dfs(node: int, cur_dist: int) -> None:
            dist_from_root[node] = cur_dist
            for child_node in adj_list[node]:
                if child_node == parent[node]: continue
                parent[child_node] = node
                dfs(child_node, cur_dist + 1)
        dfs(0, 0)
        bob_dist = 0
        while bob != 0:
            if bob_dist < dist_from_root[bob]:
                amount[bob] = 0
            elif bob_dist == dist_from_root[bob]:
                amount[bob] //= 2
            bob = parent[bob]
            bob_dist += 1
        queue = deque([(0, 0)]) # (node, profit)
        best = -inf
        is_leaf = lambda node: parent[node] != -1 and len(adj_list[node]) == 1
        while queue:
            node, profit = queue.popleft()
            profit += amount[node]
            if is_leaf(node): best = max(best, profit)
            for child_node in adj_list[node]:
                if child_node == parent[node]: continue
                queue.append((child_node, profit))
        return best
```

## 2468. Split Message Based on Limit

### Solution 1:

```py
class Solution:
    def splitMessage(self, message: str, limit: int) -> List[str]:
        n = len(message)
        def num_parts() -> int:
            for i in range(1, 5):
                num_chars = limit - i - 3
                parts = 0
                remaining_chars = n
                m = 9
                for j in range(1, i+1):
                    num_chars -= 1
                    chars = num_chars*m
                    if chars >= remaining_chars:
                        return parts + (remaining_chars + num_chars - 1)//num_chars
                    parts += m
                    remaining_chars -= chars
                    m *= 10
            return -1
        nparts = num_parts()
        result = []
        if nparts == -1: return result
        index = 0
        for i in range(1, nparts + 1):
            suffix = f'<{i}/{nparts}>'
            part_len = limit - len(suffix)
            result.append(f'{message[index:index+part_len]}{suffix}')
            index += part_len
        return result
```

## 374. Guess Number Higher or Lower

### Solution 1:  bisect + binary search with custom key comparator

```py
class Solution:
    def guessNumber(self, n: int) -> int:
        return bisect.bisect_right(range(1, n+1), 0, key = lambda num: -guess(num))
```

## 2472. Maximum Number of Non-overlapping Palindrome Substrings

### Solution 1:  iterative dp + counter + palindrome

```py
class Solution:
    def maxPalindromes(self, s: str, k: int) -> int:
        n = len(s)
        memo = Counter()
        def palindrome_len(i: int , j: int) -> int:
            while i > 0 and j < n-1 and s[i] == s[j] and j-i+1 < k:
                i -= 1
                j += 1
            return j-i+1-(2 if s[i] != s[j] else 0)
        for i in range(n):
            p1 = palindrome_len(i, i)
            if p1 >= k:
                memo[i+p1//2] = max(memo[i+p1//2], memo[i-p1//2-1] + 1)
            if i < n -1:
                p2 = palindrome_len(i, i+1)
                if p2 >= k:
                    memo[i+p2//2] = max(memo[i+p2//2], memo[i-p2//2] + 1)
            memo[i] = max(memo[i], memo[i-1])
        return memo[n-1] 
```

## 2469. Convert the Temperature

### Solution 1:  trivial

```py
class Solution:
    def convertTemperature(self, celsius: float) -> List[float]:
        return [celsius + 273.15, celsius*1.80+32.00]
```

## 2470. Number of Subarrays With LCM Equal to K

### Solution 1:  brute force + math + lcm

```py
class Solution:
    def subarrayLCM(self, nums: List[int], k: int) -> int:
        n = len(nums)
        result = 0
        for i in range(n):
            lcm_ = 1
            for j in range(i, n):
                lcm_ = lcm(lcm_, nums[j])
                if lcm_ > k: break
                result += (lcm_ == k)
        return result
```

## 2471. Minimum Number of Operations to Sort a Binary Tree by Level

### Solution 1:  queue + bfs + build array for each level + construct (directed graph for each level) + calculate the cycle lengths for each level

```py
class Solution:
    def minimumOperations(self, root: Optional[TreeNode]) -> int:
        def calc_swaps(arr):
            sarr = sorted(arr)
            compressed = dict()
            for i, num in enumerate(sarr):
                compressed[num] = i
            adj_list = dict()
            for i, num in enumerate(arr):
                correct_index = compressed[num]
                adj_list[i] = correct_index
            visited = [0]*len(arr)
            num_swaps = 0
            queue = [(num, 0) for num in range(len(arr))]
            while queue:
                node, cycle_len = queue.pop()
                if visited[node]: continue
                visited[node] = 1
                nei_node = adj_list[node]
                if visited[nei_node]:
                    num_swaps += cycle_len
                    continue
                queue.append((nei_node, cycle_len + 1))
            return num_swaps
        index_dict = defaultdict(list)
        queue = deque([root])
        cnt = 0
        while queue:
            sz = len(queue)
            for _ in range(sz):
                node = queue.popleft()
                index_dict[cnt].append(node.val)
                queue.extend(filter(None, (node.left, node.right)))
            cnt += 1
        result = 0
        for arr in index_dict.values():
            result += calc_swaps(arr)
        return result
```

## 2403. Minimum Time to Kill All Monsters

### Solution 1:  bitmask + minheap

```py
class Solution:
    def minimumTime(self, power: List[int]) -> int:
        n = len(power)
        power.sort()
        endmask = (1 << n) - 1
        dp = defaultdict(lambda: inf)
        dp[0] = 0
        minheap = [(0, 1, 0)] # (day, gain mask)
        while minheap:
            day, gain, mask = heappop(minheap)
            if day > dp[mask]: continue
            if mask == endmask: return day
            for i in range(n):
                if (mask>>i)&1: continue # monster already killed in this mask
                num_days = math.ceil(power[i]/gain) + dp[mask]
                nstate = mask|(1<<i)
                if num_days < dp[nstate]:
                    dp[nstate] = num_days
                    heappush(minheap, (num_days, gain+1, nstate))
```

## 263. Ugly Number

### Solution 1:  math + simulation

```py
class Solution:
    def isUgly(self, n: int) -> bool:
        for p in [2, 3, 5]:
            while n%p == 0 < n:
                n //= p
        return n == 1
```

## 119. Pascal's Triangle II

### Solution 1:  iterative dp + O(n) time and O(n) space

```py
class Solution:
    def getRow(self, rowIndex: int) -> List[int]:
        prev_row = [1]
        for i in range(1, rowIndex + 1):
            row = [1]
            for j in range(1, i):
                row.append(prev_row[j-1]+prev_row[j])
            prev_row = row + [1]
        return prev_row
```

## 931. Minimum Falling Path Sum

### Solution 1:  in-place + space optimized iterative dp + O(1) space

```py
class Solution:
    def minFallingPathSum(self, matrix: List[List[int]]) -> int:
        n = len(matrix)
        for i, j in product(range(1, n), range(n)):
            matrix[i][j] = min(matrix[i-1][max(0, j-1)], matrix[i-1][j], matrix[i-1][min(n-1, j+1)]) + matrix[i][j]
        return min(matrix[-1])
```

## 120. Triangle

### Solution 1:  iterative dp

```py
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        row = triangle[-1]
        for cur_row in reversed(triangle[:-1]):
            next_row = [inf]*len(cur_row)
            for i, val in enumerate(cur_row):
                next_row[i] = min(row[i], row[i+1]) + val
            row = next_row
        return row[0]
```

## 516. Longest Palindromic Subsequence

### Solution 1:

```py

```

## 64. Minimum Path Sum

### Solution 1:  iterative dp + space optimized O(C) space + accumulate

```py
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        row = list(accumulate(grid[0]))
        for r in range(1, R):
            nrow = [inf]*C
            for c in range(C):
                nrow[c] = min(nrow[c], row[c] + grid[r][c])
                if c > 0:
                    nrow[c] = min(nrow[c], nrow[c-1] + grid[r][c])
            row = nrow            
        return row[-1]
```

## 264. Ugly Number II

### Solution 1:  set + minheap

```py
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        seen = set([1])
        minheap = [1]
        for _ in range(n):
            num = heappop(minheap)
            for mul in map(lambda x: num*x, (2, 3, 5)):
                if mul in seen: continue
                seen.add(mul)
                heappush(minheap, mul)
        return num
```

```py
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        pointers = [0]*3
        numbers = [1]
        for _ in range(n-1):
            min_val = inf
            for i, val in enumerate([2, 3, 5]):
                cand = numbers[pointers[i]]*val
                min_val = min(min_val, cand)
            for i, val in enumerate([2, 3, 5]):
                pointers[i] += (min_val == numbers[pointers[i]]*val)
            numbers.append(min_val)
        return numbers[-1]
```

## 96. Unique Binary Search Trees

### Solution 1:  Catalan Numbers + iterative dp

![image](images/unique_BUT_1.png)
![image](images/unique_BUT_2.png)

```py
class Solution:
    def numTrees(self, n: int) -> int:
        dp = [0]*(n+1)
        dp[0] = 1
        for i in range(1, n + 1):
            for j in range(i):
                dp[i] += dp[j]*dp[i-j-1]
        return dp[-1]
```

### Solution 2:  catalan's numbers + analytical formula + O(n) time

```py
class Solution:
    def numTrees(self, n: int) -> int:
        cn = 1
        for i in range(1, n + 1):
            cn = (2*(2*i - 1)*cn)//(i + 1)
        return cn
```

## 1143. Longest Common Subsequence

### Solution 1:  iterative dp + space optimized O(C) space

```py
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        R, C = len(text1), len(text2)
        dp = [0]*(C+1)
        for r in range(R):
            ndp = [0]*(C+1)
            len_ = 0
            for c in range(C):
                len_ = max(ndp[c], dp[c+1])
                ndp[c+1] = max(len_, dp[c] + (text1[r] == text2[c]))
            dp = ndp
        return max(dp)
```

## 72. Edit Distance

### Solution 1: recursive dp + 4 decisions at each state

```py
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        n1, n2 = len(word1), len(word2)
        @cache
        def dp(i, j):
            if j == n2: return n1 - i
            replace = dp(i+1, j+1)+1 if i < n1 else inf
            insert = dp(i, j+1)+1
            remove = dp(i+1, j)+1 if i < n1 else inf
            noop = dp(i+1, j+1) if i < n1 and word1[i] == word2[j] else inf
            return min(noop, replace, insert, remove)
        return dp(0, 0)
```

## 152. Maximum Product Subarray

### Solution 1:  prefix max + iterative dp

```py
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        result = -inf
        pos, neg = 1, 1
        for num in nums:
            npos, nneg = num*pos, num*neg
            neg = min(nneg, npos)
            pos = max(npos, nneg)
            result = max(result, pos)
            pos = max(pos, 1)
            if neg >= 0:
                neg = 1
        return result
```

## 322. Coin Change

### Solution 1:  iterative dp + O(amount*len(coins)) time

```py
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        n = len(coins)
        dp = [inf]*(amount+1)
        dp[0] = 0
        for amt in range(1, amount+1):
            for coin in coins:
                if coin <= amt:
                    dp[amt] = min(dp[amt], dp[amt-coin] + 1)
        return dp[-1] if dp[-1] < inf else -1
```

## 518. Coin Change II

### Solution 1:  recursive dp + O(amount*len(coins)) time + trick to avoid repeat combinations by increasing i + greedy aspect + unbounded knapsack

![image](images/unbounded_knapsack.png)

```py
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        n = len(coins)
        @cache
        def dfs(i: int, amt: int) -> int:
            if amt == 0: return 1
            if amt < 0 or i == n: return 0
            cur = dfs(i+1, amt) + dfs(i, amt - coins[i])
            return cur
        return dfs(0, amount)
```

```py
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        n = len(coins)
        dp = [0] * (amount + 1)
        dp[0] = 1
        for coin, i in product(coins, range(1, amount + 1)):
            if coin <= i:
                dp[i] += dp[i - coin]
        return dp[-1]
```

## 1263. Minimum Moves to Move a Box to Their Target Location

### Solution 1:  bfs

```py
class Solution:
    def is_reachable(self, grid: List[List[str]], start: Tuple[int, int], target: Tuple[int, int], box: Tuple[int, int]) -> bool:
        R, C = len(grid), len(grid[0])
        wall = '#'
        in_bounds = lambda r, c: 0 <= r < R and 0 <= c < C
        queue = deque([start])
        seen = set([start])
        while queue:
            r, c = queue.popleft()
            if (r, c) == target: return True
            for nr, nc in [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]:
                if not in_bounds(nr, nc) or grid[nr][nc] == wall or (nr, nc) == box or (nr, nc) in seen: continue
                seen.add((nr, nc))
                queue.append((nr, nc))
        return False
    def minPushBox(self, grid: List[List[str]]) -> int:
        R, C = len(grid), len(grid[0])
        in_bounds = lambda r, c: 0 <= r < R and 0 <= c < C
        def is_vertical(r: int, c: int) -> bool:
            for nr, nc in [(r+1, c), (r-1, c)]:
                if not in_bounds(nr, nc) or grid[nr][nc] == wall: return False
            return True
        def is_horizontal(r: int, c: int) -> bool:
            for nr, nc in [(r, c+1), (r, c-1)]:
                if not in_bounds(nr, nc) or grid[nr][nc] == wall: return False
            return True
        seen = set()
        player, wall, floor, target, box = 'S', '#', '.', 'T', 'B'
        initial = [0, 0, 0, 0]
        for r, c in product(range(R), range(C)):
            if grid[r][c] == player:
                initial[0], initial[1] = r, c
            if grid[r][c] == box:
                initial[2], initial[3] = r, c
        queue = deque([tuple(initial)])
        pushes = 0
        while queue:
            sz = len(queue)
            for _ in range(sz):
                sr, sc, br, bc = queue.popleft() # (player_r, player_c, box_r, box_c)
                if grid[br][bc] == target: return pushes
                nei_cells = []
                if is_vertical(br, bc):
                    nei_cells.extend([(br+1, bc), (br-1, bc)])
                if is_horizontal(br, bc):
                    nei_cells.extend([(br, bc+1), (br, bc-1)])
                nei_cells = list(filter(lambda x: self.is_reachable(grid, (sr, sc), x, (br, bc)), nei_cells))
                for nr, nc in nei_cells:
                    dr, dc = br - nr, bc - nc
                    nbr, nbc = br + dr, bc + dc
                    nstate = (br, bc, nbr, nbc)
                    if nstate in seen: continue
                    seen.add(nstate)
                    queue.append(nstate)
            pushes += 1
        return -1
```

## 224. Basic Calculator

### Solution 1:  greedy + stack

```py
class Solution:
    def calculate(self, s: str) -> int:
        num = 0
        sign = 1
        stack = [0]
        for ch in s:
            if ch.isspace(): continue
            elif ch.isdigit():
                num = num*10 + int(ch)
            elif ch == '+':
                stack[-1] += sign*num
                sign = 1
                num = 0
            elif ch == '-':
                stack[-1] += sign*num
                sign = -1
                num = 0
            elif ch == '(':
                stack.extend([sign, 0])
                sign = 1
                num = 0
            else:
                last_num = stack.pop() + sign*num
                last_sign = stack.pop()
                stack[-1] += last_sign*last_num
                sign = 1
                num = 0
        return stack[0]+sign*num
```

##

### Solution 1:

```py

```

##

### Solution 1:

```py

```

## 2477. Minimum Fuel Cost to Report to the Capital

### Solution 1:  postorder traversal of rooted tree to get the size of each subtree + preorder traversal of rooted tree to compute the number of cars needed to travel from child node to node.

```py
class Solution:
    def minimumFuelCost(self, roads: List[List[int]], seats: int) -> int:
        n = len(roads) + 1
        adj_list = [[] for _ in range(n)]
        sizes = [1]*n
        for u, v in roads:
            adj_list[u].append(v)
            adj_list[v].append(u)
        self.fuel = 0
        def postorder(parent: int, node: int) -> None:
            for child in adj_list[node]:
                if child == parent: continue
                postorder(node, child)
                sizes[node] += sizes[child]
        def preorder(parent: int, node: int) -> None:
            for child in adj_list[node]:
                if child == parent: continue
                num_cars = math.ceil(sizes[child]/seats)
                self.fuel += num_cars
                preorder(node, child)
        postorder(-1, 0)
        preorder(-1, 0)
        return self.fuel
```

### Solution 2:  single dfs with postorder traversal

```py
class Solution:
    def minimumFuelCost(self, roads: List[List[int]], seats: int) -> int:
        n = len(roads) + 1
        adj_list = [[] for _ in range(n)]
        sizes = [1]*n
        for u, v in roads:
            adj_list[u].append(v)
            adj_list[v].append(u)
        self.fuel = 0
        def dfs(parent: int, node: int, people: int = 1) -> int:
            for child in adj_list[node]:
                if child == parent: continue
                people += dfs(node, child)
            self.fuel += (math.ceil(people/seats) if node else 0) # cost to move all these people from this subtree to parent node
            return people
        dfs(-1, 0)
        return self.fuel
                
```

## 2478. Number of Beautiful Partitions

### Solution 1:  

```py

```

## 516. Longest Palindromic Subsequence

### Solution 1:  iterative dp + counter

```py
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        n = len(s)
        dp = Counter()
        for i in reversed(range(n)):
            dp[(i, i)] = 1
            for j in range(i+1, n):
                if s[i] == s[j]:
                    dp[(i, j)] = dp[(i+1, j-1)] + 2
                else:
                    dp[(i, j)] = max(dp[(i+1, j)], dp[(i, j-1)])
        return dp[(0, n-1)]
```

### Solution 2:  iterative dp + space optimized O(n) space

```py
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        n = len(s)
        dp = Counter()
        for i in reversed(range(n)):
            ndp = Counter()
            ndp[(i, i)] = 1
            for j in range(i+1, n):
                if s[i] == s[j]:
                    ndp[(i, j)] = dp[(i+1, j-1)] + 2
                else:
                    ndp[(i, j)] = max(dp[(i+1, j)], ndp[(i, j-1)])
            dp = ndp
        return dp[(0, n-1)]
```

## 279. Perfect Squares

### Solution 1:  iterative dp + O(n*sqrt(n)) time

```py
class Solution:
    def numSquares(self, n: int) -> int:
        perfect_squares = []
        i = 1
        while i*i <= n:
            perfect_squares.append(i*i)
            i += 1
        dp = [inf]*(n+1)
        dp[0] = 0
        for i in range(1, n+1):
            for ps in perfect_squares:
                if ps > i: break
                dp[i] = min(dp[i], dp[i-ps] + 1)
        return dp[-1]
```

### Solution 2:  minheap + find shortest path to n

```py
class Solution:
    def numSquares(self, n: int) -> int:
        perfect_squares = []
        i = 1
        while i*i <= n:
            perfect_squares.append(i*i)
            i += 1
        minheap = [(0, 0)]
        seen = set()
        while minheap:
            steps, val = heappop(minheap)
            val = abs(val)
            if val == n: return steps
            for ps in perfect_squares:
                if val + ps > n: break
                if val + ps in seen: continue
                seen.add(val+ps)
                heappush(minheap, (steps + 1, -(val + ps)))
        return -1
```

### Solution 3:  bfs on n-ary tree

```py
class Solution:
    def numSquares(self, n: int) -> int:
        perfect_squares = []
        i = 1
        while i*i <= n:
            perfect_squares.append(i*i)
            i += 1
        queue = deque([0])
        seen = [0]*(n+1)
        steps = 0
        while queue:
            sz = len(queue)
            for _ in range(sz):
                v = queue.popleft()
                if v == n: return steps
                for ps in perfect_squares:
                    nv = v + ps
                    if nv > n: break
                    if seen[nv]: continue
                    seen[nv] = 1
                    queue.append(nv)
            steps += 1
        return -1
```

## 343. Integer Break

### Solution 1:  iterative dp + O(n^2) time

![image](images/integer_break.png)

```py
class Solution:
    def integerBreak(self, n: int) -> int:
        dp = [0] * (n + 1)
        dp[1] = 1
        for i in range(2, n + 1):
            dp[i] = i if i < n else 0
            for j in range(1, i // 2 + 1):
                dp[i] = max(dp[i], dp[j] * dp[i - j])
        return dp[-1]
```

## 1926. Nearest Exit from Entrance in Maze

### Solution 1:  bfs

```py
class Solution:
    def nearestExit(self, maze: List[List[str]], entrance: List[int]) -> int:
        R, C = len(maze), len(maze[0])
        floor, wall = '.', '+'
        queue = deque([tuple(entrance)])
        steps = 0
        seen = set([tuple(entrance)])
        in_bounds = lambda r, c: 0 <= r < R and 0 <= c < C
        is_border = lambda r, c: r in (0, R-1) or c in (0, C-1)
        def neighborhood(r: int, c: int) -> Iterable[int]:
            for nr, nc in [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]:
                if not in_bounds(nr, nc) or maze[nr][nc] == wall or (nr, nc) in seen: continue
                yield nr, nc
        while queue:
            sz = len(queue)
            for _ in range(sz):
                r, c = queue.popleft()
                if steps > 0 and is_border(r, c): return steps
                for nr, nc in neighborhood(r, c):
                    seen.add((nr, nc))
                    queue.append((nr, nc))
            steps += 1
        return -1
```

## 1201. Ugly Number III

### Solution 1:  binary search + number theory + lcm (least common multiple)

```py
class Solution:
    def nthUglyNumber(self, n: int, a: int, b: int, c: int) -> int:
        left, right = 0, 2*10**9
        num_terms = lambda target, integers: target//lcm(*integers)
        def possible(target: int) -> bool:
            total_terms = num_terms(target, [a]) + num_terms(target, [b]) + num_terms(target, [c]) - num_terms(target, [a,b]) - num_terms(target, [a, c]) - num_terms(target, [b, c]) + num_terms(target, [a, b, c])
            return total_terms >= n
        while left < right:
            mid = (left + right) >> 1
            if possible(mid):
                right = mid
            else:
                left = mid + 1
        return left
```

## 246. Strobogrammatic Number

### Solution 1:  set + map + reversed

```py
class Solution:
    def isStrobogrammatic(self, num: str) -> bool:
        for x in ['2', '3', '4', '5', '7']:
            if x in num: return False
        flipped_num = {'8': '8', '6': '9', '9': '6', '1': '1', '0': '0'}
        return num == ''.join(map(lambda dig: flipped_num[dig], reversed(num)))
```

### Solution 2:  two pointers + math.ceil

```py
class Solution:
    def isStrobogrammatic(self, num: str) -> bool:
        for x in ['2', '3', '4', '5', '7']:
            if x in num: return False
        flipped_num = {'8': '8', '6': '9', '9': '6', '1': '1', '0': '0'}
        for i in range(math.ceil(len(num)/2)):
            if flipped_num[num[i]] != num[~i]: return False
        return True
```

## 319. Bulb Switcher

### Solution 1:  math + count of perfect squares

### Description 

realization that all divisors of any integer that is not a perfect square, come in pairs x = 12, (1, 12), (2, 6), (3, 4) + but perfect squares come in pairs but the last pair is always going to be duplicate cause x*x = x^2 is a perfect square, so for example x = 36: (1, 36), (2, 18), (3, 12), (4, 9), (6, 6), but that 6 counts only once, so with an odd number of divisors it will be switched into on. 

```py
class Solution:
    def bulbSwitch(self, n: int) -> int:
        return isqrt(n)
```

## 907. Sum of Subarray Minimums

### Solution 1:  monotonic stack + for each index find left and right boundary 

```py
class Solution:
    def sumSubarrayMins(self, arr: List[int]) -> int:
        mod = int(1e9)+7
        n = len(arr)
        arr = [-1] + arr + [0]
        stack = [0]
        res = 0
        for i, v in enumerate(arr[1:], start = 1):
            while v <= arr[stack[-1]]:
                mid = stack.pop()
                left = stack[-1]
                right = i
                cnt = (mid-left)*(right-mid)
                res = (res + cnt*arr[mid])%mod
            stack.append(i)
        return res
```

## 2104. Sum of Subarray Ranges

### Solution 1:  increasing monotonic stack + decreasing monotonic stack

```py
class Solution:
    def subArrayRanges(self, nums: List[int]) -> int:
        stack = [0]
        max_num = int(1e9)
        arr = [-max_num-2] + nums + [-max_num-1]
        res = 0
        for i, v in enumerate(arr[1:], start = 1):
            while v <= arr[stack[-1]]:
                mid = stack.pop()
                left = stack[-1]
                right = i
                cnt = (mid-left)*(right-mid)
                res -= cnt*arr[mid]
            stack.append(i)
        arr = [max_num+2] + nums + [max_num+1]
        stack = [0]
        for i, v in enumerate(arr[1:], start = 1):
            while v >= arr[stack[-1]]:
                mid = stack.pop()
                left = stack[-1]
                right = i
                cnt = (mid-left)*(right-mid)
                res += cnt*arr[mid]
            stack.append(i)
        return res
```

## 2481. Minimum Cuts to Divide a Circle

### Solution 1:  math

```py
class Solution:
    def numberOfCuts(self, n: int) -> int:
        if n == 1: return 0
        return n if n&1 else n//2
```

## 2482. Difference Between Ones and Zeros in Row and Column

### Solution 1:  store the count for row and column to avoid recomputation + simulation

```py
class Solution:
    def onesMinusZeros(self, grid: List[List[int]]) -> List[List[int]]:
        R, C = len(grid), len(grid[0])
        rows, cols = [0]*R, [0]*C
        for r, c in product(range(R), range(C)):
            rows[r] += grid[r][c]
            cols[c] += grid[r][c]
        for r, c in product(range(R), range(C)):
            onesRow, onesCol, zerosRow, zerosCol = rows[r], cols[c], R-rows[r], C-cols[c]
            grid[r][c] = onesRow + onesCol - zerosRow - zerosCol
        return grid
```

## 2483. Minimum Penalty for a Shop

### Solution 1:  prefix and suffix count

```py
class Solution:
    def bestClosingTime(self, customers: str) -> int:
        customers += '$'
        suffixY = customers.count('Y')
        prefixN = 0
        minPenalty, minHour = inf, inf
        for i, cust in enumerate(customers):
            penalty = suffixY + prefixN
            if penalty < minPenalty:
                minPenalty = penalty
                minHour = i
            prefixN += (cust == 'N')
            suffixY -= (cust == 'Y')
        return minHour
```

## 2484. Count Palindromic Subsequences

### Solution 1:  recursive dp + state is (index, two starting characters, remaining characters)

```py
class Solution:
    def countPalindromes(self, s: str) -> int:
        n = len(s)
        mod = int(1e9) + 7
        @cache
        def dp(i: int, start: str, remaining: int) -> int:
            if remaining == 0: return 1
            if i == n: return 0
            if remaining > 3:
                take = dp(i + 1, start + s[i], remaining - 1)
                skip = dp(i + 1, start, remaining)
            elif remaining == 3:
                take = dp(i + 1, start, remaining - 1)
                skip = dp(i + 1, start, remaining)
            else:
                take = dp(i + 1, start, remaining - 1) if s[i] == start[remaining-1] else 0
                skip = dp(i + 1, start, remaining)
            return (take + skip)%mod
        return dp(0, '', 5)
```

### Solution 2: prefix and suffix count + multiply prefix and suffix about pivot i + xy_yx count

```py
class Solution:
    def countPalindromes(self, s: str) -> int:
        n = len(s)
        mod = int(1e9) + 7
        pref, cnts = [[[0]*10 for _ in range(10)] for _ in range(n)], [0]*10
        cnts[ord(s[0]) - ord('0')] += 1
        for i in range(1, n):
            dig = ord(s[i]) - ord('0')
            for j, k in product(range(10), repeat = 2):
                pref[i][j][k] = pref[i-1][j][k]
                if k == dig: pref[i][j][k] += cnts[j]
            cnts[dig] += 1
        suf, cnts = [[[0]*10 for _ in range(10)] for _ in range(n)], [0]*10
        cnts[ord(s[-1]) - ord('0')] += 1
        for i in reversed(range(n-1)):
            dig = ord(s[i]) - ord('0')
            for j, k in product(range(10), repeat = 2):
                suf[i][j][k] = suf[i+1][j][k]
                if k == dig: suf[i][j][k] += cnts[j]
            cnts[dig] += 1
        res = 0
        for i, j, k in product(range(2, n-2), range(10), range(10)):
            res = (res + pref[i-1][j][k]*suf[i+1][j][k])%mod
        return res
```

### Solution 3:  iterative dp + perform count for number of match for each 100 patterns

```py
class Solution:
    def countPalindromes(self, s: str) -> int:
        ans = 0
        mod = int(1e9) + 7
        n = len(s)
        for x in range(10):
            for y in range(10):
                pat = f'{x}{y}${y}{x}'
                dp = [0]*6
                dp[-1] = 1
                for i in range(n):
                    for j in range(5):
                        if s[i] == pat[j] or j == 2: dp[j] += dp[j+1]
                ans = (ans + dp[0])%mod
        return ans 
```

## 2480. Form a Chemical Bond

### Solution 1: self join

```sql
select
    l.symbol as metal,
    r.symbol as nonmetal
from Elements l, Elements r
where l.type = 'Metal' 
and r.type = 'Nonmetal'
```

## 2485. Find the Pivot Integer

### Solution 1:  prefix and suffix sum + O(n) time

```py
class Solution:
    def pivotInteger(self, n: int) -> int:
        psum = 0
        ssum = n*(n+1)//2
        for i in range(1, n+1):
            psum += i
            if psum == ssum: return i
            ssum -= i
        return -1
```

### Solution 2:  math + algebra + algebraic equation + O(sqrt(n)) time

```py
class Solution:
    def pivotInteger(self, n: int) -> int:
        sum_ = n*(n+1)//2
        pi = int(sqrt(sum_))
        return pi if pi*pi == sum_ else -1
```

## 2486. Append Characters to String to Make Subsequence

### Solution 1:  two pointers + greedy

```py
class Solution:
    def appendCharacters(self, s: str, t: str) -> int:
        n = len(t)
        i = 0
        for ch in s:
            if i < n and t[i] == ch: i += 1
        return n - i
```

## 2487. Remove Nodes From Linked List

### Solution 1:  stack + linked list + O(n) space

```py
class Solution:
    def removeNodes(self, head: Optional[ListNode]) -> Optional[ListNode]:
        stack = []
        while head:
            while stack and stack[-1] < head.val:
                stack.pop()
            stack.append(head.val)
            head = head.next
        sentinel_node = ListNode()
        cur = sentinel_node
        for v in stack:
            cur.next = ListNode(v)
            cur = cur.next
        return sentinel_node.next
```

### Solution 2:  reverse linked list twice + remove linked list node when the value is less than max value seen so far + O(1) space

```py
class Solution:
    def reverse(self, head: Optional[ListNode]) -> Optional[ListNode]:
        tail = None
        while head:
            nxt = head.next
            head.next = tail
            tail = head
            head = nxt
        return tail
    def removeNodes(self, head: Optional[ListNode]) -> Optional[ListNode]:
        node = tail = self.reverse(head)
        maxVal = node.val
        while node.next:
            if maxVal > node.next.val:
                node.next = node.next.next
            else:
                node = node.next
                maxVal = node.val
        return self.reverse(tail)
```

## 2488. Count Subarrays With Median K

### Solution 1:  convert to 0, 1, -1 + prefix sum + find x = 0 - prefix_sum and y = 1 - prefix_sum + balance look for 0 and 1

```py
class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        res = 1
        for i, num in enumerate(nums):
            if num < k:
                nums[i] = -1
            elif num > k:
                nums[i] = 1
            else:
                nums[i] = 0
        right_sums = Counter()
        right_sum = 0
        left_sum = 0
        found = False
        for num in nums:
            if num == 0: 
                found = True
            if found:
                right_sum += num
                right_sums[right_sum] += 1
            else:
                left_sum += num
        for num in nums:
            if num == 0: 
                res += right_sums[1] + right_sums[0] - 1
                break
            x = 0 - left_sum
            y = 1 - left_sum
            res += right_sums[x] + right_sums[y]
            left_sum -= num
        return res
```

## 265. Paint House II

### Solution 1:  iterative dp + space optimized O(1) and O(nk) time

Iterate over all the houses, and for each one track the minimum cost so far up to that house, and track the color of previous house that result in that minimum cost.  But you can't have two adjacent houses with same color.  So when looking at colors for next house, you can't always use minimum so far up to the i - 1 house.  So you need to track a second minimum cost as well that will be used when the current house is considering the same color as that which contributed to the minimum cost. 

```py
class Solution:
    def minCostII(self, costs: List[List[int]]) -> int:
        prev_min_color, prev_min_cost, prev_second_min_cost = None, inf, inf
        for house in costs:
            min_color, min_cost, second_min_cost = None, inf, inf
            for color, cost in enumerate(house):
                ncost = cost
                if prev_min_color is not None and prev_min_color == color:
                    ncost += prev_second_min_cost
                elif prev_min_color is not None:
                    ncost += prev_min_cost
                if ncost <= min_cost:
                    second_min_cost = min_cost
                    min_cost = ncost
                    min_color = color
                elif ncost < second_min_cost:
                    second_min_cost = ncost
            prev_min_color, prev_min_cost, prev_second_min_cost = min_color, min_cost, second_min_cost
        return prev_min_cost
```

## 206. Reverse Linked List

### Solution 1:  O(1) space + linked list

```py
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        tail = None
        while head:
            nxt = head.next
            head.next = tail
            tail = head
            head = nxt
        return tail
```

## 242. Valid Anagram

### Solution 1:  sort

```py
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        return sorted(s) == sorted(t)
```

## 74. Search a 2D Matrix

### Solution 1:  binary search row + binary search column + O(logm+logn) time

```py
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        R, C = len(matrix), len(matrix[0])
        left, right = 0, R-1
        while left < right:
            mid = (left + right + 1) >> 1
            if target < matrix[mid][0]:
                right = mid - 1
            else:
                left = mid
        i = bisect_right(matrix[left], target) - 1
        return matrix[left][i] == target
```

## 1020. Number of Enclaves

### Solution 1:  bfs + queue + in place + matrix + O(mn) time

```py
class Solution:
    def numEnclaves(self, grid: List[List[int]]) -> int:
        sea, land = 0, 1
        R, C = len(grid), len(grid[0])
        in_bounds = lambda r, c: 0 <= r < R and 0 <= c < C
        on_boundary = lambda r, c: r in (0, R-1) or c in (0, C-1)
        neighborhood = lambda r, c: ((r+1, c), (r-1, c), (r, c+1), (r, c-1))
        def bfs(r: int, c: int) -> None:
            queue = deque([(r, c)])
            grid[r][c] = sea
            while queue:
                r, c = queue.popleft()
                for nr, nc in neighborhood(r, c):
                    if not in_bounds(nr, nc) or grid[nr][nc] == sea: continue
                    grid[nr][nc] = sea
                    queue.append((nr, nc))
        for r, c in product(range(R), range(C)):
            if grid[r][c] == sea: continue
            if on_boundary(r, c):
                bfs(r, c)
        return sum(map(sum, grid))
```

## 1905. Count Sub Islands

### Solution 1:  bfs + queue + in place + matrix + O(mn) time

```py
class Solution:
    def countSubIslands(self, grid1: List[List[int]], grid2: List[List[int]]) -> int:
        water, land = 0, 1
        R, C = len(grid1), len(grid1[0])
        in_bounds = lambda r, c: 0 <= r < R and 0 <= c < C
        neighborhood = lambda r, c: ((r+1, c), (r-1, c), (r, c+1), (r, c-1))
        def bfs(r: int, c: int) -> bool:
            queue = deque([(r, c)])
            is_subisland = grid1[r][c] == grid2[r][c]
            grid2[r][c] = water # mark as visited
            while queue:
                r, c = queue.popleft()
                for nr, nc in neighborhood(r, c):
                    if not in_bounds(nr, nc) or grid2[nr][nc] == water: continue
                    is_subisland &= grid1[nr][nc] == grid2[nr][nc]
                    grid2[nr][nc] = water
                    queue.append((nr, nc))
            return is_subisland
        res = 0
        for r, c in product(range(R), range(C)):
            if grid2[r][c] == land:
                res += bfs(r, c)
        return res
```

### Solution 2:  bfs + queue + remove all non sub islands + count remaining islands in grid2

```py
class Solution:
    def countSubIslands(self, grid1: List[List[int]], grid2: List[List[int]]) -> int:
        water, land = 0, 1
        R, C = len(grid1), len(grid1[0])
        in_bounds = lambda r, c: 0 <= r < R and 0 <= c < C
        neighborhood = lambda r, c: ((r+1, c), (r-1, c), (r, c+1), (r, c-1))
        def bfs(r: int, c: int) -> None:
            queue = deque([(r, c)])
            grid2[r][c] = water # mark as visited
            while queue:
                r, c = queue.popleft()
                for nr, nc in neighborhood(r, c):
                    if not in_bounds(nr, nc) or grid2[nr][nc] == water: continue
                    grid2[nr][nc] = water
                    queue.append((nr, nc))
        res = 0
        # REMOVING ANY ISLAND IN GRID 2 THAT IS NOT A SUB ISLAND OF SOME ISLAND IN GRID1
        for r, c in product(range(R), range(C)):
            if grid2[r][c] == land and grid1[r][c] == water:
                bfs(r, c)
        # COUNT THE REMAINING ISLANDS IN GRID2 BECAUSE THEY ARE SUBISLANDS
        for r, c in product(range(R), range(C)):
            if grid2[r][c] == land:
                res += 1
                bfs(r, c)
        return res  
```

## 1207. Unique Number of Occurrences

### Solution 1:  set + counter + O(n) time + walrus operator

```py
class Solution:
    def uniqueOccurrences(self, arr: List[int]) -> bool:
        return len(c := Counter(arr)) == len(set(c.values()))
```

## 1704. Determine if String Halves Are Alike

### Solution 1:  sum + string

```py
class Solution:
    def halvesAreAlike(self, s: str) -> bool:
        return sum([1 if i < len(s)//2 else -1 for i, ch in enumerate(s) if ch in 'aeiouAEIOU']) == 0
```

## 1165. Single-Row Keyboard

### Solution 1:  string + convert characters to integer value (ascii value) + array as hash table

```py
class Solution:
    def calculateTime(self, keyboard: str, word: str) -> int:
        word = keyboard[0] + word
        n = len(word)
        pos = [0]*26
        unicode = lambda ch: ord(ch) - ord('a')
        for i, ch in enumerate(keyboard):
            pos[unicode(ch)] = i
        return sum([abs(pos[unicode(word[i])] - pos[unicode(word[i-1])]) for i in range(1, n)])
            
```

## 144. Binary Tree Preorder Traversal

### Solution 1: iterative + stack + filter + left sided preorder traversal + dfs

```py
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        stack, res = [root] if root else [], []
        while stack:
            node = stack.pop()
            res.append(node.val)
            stack.extend(list(filter(None, (node.right, node.left))))
        return res
```

## 94. Binary Tree Inorder Traversal

### Solution 1:  stack + iterative + inorder traversal

```py
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        stack, res = [], []
        while root or stack:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            res.append(root.val)
            root = root.right
        return res
```

## 145. Binary Tree Postorder Traversal

### Solution 1:  two stacks + main stack + right child stack + iterative postorder traversal

```py
class Solution:
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        stack, rightChildStack, res = [], [], []
        while root or stack:
            if root:
                stack.append(root)
                if root.right:
                    rightChildStack.append(root.right)
                root = root.left
            elif stack and rightChildStack and stack[-1].right == rightChildStack[-1]:
                root = rightChildStack.pop()
            else:
                root = stack.pop()
                res.append(root.val)
                root = None
        return res
```

## 102. Binary Tree Level Order Traversal

### Solution 1:  bfs + sum of lists (concatenation of lists) + filter

```py
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        res = []
        levels = [root] if root else []
        while levels:
            res.append([node.val for node in levels])
            levels = sum([list(filter(None, (node.left, node.right))) for node in levels], start = [])
        return res
```

## 104. Maximum Depth of Binary Tree

### Solution 1:  level order traversal + bfs + concatenation of lists + filter

```py
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        depth = 0
        levels = [root] if root else []
        while levels:
            levels = sum([list(filter(None, (node.left, node.right))) for node in levels], start = [])
            depth += 1
        return depth
```

## 203. Remove Linked List Elements

### Solution 1:  linked list + dummy node + removing

```py
class Solution:
    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        sentinel = ListNode(next = head)
        cur = sentinel
        while cur and cur.next:
            if cur.next.val == val:
                cur.next = cur.next.next
            else:
                cur = cur.next
        return sentinel.next
```

## 53. Maximum Subarray

### Solution 1:  iterative dp + O(1) space + kadane's algorithm

```py
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        curSum, maxSum = 0, -inf
        for num in nums:
            curSum = max(num, curSum + num)
            maxSum = max(maxSum, curSum)
        return maxSum
```

## 1. Two Sum

### Solution 1:  dictionary + O(n) space + O(n) time

```py
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        seen = {}
        for i, num in enumerate(nums):
            cand = target-num
            if cand in seen:
                return [seen[cand], i]
            seen[num] = i
        return [0,0]
```

## 88. Merge Sorted Array

### Solution 1:  backwards scan

```py
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        n -= 1
        m -= 1
        for i in reversed(range(n+m+2)):
            if n < 0:
                nums1[i] = nums1[m]
                m -= 1
            elif m < 0:
                nums1[i] = nums2[n]
                n -= 1
            elif nums1[m] > nums2[n]:
                nums1[i] = nums1[m]
                m -= 1
            else:
                nums1[i] = nums2[n]
                n -= 1
```

## 350. Intersection of Two Arrays II

### Solution 1:  bucket sort + bucket count

```py
class Solution:
    def bucket_count(self, nums: List[int]) -> List[int]:
        bucket = [0]*1001
        for num in nums:
            bucket[num] += 1
        return bucket
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        b1, b2 = self.bucket_count(nums1), self.bucket_count(nums2)
        res = []
        for i in range(1001):
            cnt = min(b1[i], b2[i])
            res.extend([i]*cnt)
        return res
```

## 566. Reshape the Matrix

### Solution 1:  matrix + looping 

```py
class Solution:
    def matrixReshape(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
        m, n = len(mat), len(mat[0])
        if m*n != r*c: return mat
        row = col = 0
        res = [[0]*c for _ in range(r)]
        for i, j in product(range(m), range(n)):
            res[row][col] = mat[i][j]
            col += 1
            if col == c:
                col = 0
                row += 1
        return res
```

### Solution 2:  matrix + modulus + internal storage of 2d matrix is in a 1d array

```py
class Solution:
    def matrixReshape(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
        m, n = len(mat), len(mat[0])
        if m*n != r*c: return mat
        cnt = 0
        res = [[0]*c for _ in range(r)]
        for i, j in product(range(m), range(n)):
            res[cnt//c][cnt%c] = mat[i][j]
            cnt += 1
        return res
```

## 226. Invert Binary Tree

### Solution 1:  bfs + swap children nodes for node in level + sum of lists

```py
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        levels = [node := root] if root else []
        while levels:
            for node in levels:
                node.left, node.right = node.right, node.left
            levels = sum([list(filter(None, (node.left, node.right))) for node in levels], start = [])
        return root
```

```py
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        level = [node := root] if root else []
        while level:
            nlevel = []
            for node in level:
                node.left, node.right = node.right, node.left
                nlevel.extend(list(filter(None, (node.left, node.right))))
            level = nlevel
        return root
```

### Solution 2: recursive + swap children nodes for node + return node

```py
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root: return root
        root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
        return root
```

## 1657. Determine if Two Strings Are Close

### Solution 1:  counter + set to check for character occurrences + sort the values of counter to check equal frequencies of characters

```py
class Solution:
    def closeStrings(self, word1: str, word2: str) -> bool:
        return set(c1 := Counter(word1)) == set(c2 := Counter(word2)) and sorted(c1.values()) == sorted(c2.values())
```

## 451. Sort Characters By Frequency

### Solution 1:  string + map + sort + counter

```py
class Solution:
    def frequencySort(self, s: str) -> str:
        res = ''
        for key, vals in groupby(Counter(s).most_common(), key = lambda pair: pair[0]):
            freq = list(vals)[0][1]
            res += key*freq
        return res
```

```py
class Solution:
    def frequencySort(self, s: str) -> str:
        return ''.join(sum([[key*freq] for key, freq in map(lambda pair: (pair[0], list(pair[1])[0][1]), groupby(Counter(s).most_common(), key = lambda pair: pair[0]))], start = []))
```

```py
class Solution:
    def frequencySort(self, s: str) -> str:
        return ''.join(chain([key*freq for key, freq in map(lambda pair: (pair[0], list(pair[1])[0][1]), groupby(Counter(s).most_common(), key = lambda pair: pair[0]))]))
```

```py
class Solution:
    def frequencySort(self, s: str) -> str:
        return ''.join([key*freq for key, freq in sorted(Counter(s).items(), key = lambda pair: pair[1], reverse = True)])
```

```py
class Solution:
    def frequencySort(self, s: str) -> str:
        counts = Counter(s)
        return "".join(sorted(s, key = lambda x: (counts[x], x), reverse = True))
```

## 1162. As Far from Land as Possible

### Solution 1:

```py
class Solution:
    def maxDistance(self, grid: List[List[int]]) -> int:
        water, land = 0, 1
        n = len(grid)
        queue = deque([(r, c, 0) for r, c in product(range(n), repeat = 2) if grid[r][c] == land])
        seen = set([(r, c) for r, c in product(range(n), repeat = 2) if grid[r][c] == land])
        res = -1
        in_bounds = lambda r, c: 0 <= r < n and 0 <= c < n
        neighborhood = lambda r, c: [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
        while queue:
            r, c, dist = queue.popleft()
            if grid[r][c] == water:
                res = max(res, dist)
            for nr, nc in neighborhood(r, c):
                if not in_bounds(nr, nc) or (nr, nc) in seen: continue
                seen.add((nr, nc))
                queue.append((nr, nc, dist+1))
        return res
```

## 806. Number of Lines To Write String

### Solution 1:  simulation + string

```py
class Solution:
    def numberOfLines(self, widths: List[int], s: str) -> List[int]:
        unicode = lambda ch: ord(ch) - ord('a')
        row, pixels = 1, 0
        for width in map(lambda ch: widths[unicode(ch)], s):
            pixels += width
            if pixels > 100:
                pixels = width
                row += 1
        return [row, pixels]
```

## 841. Keys and Rooms

### Solution 1: iterative dfs + stack + memoized

```py
class Solution:
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        n = len(rooms)
        stack = [0]
        vis = [0]*n
        vis[0] = 1
        while stack:
            i = stack.pop()
            n -= 1
            for key in rooms[i]:
                if vis[key]: continue
                stack.append(key)
                vis[key] = 1
        return n == 0
```

## 934. Shortest Bridge

### Solution 1:  queue + iterative + dfs + bfs + memoized

```py
class Solution:
    def shortestBridge(self, grid: List[List[int]]) -> int:
        n = len(grid)
        water, land, baseIsland = 0, 1, 2
        in_bounds = lambda r, c: 0 <= r < n and 0 <= c < n
        neighborhood = lambda r, c: [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
        seen = set()
        queue = deque()
        def fill(r: int, c: int):
            stack = [(r, c)]
            seen.add((r, c))
            while stack:
                r, c = stack.pop()
                grid[r][c] = baseIsland
                queue.append((r, c, 0))
                for nr, nc in neighborhood(r, c):
                    if not in_bounds(nr, nc) or (nr, nc) in seen or grid[nr][nc] == water: continue
                    stack.append((nr, nc))
                    seen.add((nr, nc))
        for r, c in product(range(n), repeat = 2):
            if grid[r][c] == land:
                fill(r, c)
                break
        while queue:
            r, c, dist = queue.popleft()
            if grid[r][c] == land: return dist - 1
            for nr, nc in neighborhood(r, c):
                if not in_bounds(nr, nc) or (nr, nc) in seen: continue
                queue.append((nr, nc, dist + 1))
                seen.add((nr, nc))
        return -1
```

## 417. Pacific Atlantic Water Flow

### Solution 1:  Find all the cells that can flow into atlantic and pacific ocean separately + dfs + set intersection + intersectin of what can reach atlantic and pacific gives the result of what can flow into both oceans

```py
class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        R, C = len(heights), len(heights[0])
        pacific = [(r, c) for r, c in product(range(R), range(C)) if r == 0 or c == 0]
        atlantic = [(r, c) for r, c in product(range(R), range(C)) if r == R-1 or c == C-1]
        pacificReach = set(pacific)
        atlanticReach = set(atlantic)
        in_bounds = lambda r, c: 0 <= r < R and 0 <= c < C
        neighborhood = lambda r, c: [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
        def dfs(stack: List[Tuple[int, int]], visited: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
            while stack:
                r, c = stack.pop()
                for nr, nc in neighborhood(r, c):
                    if not in_bounds(nr, nc) or heights[nr][nc] < heights[r][c] or (nr, nc) in visited: continue
                    visited.add((nr, nc))
                    stack.append((nr, nc))
            return visited
        return dfs(pacific, pacificReach) & dfs(atlantic, atlanticReach)
```

## 2490. Circular Sentence

### Solution 1:  modular arithmetic + string to array with space delimiter

```py
class Solution:
    def isCircularSentence(self, sentence: str) -> bool:
        arr = sentence.split()
        n = len(arr)
        for i in range(1, n+1):
            if arr[i%n][0] != arr[i-1][-1]: return False
        return True
```

## 2491. Divide Players Into Teams of Equal Skill

### Solution 1:  two pointer + sort + greedy pair strongest with weakest player + O(nlogn) time

```py
class Solution:
    def dividePlayers(self, skill: List[int]) -> int:
        skill.sort()
        n = len(skill)
        expectedSkill = 2*sum(skill)//n
        res = 0
        for i in range(n//2):
            cur = skill[i] + skill[~i]
            if cur != expectedSkill: return -1
            res += skill[i]*skill[~i]
        return res
```

### Solution 2:  counter + expected skill value + checking that the skill and the complement skill value have equal number of occurrences + O(n) time

```py
class Solution:
    def dividePlayers(self, skill: List[int]) -> int:
        n = len(skill)
        expected = 2*sum(skill)//n
        res = 0
        counts = Counter(skill)
        for key, vals in counts.items():
            if vals != counts[expected-key]: return -1
            res += vals*key*(expected-key)
        return res//2
```

## 2492. Minimum Score of a Path Between Two Cities

### Solution 1: modified union find + compute minimum score in each connected graph

```py
class UnionFind:
    def __init__(self):
        self.size = dict()
        self.parent = dict()
        self.minRoad = dict()
    
    def find(self,i: int, w: int) -> int:
        if i not in self.parent:
            self.size[i] = 1
            self.parent[i] = i
            self.minRoad[i] = w
        while i != self.parent[i]:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self,i: int, j: int, w: int) -> bool:
        i, j = self.find(i, w), self.find(j, w)
        self.minRoad[i] = min(self.minRoad[i], self.minRoad[j], w)
        self.minRoad[j] = self.minRoad[i]
        if i!=j:
            if self.size[i] < self.size[j]:
                i,j=j,i
            self.parent[j] = i
            self.size[i] += self.size[j]
            return True
        return False
    
    @property
    def root_count(self):
        return sum(node == self.find(node) for node in self.parent)

    def __repr__(self) -> str:
        return f'minRoad: {self.minRoad}, parents: {[(i+1, parent) for i, parent in enumerate(self.parent)]}, sizes: {self.size}'
    
class Solution:
    def minScore(self, n: int, roads: List[List[int]]) -> int:
        m = len(roads)
        dsu = UnionFind()
        for u, v, w in roads:
            dsu.union(u, v, w)
        return min(dsu.minRoad[dsu.find(1, inf)], dsu.minRoad[dsu.find(n, inf)])
```

### Solution 2:  bfs from city 1 + store the minimum score for all the cities connected to city 1 + 1 connected graph

```py
class Solution:
    def minScore(self, n: int, roads: List[List[int]]) -> int:
        adj_list = defaultdict(list)
        for u, v, w in roads:
            adj_list[u].append((v, w))
            adj_list[v].append((u, w))
        queue = deque([1])
        vis = set()
        res = inf
        while queue:
            node = queue.popleft()
            for nei_node, nei_w in adj_list[node]:
                res = min(res, nei_w)
                if nei_node in vis: continue
                vis.add(nei_node)
                queue.append((nei_node))
        return res
```

## 2493. Divide Nodes Into the Maximum Number of Groups

### Solution 1:  all pairs shortest path problem with bfs (no weights) + bipartite graph if no odd length cycles + dfs for detecting odd length cycles with coloring

```py

```

## 1319. Number of Operations to Make Network Connected

### Solution 1:  Union find + You need at least n-1 edges to connected all the nodes + It takes one edge to merge two connected graphs

```py
class UnionFind:
    def __init__(self,n: int):
        self.size = [1]*n
        self.parent = list(range(n))
    
    def find(self,i: int) -> int:
        if i==self.parent[i]:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self,i: int,j: int) -> bool:
        i, j = self.find(i), self.find(j)
        if i!=j:
            if self.size[i] < self.size[j]:
                i,j=j,i
            self.parent[j] = i
            self.size[i] += self.size[j]
            return True
        return False
    
    @property
    def root_count(self):
        return sum(node == self.find(node) for node in range(len(self.parent)))
    def __repr__(self) -> str:
        return f'parents: {[(i, parent) for i, parent in enumerate(self.parent)]}, sizes: {self.size}'
    
class Solution:
    def makeConnected(self, n: int, connections: List[List[int]]) -> int:
        dsu = UnionFind(n)
        cnt = sum([1 for u, v in connections if not dsu.union(u, v)])
        needed = dsu.root_count - 1
        return needed if needed <= cnt else -1
```

### Solution 2: count connected graphs with dfs to mark islands as visited + count number of islands

```py
class Solution:
    def makeConnected(self, n: int, connections: List[List[int]]) -> int:
        if len(connections) < n - 1: return -1
        adj_list = [[] for _ in range(n)]
        for u, v in connections:
            adj_list[u].append(v)
            adj_list[v].append(u)
            
        def dfs(node: int) -> None:
            if vis[node]: return
            vis[node] = 1
            for nei_node in adj_list[node]:
                dfs(nei_node)
                
        components = 0
        vis = [0]*n
        for i in range(n):
            if vis[i]: continue
            dfs(i)
            components += 1
        return components - 1
```

## 876. Middle of the Linked List

### Solution 1:  slow and fast pointer + linked list

```py
class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow
```

## 328. Odd Even Linked List

### Solution 1:  two pointers + linked list + swapping even and odd

```py
class Solution:
    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head: return head
        odd, even = head, head.next
        evenHead = even
        while even and even.next:
            odd.next = even.next
            odd = odd.next
            even.next = odd.next
            even = even.next
        odd.next = evenHead
        return head
```

## 938. Range Sum of BST

### Solution 1:  recursion + binary search tree

```py
class Solution:
    def rangeSumBST(self, root: Optional[TreeNode], low: int, high: int) -> int:
        if not root: return 0
        res = root.val if low <= root.val <= high else 0
        if root.val < high:
            res += self.rangeSumBST(root.right, low, high)
        if root.val > low:
            res += self.rangeSumBST(root.left, low, high)
        return res 
```

## 872. Leaf-Similar Trees

### Solution 1: recursion + generator + zip_longest

```py
class Solution:
    def dfs(self, root):
        if not root.left and not root.right: yield root.val
        if root.left:
            yield from self.dfs(root.left)
        if root.right:
            yield from self.dfs(root.right)
    def leafSimilar(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        return all(itertools.starmap(operator.eq, zip_longest(self.dfs(root1), self.dfs(root2))))
```

## 1026. Maximum Difference Between Node and Ancestor

### Solution 1:  postorder recursion with maximus and minimus of the subtrees

```py
class Solution:
    def maxAncestorDiff(self, root: Optional[TreeNode]) -> int:
        
        self.max_diff = -inf
        def maximus(node):
            res = -inf
            if node.left:
                res = max(res, maximus(node.left))
            if node.right:
                res = max(res, maximus(node.right))
            if res != -inf:
                self.max_diff = max(self.max_diff, abs(res - node.val))
            return max(node.val, res)
        maximus(root)

        def minimus(node):
            res = inf 
            if node.left:
                res = min(res, minimus(node.left))
            if node.right:
                res = min(res, minimus(node.right))
            if res != inf:
                self.max_diff = max(self.max_diff, abs(res - node.val))
            return min(node.val, res)
        minimus(root)
        
        return self.max_diff
```

```py
class Solution:
    def maxAncestorDiff(self, root: Optional[TreeNode], min_val = math.inf, max_val = -math.inf) -> int:
        if not root: return abs(max_val - min_val)
        nmin, nmax = min(min_val, root.val), max(max_val, root.val)
        return max(self.maxAncestorDiff(root.left, nmin, nmax), self.maxAncestorDiff(root.right, nmin, nmax))
```

## 1339. Maximum Product of Splitted Binary Tree

### Solution 1:  dfs + postorder traversal

```py
class Solution:
    def getSum(self, root: Optional[TreeNode]) -> int:
        if not root: return 0
        return root.val + self.getSum(root.left) + self.getSum(root.right)
    def maxProduct(self, root: Optional[TreeNode]) -> int:
        total = self.getSum(root)
        self.res = 0
        mod = int(1e9)+7
        def dfs(node: Optional[TreeNode]) -> int:
            if not node: return 0
            subSum = dfs(node.left) + dfs(node.right) + node.val
            self.res = max(self.res, subSum*(total-subSum))
            return subSum
        dfs(root)
        return self.res%mod
```

## 2496. Maximum Value of a String in an Array

### Solution 1:  max function + set to find if any characters is lowercase alphabet + list comprehension

```py
class Solution:
    def maximumValue(self, strs: List[str]) -> int:
        return max([len(s) if set(string.ascii_lowercase)&set(s) else int(s) for s in strs])
```

```py
class Solution:
    def maximumValue(self, strs: List[str]) -> int:
        return max([len(s) if any(ch in string.ascii_lowercase for ch in s)else int(s) for s in strs])
```

## 2497. Maximum Star Sum of a Graph

### Solution 1:  maxheap + adjacency list

```py
class Solution:
    def maxStarSum(self, vals: List[int], edges: List[List[int]], k: int) -> int:
        n = len(vals)
        adj_list = [[] for _ in range(n)]
        for u, v in edges:
            heappush(adj_list[u], (-vals[v], v))
            heappush(adj_list[v], (-vals[u], u))
        res = -inf
        for i in range(n):
            cur = vals[i]
            res = max(res, cur)
            for _ in range(k):
                if not adj_list[i]: break
                nv, _ = heappop(adj_list[i])
                cur -= nv
                res = max(res, cur)
        return res
```

### Solution 2: compute sum star for each one + adjacency list of index with set (no duplicate) + compute sorted of values for neighbor of k elements for each star + take max

```py
class Solution:
    def maxStarSum(self, vals: List[int], edges: List[List[int]], k: int) -> int:
        n = len(vals)
        adj_list = defaultdict(set)
        for u, v in edges:
            if vals[u] > 0:
                adj_list[v].add(u)
            if vals[v] > 0:
                adj_list[u].add(v)
        star = -inf
        for i, v in enumerate(vals):
            curStar = v + sum(sorted(map(lambda j: vals[j], adj_list[i]), reverse = True)[:k])
            star = max(star, curStar)
        return star
```

## 2498. Frog Jump II

### Solution 1:  greedy + binary search 

```py
class Solution:
    def maxJump(self, stones: List[int]) -> int:
        n = len(stones)
        left, right = 1, stones[-1] - stones[0]
        def possible(target: int) -> bool:
            vis = [0]*n
            vis[0] = 1
            pos = 0
            for i in range(1, n):
                if stones[i] - pos > target:
                    if vis[i-1]: return False
                    pos = stones[i-1]
                    vis[i-1] = 1
            if pos - stones[-1] > target: return False # couldn't reach last stone
            # backwards pass now starting from last to first stone
            pos = stones[-1]
            for i in reversed(range(n-1)):
                if not vis[i]:
                    if pos - stones[i] > target: return False
                    pos = stones[i]
            return pos - stones[0] <= target
        while left < right:
            mid = (left + right) >> 1
            res = possible(mid)
            if res:
                right = mid
            else:
                left = mid + 1
        return left
```

### Solution 2: greedy + always skip one step on both trips + optimal way to minimize the difference + jump to each rock in both directions

```py
class Solution:
    def maxJump(self, stones: List[int]) -> int:
        return max([stones[1] - stones[0]] + [stones[i] - stones[i-2] for i in range(2, len(stones))], default = 0)
```

## 2499. Minimum Total Cost to Make Arrays Unequal

### Solution 1:  greedy + the max occurring value for index that need be swapped, if it occurrs more than half the index need swap then will need to look outside the necessary index to swap to swap with an index that was good + But can only do it if the nums1 and nums2 are not equal to that max occurring value, cause else it is not helpful to swap with it. 

```py
class Solution:
    def minimumTotalCost(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        freq = Counter()
        res = swapCnt = maxFreqVal = 0
        for i, (n1, n2) in enumerate(zip(nums1, nums2)):
            if n1 != n2: continue
            swapCnt += 1
            res += i
            freq[n1] += 1
            maxFreqVal = max((maxFreqVal, n1), key = lambda num: freq[num])
        i = 0
        while freq[maxFreqVal] > swapCnt//2 and i < n:
            if nums1[i] != nums2[i] and maxFreqVal not in (nums1[i], nums2[i]):
                res += i
                swapCnt += 1
            i += 1
        return res if freq[maxFreqVal] <= swapCnt//2 else -1
```

## 2500. Delete Greatest Value in Each Row

### Solution 1:  sort

```py
class Solution:
    def deleteGreatestValue(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        ans = [0]*C
        for row in grid:
            row.sort()
            for c in range(C):
                ans[c] = max(ans[c], row[c])
        return sum(ans)
```

## 2501. Longest Square Streak in an Array

### Solution 1:  counter + longest streak is 5 elements

```py
class Solution:
    def longestSquareStreak(self, nums: List[int]) -> int:
        res = -1
        nums.sort()
        n = len(nums)
        freq = Counter()
        for num in nums:
            root = int(math.sqrt(num))
            if root*root == num:
                freq[num] = freq[root] + 1
            else:
                freq[num] = 1
        res = freq.most_common()[0][1]
        return res if res > 1 else -1
```

## 2504. Concatenate the Name and the Profession

### Solution 1:  concat + left + order by

```py
select
    person_id,
    concat(name, '(', left(profession, 1), ')') as name
from Person
order by person_id desc
```

## 2502. Design Memory Allocator

### Solution 1:  dictionary + array + brute force

```py
class Allocator:

    def __init__(self, n: int):
        self.used = defaultdict(list)
        self.arr = [0]*n

    def allocate(self, size: int, mID: int) -> int:
        curFree = 0
        for i, v in enumerate(self.arr):
            if v == 0:
                curFree += 1
            else:
                curFree = 0
            if curFree == size:
                for j in range(i-size+1, i+1):
                    self.arr[j] = mID
                    self.used[mID].append(j)
                return i - size + 1
        return -1

    def free(self, mID: int) -> int:
        res = len(self.used[mID])
        for i in self.used[mID]:
            self.arr[i] = 0
        self.used[mID].clear()
        return res
```
SS
## 2503. Maximum Number of Points From Grid Queries

### Solution 1:  minheap to traverse graph from smallest to largest + offline query + sort queries + minheap bfs + O(RCLog(RC) + kLog(k)) time

```py
class Solution:
    def maxPoints(self, grid: List[List[int]], queries: List[int]) -> List[int]:
        R, C = len(grid), len(grid[0])
        k = len(queries)
        queries = sorted([(q, i) for i, q in enumerate(queries)])
        minheap = [(grid[0][0], 0, 0)]
        vis = set([(0,0)])
        q = 0
        steps = 0
        
        ans = [0]*k
        neighborhood = lambda r, c: ((r+1, c), (r-1,c), (r,c+1), (r,c-1))
        in_bounds = lambda r, c: 0<=r<R and 0<=c<C
        while minheap:
            val, r, c = heappop(minheap)
            while q < k and val >= queries[q][0]:
                ans[queries[q][1]] = steps
                q += 1
            for nr, nc in neighborhood(r, c):
                if not in_bounds(nr, nc) or (nr, nc) in vis: continue
                vis.add((nr, nc))
                heappush(minheap, (grid[nr][nc], nr, nc))
            steps += 1
        while q < k:
            ans[queries[q][1]] = steps
            q += 1
        return ans
        
```

## 1066. Campus Bikes II

### Solution 1: recursive dynamic programming with bitmask + O(n * m * 2^m) time

```py
class Solution:
    def assignBikes(self, workers: List[List[int]], bikes: List[List[int]]) -> int:
        n, m = len(workers), len(bikes)
        manhattan = lambda x1, y1, x2, y2: abs(x1-x2) + abs(y1-y2)
        # (bike mask, distance) The bitmask to represent the assigned bikes and the total distance
        @cache
        def dp(worker: int, bike: int, bike_mask: int) -> int:
            if worker == n == bike_mask.bit_count(): return 0
            if worker == n or bike == m: return inf
            take = dp(worker + 1, 0, bike_mask | (1<<bike)) if bike_mask & (1 << bike) == 0 else inf
            skip = dp(worker, bike + 1, bike_mask)
            return min(take + manhattan(*workers[worker], *bikes[bike]), skip)
        return dp(0, 0, 0)
```

```py
class Solution:
    def assignBikes(self, workers: List[List[int]], bikes: List[List[int]]) -> int:
        n, m = len(workers), len(bikes)
        manhattan = lambda x1, y1, x2, y2: abs(x1-x2) + abs(y1-y2)
        # (bike mask, distance) The bitmask to represent the assigned bikes and the total distance
        memo = {}
        states = deque([(0, 0)])
        for i, (x, y) in enumerate(workers):
            sz = len(states)
            for _ in range(sz):
                bike_mask, dist = states.popleft()
                for j, (x1, y1) in enumerate(bikes):
                    nmask = bike_mask | (1<<j)
                    if nmask == bike_mask: continue
                    ndist = dist + manhattan(x, y, x1, y1)
                    if nmask in memo and ndist >= memo[nmask]: continue
                    memo[nmask] = ndist
                    states.append((nmask, ndist))
        return min(dist for _, dist in states)
```

```py
class Solution:
    def assignBikes(self, workers: List[List[int]], bikes: List[List[int]]) -> int:
        n, m = len(workers), len(bikes)
        manhattan = lambda x1, y1, x2, y2: abs(x1-x2) + abs(y1-y2)
        memo = {}
        states = [(0, 0, 0)] # (dist, bike_mask, worker index)
        while states:
            dist, bike_mask, worker = heappop(states)
            if worker == n: return dist
            for i in range(m):
                nmask = bike_mask | (1<<i)
                if nmask == bike_mask: continue
                ndist = dist + manhattan(*workers[worker], *bikes[i]) 
                if nmask in memo and ndist >= memo[nmask]: continue
                memo[nmask] = ndist
                heappush(states, (ndist, nmask, worker + 1))
        return -1
```

```py
class Solution:
    def assignBikes(self, workers: List[List[int]], bikes: List[List[int]]) -> int:
        n, m = len(workers), len(bikes)
        manhattan = lambda x1, y1, x2, y2: abs(x1-x2) + abs(y1-y2)
        vis = set()
        states = [(0, 0, 0)] # (dist, bike_mask, worker index)
        while states:
            dist, bike_mask, worker = heappop(states)
            if (worker, bike_mask) in vis: continue
            vis.add((worker, bike_mask))
            if worker == n: return dist
            for i in range(m):
                nmask = bike_mask | (1<<i)
                if nmask == bike_mask: continue
                ndist = dist + manhattan(*workers[worker], *bikes[i]) 
                heappush(states, (ndist, nmask, worker + 1))
        return -1
```

## 1971. Find if Path Exists in Graph

### Solution 1:  bfs + adjacency list + graph algorithm

```py
class Solution:
    def validPath(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        m = len(edges)
        queue = deque([source])
        vis = set([source])
        adj_list = [[] for _ in range(n)]
        for u, v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
        while queue:
            node = queue.popleft()
            if node == destination: return True
            for nei in adj_list[node]:
                if nei in vis: continue
                vis.add(nei)
                queue.append(nei)
        return False
```

## 790. Domino and Tromino Tiling

### Solution 1:  recursive dp + partial and full tiling + O(n) time

```py
class Solution:
    def numTilings(self, n: int) -> int:
        mod = int(1e9) + 7
        @cache
        def full(i: int) -> int:
            return i if i <= 2 else (full(i-2) + full(i-1) + partial(i-1))%mod
        @cache 
        def partial(i: int) -> int:
            return i if i == 2 else (partial(i-1) + 2*full(i-2))%mod
        return full(n)
```

### Solution 2:  modular arithmetic + numpy linalg.matrix_power function + transition matrix + O(logn) time

```py
import numpy as np
class Solution:
    # A = TB linear algebra equation 
    def numTilings(self, n: int) -> int:
        MOD = int(1e9+7)
        B = np.array([[1,1,0]]).T
        T = np.linalg.matrix_power(np.array([[2,0,1], [1,0,0], [0, 1,0]], dtype='object'), n-1)
        A = np.matmul(T, B)
        return A[0,0] % MOD
```

### Solution 3: Iterative Dynamic Programming via recurrence relation observed via brute force. 

```c++
const int MOD = 1e9+7;
int numTilings(int n) {
    if (n<=2) {
        return n==1 ? 1 : 2;
    }
    vector<int> dp(n+1,0);
    dp[0]=1; dp[1]=1, dp[2]=2;
    for (int i=3;i<=n;i++) {
        dp[i] = ((2*dp[i-1])%MOD+dp[i-3])%MOD;
    }
    return dp.back();
}
```

### Solution 4: Only the three previous values are needed, so we can use dynamic programming with O(1) space.

```c++
const int MOD = 1e9+7;
int numTilings(int n) {
    if (n<=2) {
        return n==1 ? 1 : 2;
    }
    int a = 1, b = 1, c = 2;
    for (int i=3;i<=n;i++) {
        int tmp = c;
        c = ((2*c)%MOD + a)%MOD;
        a = b;
        b = tmp;
    }
    return c;
}
```

### Solution 5:  it looks like you can use modular exponentiation to solve this in O(log(n)) for further speedup if necessary. 

```c++
const int MOD = 1e9 + 7;
struct Matrix {
    int numRows, numCols;
    vector<vector<int>> M;
    // initialize the 2-dimensional array representation for the matrix with 
    // a given value. 
    void init(int r, int c, int val) {
        numRows = r, numCols = c;
        M.resize(r);
        for (int i = 0;i<r;i++) {
            M[i].assign(c, val);
        }
    }
    // neutral matrix is just one's along the main diagonal (identity matrix)
    void neutral(int r, int c) {
        numRows = r, numCols = c;
        M.resize(r);
        for (int i = 0;i<r;i++) {
            M[i].assign(c, 0);
        }
        for (int i = 0;i<r;i++) {
            for (int j = 0; j < c;j++) {
                if (i==j) {
                    M[i][j]=1;
                }
            }
        }

    }
    // Set's a pair of coordinates on the matrix with the specified value, works for a transition matrix
    // where you need ones in places. 
    void set(vector<pair<int,int>>& locs, int val) {
        int r, c;
        for (auto loc : locs) {
            tie(r, c) = loc;
            M[r][c] = val;
        }
    }
    void setSingle(int r, int c, int val) {
        M[r][c] = val;
    }
    // this matrix times another matrix. 
    void operator*=(const Matrix& B) {
        int RB = B.M.size(), CB = B.M[0].size();
        vector<vector<int>> result(numRows, vector<int>(CB, 0));
        for (int i = 0;i < numRows;i++) {
            for (int j = 0;j < CB;j++) {
                int sum = 0;
                for (int k = 0;k < RB;k++) {
                    sum = (sum + ((long long)M[i][k]*B.M[k][j])%MOD)%MOD;
                }
                result[i][j] = sum;
            }
        }
        numRows = numCols, numCols = RB;
        swap(M, result);
    }

    void transpose() {
        int R = numCols, C = numRows;
        vector<vector<int>> matrix(R, vector<int>(C,0));
        for (int i = 0;i < numRows;i++) {
            for (int j = 0;j < numCols;j++) {
                matrix[j][i]=M[i][j];
            }
        }
        swap(numRows,numCols); // transpose swaps the rows and columns
        swap(M,matrix); // swap these two
    }
    // Method to convert a row and column to a unique integer that identifies a row, column combination
    // that can be used in hashing
    int getId(int row, int col) {
        return numRows*row+col;
    }

};

Matrix operator*(const Matrix& A, const Matrix& B) {
    int RA = A.M.size(), CA = A.M[0].size(), RB = B.M.size(), CB = B.M[0].size();
    if (CA!=RB) {
        printf("CA and RB are not equal\n");
        return A;
    }
    Matrix result;
    result.init(RA,CB,0);
    for (int i = 0;i < RA;i++) {
        for (int j = 0; j < CB; j++) {
            int sum = 0;
            for (int k = 0;k < RB;k++) {
                sum = (sum+((long long)A.M[i][k]*B.M[k][j])%MOD)%MOD;
            }
            result.M[i][j]=sum;
        }
    }
    return result;
}

// matrix exponentiation
Matrix matrix_power(Matrix& A, int b) {
    Matrix result;
    result.neutral(A.numRows, A.numCols);
    while (b > 0) {
        if (b % 2 == 1) {
            result = (result*A);
        }
        A *= A;
        b /= 2;
    }
    return result;
}
class Solution {
public:
    const int MOD = 1e9+7;
    int N;
    int numTilings(int n) {
        Matrix transition, base;
        transition.init(3, 3, 0);
        base.init(3, 1, 1);
        vector<pair<int,int>> ones = {{0,2}, {1,0},{2,1}};
        transition.set(ones, 1);
        transition.setSingle(0,0,2);
        base.setSingle(2,0,0);
        Matrix expo = matrix_power(transition, n-1); // exponentiated transition matrix
        Matrix result = expo*base;
        return result.M[0][0];
    }
};
```

## 2506. Count Pairs Of Similar Strings

### Solution 1:  similar if sets are equal between words + implies words have same characters

```py
class Solution:
    def similarPairs(self, words: List[str]) -> int:
        n = len(words)
        res = 0
        for i in range(n):
            for j in range(i+1, n):
                s1, s2 = set(words[i]), set(words[j])
                res += (s1 == s2)
        return res
```

## 2507. Smallest Value After Replacing With Sum of Prime Factors

### Solution 1:  prime sieve + sum of prime factors

```py
class Solution:
    def smallestValue(self, n: int) -> int:
        def prime_sieve(lim):
            sieve,primes = [[] for _ in range(lim + 1)], set()
            for integer in range(2,lim + 1):
                if not len(sieve[integer]):
                    primes.add(integer)
                    for possibly_divisible_integer in range(integer,lim+1,integer):
                        current_integer = possibly_divisible_integer
                        while not current_integer%integer:
                            sieve[possibly_divisible_integer].append(integer)
                            current_integer //= integer
            return sieve
        sieve = prime_sieve(n)
        while len(sieve[n]) > 1:
            nxt = sum(sieve[n])
            if nxt == n: break
            n = nxt
        return n
```

### Solution 2:  find all prime factors for each number + early termination in the loop + sum of prime factors + sqrt(n) time about

```py
class Solution:
    def smallestValue(self, n: int) -> int:
        def sopf(n: int) -> int:
            sum_ = 0
            for i in range(2, int(math.sqrt(n)) + 1):
                if n < i: break
                while n%i == 0:
                    sum_ += i
                    n //= i
            return sum_ + (n if n > 1 else 0)

        def pivot_sopf(n: int) -> int:
            while n != (n := sopf(n)): pass
            return n
        return pivot_sopf(n)
```

## 2508. Add Edges to Make Degrees of All Nodes Even

### Solution 1:  Since can only add at most 2 edges, you can only consider the cases of 0, 2, 4 odd degree edges.  if 0 it is trivially true + if 2 it is true if you can find an external node to connect them to if they are adjacent can use just intersection of adjacency list needs to be less than n + case 4 just requires that you can find two pairs of non adjacent nodes to connect to each other and thus using the 2 edges you are given

```py
class Solution:
    def isPossible(self, n: int, edges: List[List[int]]) -> bool:
        degrees = [0]*(n+1)
        adj_list = [set() for _ in range(n+1)]
        for u, v in edges:
            adj_list[u].add(v)
            adj_list[v].add(u)
            degrees[u] += 1
            degrees[v] += 1
        odd_nodes = []
        for i, deg in enumerate(degrees):
            if deg&1:
                odd_nodes.append(i)
        def non_adjacent(pairs: List[List[int]]) -> bool:
            (x1, y1), (x2, y2) = map(lambda x: x, pairs)
            return False if x1 in adj_list[y1] or x2 in adj_list[y2] else True
        m = len(odd_nodes)
        if m == 2:
            x, y = odd_nodes
            return len(adj_list[x] | adj_list[y]) < n
        if m == 4:
            for mask in range(1, 1 << 4):
                if mask.bit_count() != 2: continue
                pairs = [[] for _ in range(2)]
                for i in range(4):
                    if (mask>>i)&1:
                        pairs[1].append(odd_nodes[i])
                    else:
                        pairs[0].append(odd_nodes[i])
                if non_adjacent(pairs): return True
        return m == 0
```

## 2509. Cycle Length Queries in a Tree

### Solution 1:  online queries + binary jumping + size of tree is at most 30 in depth + two pointers + the parent_node = current_node//2 + at the point that both pointers meet at same node that will give the depth for both nodes and so you can get the cycle length + O(nlogn) time since it is a complete binary tree + could think of it as you are finding the lowest common ancestor + or think of it like sum of distances to LCA

```py
class Solution:
    def cycleLengthQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        m = len(queries)
        def height(u: int, v: int) -> int:
            du = dv = 0
            while  u != v:
                if u > v:
                    u >>= 1
                    du += 1
                if v > u:
                    v >>= 1
                    dv += 1
            return du + dv
        answer = [0]*m
        for i, (u, v) in enumerate(queries):
            answer[i] = height(u, v) + 1
        return answer
```

## 1962. Remove Stones to Minimize the Total

### Solution 1:  maxheap + map

```py
class Solution:
    def minStoneSum(self, piles: List[int], k: int) -> int:
        heapify(maxheap := list(map(lambda x: -x, piles)))
        for _ in range(k):
            x = abs(heappop(maxheap))
            x -= x//2
            heappush(maxheap, -x)
        return sum(map(abs, maxheap))
```

## 1834. Single-Threaded CPU

### Solution 1:  minheap + sort + greedy + O(nlogn) time + using the minheap to store the tasks that are currently enqueued and to have them sorted by smallest processing time and then index

```py
class Solution:
    def getOrder(self, tasks: List[List[int]]) -> List[int]:
        n = len(tasks)
        arr = sorted([(enqueue, process, i) for i, (enqueue, process) in enumerate(tasks)])
        i = time = 0
        minheap = []
        answer = []
        while minheap or i < n:
            if (i < n and arr[i][0] <= time) or not minheap:
                cand_time, process, index = arr[i]
                time = max(time, cand_time)
                heappush(minheap, (process, index))
                i += 1
            else:
                x, index = heappop(minheap)
                time += x
                answer.append(index)
        return answer
```

## 520. Detect Capital

### Solution 1: check true conditions

```py
class Solution:
    def detectCapitalUse(self, word: str) -> bool:
        return word.upper() == word or word.lower() == word or f'{word[:1].upper()}{word[1:].lower()}' == word
```

## Solution 2: check true conditions

```c++
bool detectCapitalUse(string word) {
    return all_of(word.begin(),word.end(),[](const auto& a) { return isupper(a);}) || (all_of(word.begin(),word.end(),[](const auto& a) {return islower(a);})) || (isupper(word[0]) && all_of(word.begin()+1,word.end(),[](const auto& a) {return islower(a);}));
}
```

## 944. Delete Columns to Make Sorted

### Solution 1: array to mark columns to delete + comparison in loop through array + O(RC) time + O(C) extra space

```py
class Solution:
    def minDeletionSize(self, strs: List[str]) -> int:
        R, C = len(strs), len(strs[0])
        mark = [False]*C # mark for deletion
        for r, c in product(range(1, R), range(C)):
            if strs[r][c] < strs[r-1][c]: mark[c] = True
        return sum(mark)
```

### Solution 2: Alternative approaches + using sum function + zip to create transpose of the input + compare if the sorted list is equal to the original list

```py
class Solution:
    def minDeletionSize(self, strs: List[str]) -> int:
        return sum(1 for row in zip(*strs) if any(x != y for x, y in zip(sorted(row), row)))
```

```py
class Solution:
    def minDeletionSize(self, strs: List[str]) -> int:
        return sum(row > sorted(row) for row in map(list, zip(*strs)))
```

### Solution 3:  Using numpy to create transpose matrix from a python 2d list

```py
import numpy as np
class Solution:
    def minDeletionSize(self, strs: List[str]) -> int:
        matrix = [list(row) for row in strs]
        T_matrix = np.transpose(matrix)
        res = 0
        for row in T_matrix:
            if any(x != y for x, y in zip(sorted(row), row)): res += 1
        return res
```

## 1056. Confusing Number

### Solution 1:  dictionary + integer + convert to list to string to int

```py
class Solution:
    def confusingNumber(self, n: int) -> bool:
        transformations = {0: 0, 1: 1, 6: 9, 8: 8, 9: 6}
        res = []
        num = n
        while n > 0:
            dig = n%10
            if dig not in transformations: return False
            res.append(transformations[dig])
            n //= 10
        return num != int(''.join(map(str, res))) if len(res) > 0 else False
```

## 1833. Maximum Ice Cream Bars

### Solution 1:  counter/bucket sort + O(n + m) time + O(m) extra space where m = max(costs) + take least expensive ice creams first

```py
class Solution:
    def maxIceCream(self, costs: List[int], coins: int) -> int:
        buckets = [0]*(max(costs) + 1)
        for cost in costs:
            buckets[cost] += 1
        res = 0
        for i, cnt in enumerate(buckets):
            if cnt == 0: continue
            take = min(cnt, coins//i)
            if take == 0: break
            coins -= i*take
            res += take
        return res
```

## 2511. Maximum Enemy Forts That Can Be Captured

### Solution 1:

```py

```

## 2512. Reward Top K Students

### Solution 1:

```py

```

## 2513. Minimize the Maximum of Two Arrays

### Solution 1:

```py

```

## 2514. Count Anagrams

### Solution 1:

```py

```

## 2525. Categorize Box According to Criteria

### Solution 1:  case statements + conditionals

```py
class Solution:
    def categorizeBox(self, length: int, width: int, height: int, mass: int) -> str:
        vol = length*width*height
        thres1, thres2, thres3 = 10**4, 10**9, 100
        bulky = length >= thres1 or width >= thres1 or height >= thres1 or vol >= thres2
        heavy = mass >= thres3
        if bulky and heavy: return 'Both'
        if bulky: return 'Bulky'
        if heavy: return 'Heavy'
        return 'Neither'
```

## 2526. Find Consecutive Integers from a Data Stream

### Solution 1:  pointer to the count of consecutive values

```py
class DataStream:

    def __init__(self, value: int, k: int):
        self.value, self.k = value, k
        self.cnt = 0
        

    def consec(self, num: int) -> bool:
        if num != self.value:
            self.cnt = 0
        self.cnt += (num == self.value)
        return self.cnt >= self.k
```

## 2527. Find Xor-Beauty of Array

### Solution 1:  If you check it turns out the result is just the xor of all the elements in the array, because the rest cancel each other + math + bit manipulation

```py
class Solution:
    def xorBeauty(self, nums: List[int]) -> int:
        return reduce(operator.xor, nums)
```

## 2528. Maximize the Minimum Powered City

### Solution 1:  binary search + greedy + sliding window sum  + early termination in the function when it exceeds and will guarantee a false return + sliding window with pivot point and radius + radius sliding window + greedily add additions to the farthest to the right element that will affect the current element + O(nlog(k)) time

```py
class Solution:
    def maxPower(self, stations: List[int], r: int, k: int) -> int:
        n = len(stations)
        init_sum = sum(stations[:r])
        def possible(target: int) -> bool:
            res, window_sum = 0, init_sum
            additions = [0]*n
            for i in range(n):
                # window sum for the elements [i-r, i-r+1, ..., i, ..., i+r-1, i+r]
                if i + r < n:
                    window_sum += stations[i+r]
                need = max(0, target - window_sum)
                res += need
                if res > k: return False
                if need > 0:
                    window_sum += need
                    additions[min(i + r, n - 1)] += need # optimal location to add cause it will affect this one and r more elements that have not been processed
                if i - r >= 0:
                    window_sum -= stations[i-r]
                    window_sum -= additions[i-r]
            return res <= k
        left, right = 0, int(1e11)
        while left < right:
            mid = (left + right + 1) >> 1
            if possible(mid):
                left = mid
            else:
                right = mid - 1
        return left
```

## 2530. Maximal Score After Applying K Operations

### Solution 1:  maxheap + simulation + O(k + nlogn) time

```py
class Solution:
    def maxKelements(self, nums: List[int], k: int) -> int:
        heapify(maxheap := [-num for num in nums])
        res = 0
        for _ in range(k):
            num = abs(heappop(maxheap))
            res += num
            nnum = math.ceil(num/3)
            heappush(maxheap, -nnum)
        return res
```

## 2532. Time to Cross a Bridge

### Solution 1:  simulation + minheap for events + two max heaps for the left and right side of the bridge + (time, location, worker_index) for each state, put a bridge event for each possible instance want to make a decision who to cross the bridge next + O(nlogn + klogk) time + time series problem

```py
class Solution:
    def findCrossingTime(self, n: int, k: int, time: List[List[int]]) -> int:
        old_warehouse, new_warehouse, bridge = range(3)
        heapify(minheap := [(0, bridge, None)])
        heapify(left_bank_queue := [(-ltor-rtol, -i) for i, (ltor, old, rtol, new) in enumerate(time)])
        right_bank_queue = []
        last_worker_reach_left_bank = 0
        bridge_next_available = 0
        remaining_old = n
        efficiency = lambda index: time[index][0] + time[index][2]
        while minheap:
            t, location, worker = heappop(minheap)
            if location == bridge:
                if t < bridge_next_available: # bridge in use
                    pass
                elif right_bank_queue:
                    _, i = map(abs, heappop(right_bank_queue))
                    left_to_right, pick_old, right_to_left, put_new = time[i]
                    bridge_next_available = max(bridge_next_available, t + right_to_left)
                    last_worker_reach_left_bank = max(last_worker_reach_left_bank, t + right_to_left)
                    heappush(minheap, (t + right_to_left, bridge, None))
                    heappush(minheap, (t + right_to_left + put_new, new_warehouse, i))
                    heappush(minheap, (t + right_to_left + put_new, bridge, None))
                elif remaining_old > 0 and left_bank_queue:
                    _, i = map(abs, heappop(left_bank_queue))
                    left_to_right, pick_old, right_to_left, put_new = time[i]
                    bridge_next_available = max(bridge_next_available, t + left_to_right)
                    heappush(minheap, (t + left_to_right, bridge, None))
                    heappush(minheap, (t + left_to_right + pick_old, old_warehouse, i))
                    heappush(minheap, (t + left_to_right + pick_old, bridge, None))
                    remaining_old -= 1
            elif location == old_warehouse:
                heappush(right_bank_queue, (-efficiency(worker), -worker))
            elif location == new_warehouse:
                heappush(left_bank_queue, (-efficiency(worker), -worker))
        return last_worker_reach_left_bank
```

### Solution 2:  simulation + event minheap + left and right maxheap + event class and worker efficiency class + class has custom comparator for it to work in the heap 

```py
class Event:
    def __init__(self, timestamp: int, location: str, worker_index: int = None):
        self.timestamp = timestamp
        self.location = location
        self.worker_index = worker_index
        self.location_priority = ['old_warehouse', 'new_warehouse', 'bridge']

    def __repr__(self) -> str:
        return f'timestamp: {self.timestamp}, location: {self.location}'

    def __lt__(self, other) -> bool:
        if self.timestamp != other.timestamp: return self.timestamp < other.timestamp
        return self.location_priority.index(self.location) < other.location_priority.index(other.location)

class WorkerEfficiency:
    def __init__(self, efficiency: int, worker_index: int):
        self.efficiency = efficiency
        self.worker_index = worker_index

    def __lt__(self, other) -> bool:
        return self.efficiency > other.efficiency if self.efficiency != other.efficiency else self.worker_index > other.worker_index

class Solution:
    def findCrossingTime(self, n: int, k: int, time: List[List[int]]) -> int:
        # LOCATION VARIABLES
        old_warehouse, new_warehouse, bridge = ('old_warehouse', 'new_warehouse', 'bridge')
        # EFFICIENCY FUNCTION
        efficiency = lambda index: time[index][0] + time[index][2]
        # 3 HEAPS
        heapify(minheap := [Event(0, bridge)])
        heapify(left_bank_queue := [WorkerEfficiency(efficiency(i), i) for i in range(k)])
        right_bank_queue = []
        # TRACKING VARIABLES
        last_worker_reach_left_bank = 0
        bridge_next_available = 0
        remaining_old = n
        # SIMULATION TILL NO MORE EVENTS
        while minheap:
            event = heappop(minheap)
            if event.location == bridge:
                if event.timestamp < bridge_next_available: # bridge in use
                    pass
                elif right_bank_queue:
                    worker_eff = heappop(right_bank_queue)
                    i = worker_eff.worker_index
                    left_to_right, pick_old, right_to_left, put_new = time[i]
                    bridge_next_available = max(bridge_next_available, event.timestamp + right_to_left)
                    last_worker_reach_left_bank = max(last_worker_reach_left_bank, event.timestamp + right_to_left)
                    heappush(minheap, Event(event.timestamp + right_to_left, bridge))
                    heappush(minheap, Event(event.timestamp + right_to_left + put_new, new_warehouse, i))
                    heappush(minheap, Event(event.timestamp + right_to_left + put_new, bridge))
                elif remaining_old > 0 and left_bank_queue:
                    worker_eff = heappop(left_bank_queue)
                    i = worker_eff.worker_index
                    left_to_right, pick_old, right_to_left, put_new = time[i]
                    bridge_next_available = max(bridge_next_available, event.timestamp + left_to_right)
                    heappush(minheap, Event(event.timestamp + left_to_right, bridge))
                    heappush(minheap, Event(event.timestamp + left_to_right + pick_old, old_warehouse, i))
                    heappush(minheap, Event(event.timestamp + left_to_right + pick_old, bridge))
                    remaining_old -= 1
            elif event.location == old_warehouse:
                heappush(right_bank_queue, WorkerEfficiency(efficiency(event.worker_index), event.worker_index))
            elif event.location == new_warehouse:
                heappush(left_bank_queue, WorkerEfficiency(efficiency(event.worker_index), event.worker_index))
        return last_worker_reach_left_bank
```

## 2531. Make Number of Distinct Characters Equal

### Solution 1:  frequency array + swap + undo + count of distinct characters + O(26*26) time

```py
class Solution:        
    def isItPossible(self, word1: str, word2: str) -> bool:
        freq1, freq2 = Counter(word1), Counter(word2)
        for ch1, ch2 in product(freq1, freq2):
            # SWAP THE TWO CHARACTERS
            freq1[ch1] -= 1
            freq2[ch1] += 1
            freq2[ch2] -= 1
            freq1[ch2] += 1
            # THE SUM OF DISTINCT CHARACTERS
            sum1, sum2 = sum(1 for v in freq1.values() if v > 0), sum(1 for v in freq2.values() if v > 0)
            # UNDO SWAPS
            freq1[ch1] += 1
            freq2[ch1] -= 1
            freq2[ch2] += 1
            freq1[ch2] -= 1
            # IF THE SUM OF DISTINCT CHARACTERS IS EQUAL
            if sum1 == sum2: return True
        return False
```

## 2529. Maximum Count of Positive Integer and Negative Integer

### Solution 1:  binary search + O(logn) time

```py
class Solution:
    def maximumCount(self, nums: List[int]) -> int:
        left = bisect.bisect_left(nums, 0)
        right = bisect.bisect_right(nums, 0)
        return max(left, len(nums) - right)
```

## 149. Max Points on a Line

### Solution 1:  math + geometry + arc tangent + slope + O(n^2) time + hash table

```py
class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        res = 0
        for x1, y1 in points:
            cnt = Counter()
            for x2, y2 in points:
                if x1 == x2 and y1 == y2: continue
                arctan = math.atan2(y2 - y1, x2 - x1)
                cnt[arctan] += 1
                res = max(res, cnt[arctan])
        return res + 1
```

## 2214. Minimum Health to Beat Game

### Solution 1:  sum + greedily use the armor on the turn you take most damage

```py
class Solution:
    def minimumHealth(self, damage: List[int], armor: int) -> int:
        max_dmg = max(damage)
        min_health = sum(damage) + 1
        return min_health - min(armor, max_dmg)
```

## 2515. Shortest Distance to Target String in a Circular Array

### Solution 1:  visualize the circle and how if you take the distance between two pints on the circle, then there is a second distance which is the entire length of the circle minuse the distance between those two points + O(n) time

```py
class Solution:
    def closetTarget(self, words: List[str], target: str, startIndex: int) -> int:
        n = len(words)
        res = math.inf
        for i in range(n):
            if target == words[i]:
                res = min(res, abs(i - startIndex), n - abs(i - startIndex))
        return res if res < math.inf else -1
```

## 2516. Take K of Each Character From Left and Right

### Solution 1:  sliding window for the middle exclusion part + frequency array + maximize the size of the sliding window + solution is the total length minuse the size of the maximum sliding window

```py
class Solution:
    def takeCharacters(self, s: str, k: int) -> int:
        n = len(s)
        freq = Counter(s)
        if freq['a'] < k or freq['b'] < k or freq['c'] < k: return -1
        left = res = 0
        for right in range(n):
            freq[s[right]] -= 1
            while freq[s[right]] < k:
                freq[s[left]] += 1
                left += 1
            res = max(res, right - left + 1)
        return n - res
```

## 2517. Maximum Tastiness of Candy Basket

### Solution 1:  binary search + sort + greedily pick the candies as soon as you can, and pick the next one that exceeds the current delta or absolute difference needed between two prices + O(nlog(m)) time where m = max(prices)

```py
class Solution:
    def maximumTastiness(self, price: List[int], k: int) -> int:
        price.sort()
        def possible(delta: int) -> bool:
            cnt, last = 0, -math.inf
            for p in price:
                if p - last >= delta:
                    cnt += 1
                    last = p
                if cnt >= k: return True
            return cnt >= k
        left, right = 0, 10**9
        while left < right:
            mid = (left + right + 1) >> 1
            if possible(mid):
                left = mid
            else:
                right = mid - 1
        return left
```

## 2518. Number of Great Partitions

### Solution 1:  0/1 knapsack problem applied to find the count of subsets with sum equal to a capacity + inclusion/exclusion of item with no splitting + count the invalid subsets that are under k and multiple by 2 because each one can represent two pairs

```py
class Solution:
    def countPartitions(self, nums: List[int], k: int) -> int:
        if sum(nums) < 2*k: return 0
        mod = int(1e9) + 7
        n = len(nums)
        # the subproblem is the count of subsets for on the ith item at the jth sum
        dp = [[0]*(k) for _ in range(n+1)] 
        for i in range(n+1):
            dp[i][0] = 1 # i items, 0 sum/capacity 
            # only 1 unique solution which is the empty subset for this subproblem
        # i represents an item, j represents the current sum
        for i, j in product(range(1, n+1), range(1, k)):
            if nums[i-1] > j: # cannot place an item that exceeds the j, so it can only be excluded
                dp[i][j] = dp[i-1][j]
            else: # the subset count up to this sum is going to be if you combine the exclusion of this item added to the inclusion of this item
                dp[i][j] = (dp[i-1][j] + dp[i-1][j-nums[i-1]])%mod
        # count of subsets for when the sum < k, these
        count_invalid_subsets = (2 * sum(dp[-1]))%mod
        total_subsets = 1
        for _ in range(n):
            total_subsets = (total_subsets*2)%mod
        return (total_subsets - count_invalid_subsets + mod)%mod
```

### Solution 2:  Same as above without all the extra mod, just need to perform it at the end. 

```py
class Solution:
    def countPartitions(self, nums: List[int], k: int) -> int:
        mod = int(1e9) + 7
        n = len(nums)
        # the subproblem is the count of subsets for on the ith item at the jth sum
        dp = [[0]*(k) for _ in range(n+1)] 
        for i in range(n+1):
            dp[i][0] = 1 # i items, 0 sum/capacity 
            # only 1 unique solution which is the empty subset for this subproblem
        # i represents an item, j represents the current sum
        for i, j in product(range(1, n+1), range(1, k)):
            if nums[i-1] > j: # cannot place an item that exceeds the j, so it can only be excluded
                dp[i][j] = dp[i-1][j]
            else: # the subset count up to this sum is going to be if you combine the exclusion of this item added to the inclusion of this item
                dp[i][j] = dp[i-1][j] + dp[i-1][j-nums[i-1]]
        # count of subsets for when the sum < k, these
        count_invalid_subsets = 2 * sum(dp[-1])
        return max(0, 2**n - count_invalid_subsets)%mod
```

## 1443. Minimum Time to Collect All Apples in a Tree

### Solution 1:  post order dfs + rooted tree from undirected graph + O(n) time and space

```py
class Solution:
    def minTime(self, n: int, edges: List[List[int]], hasApple: List[bool]) -> int:
        adj_list = [[]*n for _ in range(n)]
        for u, v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
        def dfs(node: int, parent: int) -> int:
            time = 0
            for nei in adj_list[node]:
                if nei == parent: continue
                vertex_time = dfs(nei, node)
                if hasApple[nei] or vertex_time > 0:
                    time += 2 + vertex_time
            return time
        return dfs(0, None)
```

## 1519. Number of Nodes in the Sub-Tree With the Same Label

### Solution 1:  postorder dfs + rooted tree from undirected graph + O(26*n) time and space

```py
class Solution:
    def countSubTrees(self, n: int, edges: List[List[int]], labels: str) -> List[int]:
        adj_list =  [[] for _ in range(n)]
        for u, v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
        self.ans = [1]*n
        unicode = lambda ch: ord(ch) - ord('a')
        def dfs(node: int, parent: int) -> List[int]:
            cnts = [0]*26
            for nei in adj_list[node]:
                if nei == parent: continue
                for i, cnt in enumerate(dfs(nei, node)):
                    cnts[i] += cnt
            cnts[unicode(labels[node])] += 1
            self.ans[node] = cnts[unicode(labels[node])]
            return cnts

        dfs(0, None)
        return self.ans
```

### Solution 2:  Virtual/auxillary trees

```py

```

```cpp
class Solution {
public:
    void DFS1(int u, int prv, vector<vector<int>>& curAdj, vector<vector<int>>& newAdj, unordered_map<char, int>& vir, string& labels, vector<int>& roots) {
        if(vir.find(labels[u]) != vir.end()) newAdj[vir[labels[u]]].push_back(u);
        int prevNode = (vir.find(labels[u]) != vir.end() ? vir[labels[u]] : -1);
        vir[labels[u]] = u;
        for(auto& i: curAdj[u]){
            if(i == prv) continue;
            DFS1(i, u, curAdj, newAdj, vir, labels, roots);
        }
        if(prevNode != -1) vir[labels[u]] = prevNode;
        else {
            vir.erase(labels[u]);
            roots.push_back(u);
        }
    }
    void DFS2(int u, vector<int>& freq, vector<vector<int>>& adj) {
        int cnt = 1;
        for(auto& i: adj[u]) {
            DFS2(i, freq, adj);
            cnt += freq[i];
        }
        freq[u] = cnt;
    }
    vector<int> countSubTrees(int n, vector<vector<int>>& edges, string labels) {
        vector<vector<int>> curAdj(n), adj(n);
        vector<int> roots;
        for(auto& i: edges) {
            curAdj[i[0]].push_back(i[1]);
            curAdj[i[1]].push_back(i[0]);
        }
        unordered_map<char, int> UM;
        DFS1(0, -1, curAdj, adj, UM, labels, roots);
        vector<int> ret(n, 0);
        for(auto& i: roots) DFS2(i, ret, adj);
        return ret;
    }
};
```

## 2520. Count the Digits That Divide a Number

### Solution 1:  sum function + string + divisibility

```py
class Solution:
    def countDigits(self, num: int) -> int:
        return sum(1 for dig in map(int, str(num)) if num%dig == 0)
```

## 2521. Distinct Prime Factors of Product of Array

### Solution 1: prime factorization + O(n*sqrt(max(nums))) time and space

```py
class Solution:
    def distinctPrimeFactors(self, nums: List[int]) -> int:
        res = set()
        def get_factors(x: int) -> int:
            f = 2
            factors = set()
            while x > 1:
                while x % f == 0:
                    x //= f
                    factors.add(f)
                f += 1
            return factors
        for num in nums:
            res.update(get_factors(num))
        return len(res)
```

### Solution 2: prime sieve + precompute the prime factorizations of each integer in range of 1000 + O(mlog(logm) + n) time

```py
class Solution:
    def distinctPrimeFactors(self, nums: List[int]) -> int:
        thres = 1001
        sieve = [set() for _ in range(thres)]
        for integer in range(2, thres):
            if len(sieve[integer]) > 0: continue
            for possibly_divisible_integer in range(integer, thres, integer):
                cur_integer = possibly_divisible_integer
                while cur_integer % integer == 0:
                    sieve[possibly_divisible_integer].add(integer)
                    cur_integer //= integer
        res = set()
        for num in nums:
            res.update(sieve[num])
        return len(res)
```

### Solution 3: optimizations on above approach

```py
class Solution:
    def distinctPrimeFactors(self, nums: List[int]) -> int:
        res = set()
        nums = set(nums)
        thres = max(nums) + 1
        is_prime = [True]*thres
        for integer in range(2, thres):
            if not is_prime[integer]: continue # skips if it is not a prime integer
            for possibly_divisible_integer in range(integer, thres, integer):
                is_prime[possibly_divisible_integer] = False # since this is divisible by a prime integer
                if possibly_divisible_integer in nums: # possibly this integer is in nums and so this integer is a prime integer for it
                    res.add(integer)
        return len(res)
```

## 2522. Partition String Into Substrings With Values at Most K

### Solution 1:  greedy + take longest partition + O(n) time

```py
class Solution:
    def minimumPartition(self, s: str, k: int) -> int:
        n = len(s)
        val = res = 0
        for num in map(int, s):
            val = val*10 + num
            if num > k: return -1
            if val > k:
                res += 1
                val = num
        return res + 1
```

### Solution 2: recursive knapsack problem + O(n*log(k)) time

```py
class Solution:
    def minimumPartition(self, s: str, k: int) -> int:
        n = len(s)
        if any(num > k for num in map(int, s)): return -1
        @cache
        def knapsack(i: int, cur: int) -> int:
            if i == n: return 1
            ncur = cur*10 + int(s[i])
            skip = knapsack(i + 1, ncur) if ncur <= k else math.inf
            take = knapsack(i + 1, int(s[i])) + 1
            return min(skip, take)
        return knapsack(0, 0)
```

## 2523. Closest Prime Numbers in Range

### Solution 1:  prime sieve + early termination + O(nloglogn) time and space

```py
class Solution:
    def closestPrimes(self, left: int, right: int) -> List[int]:
        is_prime, primes = [True]*(right + 1), []
        res = [-1, -1]
        delta = math.inf
        for integer in range(2, right + 1):
            if not is_prime[integer]: continue
            if integer >= left and integer <= right:
                if primes and integer - primes[-1] < delta:
                    delta = integer - primes[-1]
                    res = [primes[-1], integer]
                if delta <= 2: return res
                primes.append(integer)
            for possibly_divisible_integer in range(integer, right + 1, integer):
                is_prime[possibly_divisible_integer] = False
        return res
```

### Solution 2:  twin primes + smallest set of primes except for (2, 3) and occurr frequently in integers smaller than 1,000,000 + O(nsqrt(n)) time

```py
class Solution:
    def closestPrimes(self, left: int, right: int) -> List[int]:
        def is_prime(x: int) -> bool:
            if x < 2: return False
            for i in range(2, int(math.sqrt(x)) + 1):
                if x % i == 0: return False
            return True
        pair = [-1, -1]
        primes = []
        delta = math.inf
        for i in range(left, right + 1):
            if not is_prime(i): continue
            primes.append(i)
            if len(primes) > 1:
                x, y = primes[-2], primes[-1]
                if y - x < delta:
                    pair = [x, y]
                    delta = y - x
                if delta <= 2: break # twin prime (common and smallest difference between primes other than (2, 3))
        return pair
```

## 1533. Find the Index of the Large Integer

### Solution 1:  binary search + make it always even length, with extra element + O(logn) time

```py
class Solution:
    def getIndex(self, reader: 'ArrayReader') -> int:
        left, right = 0, reader.length() - 1
        while left < right:
            mid = (left + right + 1) >> 1
            size = right - left + 1
            check = reader.compareSub(left, mid - 1, mid, right) if size%2 == 0 else reader.compareSub(left, mid - 1, mid, right - 1)
            if check == 1:
                right = mid - 1
            elif check == -1:
                left = mid
            else:
                return right
        return left
```

### Solution 2:  ternary search + break into three segments and compare to figure out which segment has the largest integer + O(logn) time

```py
class Solution:
    def getIndex(self, reader: 'ArrayReader') -> int:
        left, right = 0, reader.length() - 1
        while left + 1 < right:
            mid = (right - left + 1) // 3
            check = reader.compareSub(left, left + mid - 1, left + mid, left + 2*mid - 1)
            if check == 1:
                right = left + mid - 1
            elif check == -1:
                left, right = left + mid, left + 2*mid - 1
            else:
                left = left + 2*mid
        if left != right:
            return left if reader.compareSub(left, left, right, right) == 1 else right
        return left 
```

## 2535. Difference Between Element Sum and Digit Sum of an Array

### Solution 1:  strings + sum

```py
class Solution:
    def differenceOfSum(self, nums: List[int]) -> int:
        s = ds = 0
        for num in nums:
            s += num
            for dig in map(int, str(num)):
                ds += dig
        return abs(s - ds)
```

## 2536. Increment Submatrices by One

### Solution 1:  for each row store prefix sum + store the delta at index, and then that range will be updated + O(n^2 + n*q) time

```py
class Solution:
    def rangeAddQueries(self, n: int, queries: List[List[int]]) -> List[List[int]]:
        mat = [[0]*n for _ in range(n)]
        for r1, c1, r2, c2 in queries:
            for r in range(r1, r2 + 1):
                mat[r][c1] += 1
                if c2 + 1 < n:
                    mat[r][c2 + 1] -= 1
        for r in range(n):
            delta = 0
            for c in range(n):
                delta += mat[r][c]
                mat[r][c] = delta
        return mat
```

### Solution 2:  rows that hold events for start and end events + diff array that holds the current count of starts and end points + iterate over rows to update the diff array + solve in O(n^2 + q) time

```py
class Solution:
    def rangeAddQueries(self, n: int, queries: List[List[int]]) -> List[List[int]]:
        mat = [[0]*n for _ in range(n)]
        rows = [[] for _ in range(n + 1)]
        for r1, c1, r2, c2 in queries:
            rows[r1].append((1, c1, c2 + 1)) # starts interval
            rows[r2 + 1].append((-1, c1, c2 + 1)) # ends interval
        diff = [0]*(n + 1)
        for r in range(n):
            for delta, c1, c2 in rows[r]:
                diff[c1] += delta
                diff[c2] -= delta
            cur = 0
            for c in range(n):
                cur += diff[c]
                mat[r][c] = cur
        return mat
```

## 2537. Count the Number of Good Subarrays
 
### Solution 1: sliding window + math + number is added to subarray it increases the number of pairs by its previous frequency + similarly when removed

```py
class Solution:
    def countGood(self, nums: List[int], k: int) -> int:
        n = len(nums)
        pairs = Counter()
        freq = Counter()
        res = left = cur_pairs = 0
        for right in range(n):
            num = nums[right]
            cur_pairs += freq[num]
            pairs[num] += freq[num]
            freq[num] += 1
            while cur_pairs >= k:
                res += n - right
                freq[nums[left]] -= 1
                pairs[nums[left]] -= freq[nums[left]]
                cur_pairs -= freq[nums[left]]
                left += 1
        return res
```

## 2538. Difference Between Maximum and Minimum Price Sum

### Solution 1:  two dfs + store two children path sum + store parent path sum + reroot tree + O(n) time and space

```py
class Solution:
    def maxOutput(self, n: int, edges: List[List[int]], price: List[int]) -> int:
        adj_list = [[] for _ in range(n)]
        for u, v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
        self.res = 0
        path_sums1, path_sums2 = Counter(), Counter()
        path_child1, path_child2 = {}, {}
        parent_sums = Counter()
        def dfs1(node: int, parent: int) -> int:
            p1 = p2 = price[node]
            c1 = c2 = None
            for nei in adj_list[node]:
                if nei == parent: continue
                psum = price[node] + dfs1(nei, node)
                if psum > p2:
                    c1, c2 = c2, nei
                    p1, p2 = p2, psum
                elif psum > p1:
                    c1 = nei
                    p1 = psum
            if c1 is not None:
                path_sums1[node] = p1
                path_child1[node] = c1
            path_sums2[node] = p2
            path_child2[node] = c2
            return max(p1, p2)
        dfs1(0, -1)
        def dfs2(node: int, parent: int) -> None:
            # update the path from node to parents
            parent_sums[node] = parent_sums[parent] + price[node]
            if parent in path_child1 and node != path_child1[parent]:
                parent_sums[node] = max(parent_sums[node], path_sums1[parent] + price[node])
            if parent in path_child2 and node != path_child2[parent]:
                parent_sums[node] = max(parent_sums[node], path_sums2[parent] + price[node])
            # find the best path from this node as root
            psum = max(path_sums1[node], path_sums2[node], parent_sums[node]) - price[node]
            self.res = max(self.res, psum)
            for nei in adj_list[node]:
                if nei == parent: continue
                dfs2(nei, node)
        dfs2(0, -1)
        return self.res
```
[(0, i) for i in range(1, 10000)], [1] * 10000

### Solution 2:  dynamic programming + traverse each edge once + except for star case would still be O(n^2), so it only works with a dataset that doesn't include a star tree

```py
class Solution:
    def maxOutput(self, n: int, edges: List[List[int]], price: List[int]) -> int:
        adj_list = [[] for _ in range(n)]
        for u, v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
        @cache
        def dfs(node: int, parent: int) -> int:
            psum = price[node]
            for nei in adj_list[node]:
                if nei == parent: continue
                psum = max(psum, dfs(nei, node) + price[node])
            return psum
        res = 0
        for node in range(n):
            psum = dfs(node, -1)
            res = max(res, psum - price[node])
        return res
```

## 57. Insert Interval

### Solution 1:  insert interval + sort + merge overlapping intervals in linear scan + O(nlogn) time

```py
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        intervals.append(newInterval)
        intervals.sort()
        res = [intervals[0]]
        length = lambda s1, e1, s2, e2: min(e1, e2) - max(s1, s2)
        for s2, e2 in intervals[1:]:
            s1, e1 = res[-1]
            if length(s1, e1, s2, e2) >= 0:
                res[-1][0] = min(s1, s2)
                res[-1][1] = max(e1, e2)
            else:
                res.append([s2, e2])
        return res
```

### Solution 2:  merge overlapping intervals in linear scan + O(n) time

```py
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        length = lambda s1, e1, s2, e2: min(e1, e2) - max(s1, s2)
        intervals = [newInterval] + intervals
        def merge() -> None:
            x1, x2 = res[-2], res[-1]
            if length(*x1, *x2) >= 0:
                res.pop()
                res[-1][0] = min(x1[0], x2[0])
                res[-1][1] = max(x1[1], x2[1])
            elif x2[0] < x1[0]:
                res[-1], res[-2] = res[-2], res[-1]
        res = [intervals[0]]
        for x in intervals[1:]:
            res.append([*x])
            merge()
        return res
```

### Solution 3:  find intervals strictly to the left and right of the new interval + O(1) extra space + space optimized + only need to merge the extremes.

```py
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        start, end = newInterval
        left = [[s, e] for s, e in intervals if e < start]
        right = [[s, e] for s, e in intervals if s > end]
        if left + right != intervals:
            start = min(start, intervals[len(left)][0])
            end = max(end, intervals[~len(right)][1])
        return left + [[start, end]] + right
```

### Solution 4:  line sweep algorithm, O(nlogn)

```py
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        intervals.append(newInterval)
        events = []
        for s, e in intervals:
            events.append((s, -1))
            events.append((e, 1))
        events.sort()
        ans = []
        cnt = start = 0
        for p, d in events:
            cnt -= d
            if cnt == 0: ans.append((start, p)) # p is end of current interval
            elif cnt == 1 and d == -1: start = p # confirm p is start of interval
        return ans
```

## 491. Non-decreasing Subsequences

### Solution 1: n*2^n time

```py
class Solution:
    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        result = set()
        for i in range(1, 1<<n):
            is_decreasing = False
            arr = []
            for j in range(n):
                if (i>>j)&1:
                    if arr and nums[j] < arr[-1]:
                        is_decreasing = True
                        break
                    arr.append(nums[j])
            if is_decreasing or len(arr) < 2: continue
            result.add(tuple(arr))
        return result
```

```py
class Solution:
    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        result = set()
        arr = []
        def backtrack(index: int) -> None:
            if index == n:
                if len(arr) >= 2:
                    result.add(tuple(arr[:]))
                return
            if arr and nums[index] < arr[-1]:
                return backtrack(index + 1)
            arr.append(nums[index])
            backtrack(index + 1)
            arr.pop()
            backtrack(index + 1)
        backtrack(0)
        return result
```

## 974. Subarray Sums Divisible by K

### Solution 1: prefix sum + remainders + O(n + k) time

```py
class Solution:
    def subarraysDivByK(self, nums: List[int], k: int) -> int:
        res = 0
        counts = [0]*(k + 1)
        counts[0] = 1
        for psum in accumulate(nums):
            rem = psum%k 
            res += counts[rem]
            counts[rem] += 1
        return res
```

## 93. Restore IP Addresses

### Solution 1:  iterative brute force

```py
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        res = []
        n = len(s)
        valid_size = lambda ch: 0 <= int(ch) <= 255
        valid_int = lambda ch: not (ch[0] == '0' and len(ch) > 1)
        is_valid = lambda ch: len(ch) > 0 and valid_size(ch) and valid_int(ch)
        for i, j, k in product(range(n), repeat = 3):
            if not (k > j > i): continue
            cands = [s[:i], s[i:j], s[j:k], s[k:]]
            if all(is_valid(cand) for cand in cands):
                ip_addr = '.'.join(cands)
                res.append(ip_addr)
        return res
```

### Solution 2:  iterative + O(1) time + loop through the lengths of segments of the ip address

```py
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        res = []
        valid_size = lambda ch: 0 <= int(ch) <= 255
        valid_int = lambda ch: not (ch[0] == '0' and len(ch) > 1)
        is_valid = lambda ch: len(ch) > 0 and valid_size(ch) and valid_int(ch)
        for len1 in range(1, 4):
            seg1 = s[:len1]
            if not is_valid(seg1): continue
            for len2 in range(1, 4):
                seg2 = s[len1:len1 + len2]
                if not is_valid(seg2): continue
                for len3 in range(1, 4):
                    total_len = len1 + len2 + len3
                    seg3 = s[len1 + len2:total_len]
                    if not is_valid(seg3): continue
                    seg4 = s[total_len:]
                    if not is_valid(seg4): continue
                    res.append('.'.join([seg1, seg2, seg3, seg4]))
        return res
```

### Solution 3:  backtrack

```py
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        res = []
        valid_size = lambda ch: 0 <= int(ch) <= 255
        valid_int = lambda ch: not (ch[0] == '0' and len(ch) > 1)
        is_valid = lambda ch: len(ch) > 0 and valid_size(ch) and valid_int(ch)
        self.path = []
        def backtrack(cur_len: int) -> None:
            if len(self.path) > 4: return
            if cur_len == len(s):
                if len(self.path) == 4:
                    res.append('.'.join(self.path))
                return
            for seg_len in range(1, 4):
                cur_seg = s[cur_len : cur_len + seg_len]
                if not is_valid(cur_seg): return
                self.path.append(s[cur_len:cur_len + seg_len])
                backtrack(cur_len + seg_len)
                self.path.pop()
        backtrack(0)
        return res
```

## 2540. Minimum Common Value

### Solution 1: two pointers

```py
class Solution:
    def getCommon(self, nums1: List[int], nums2: List[int]) -> int:
        p1 = p2 = 0
        n1, n2 = len(nums1), len(nums2)
        while p1 < n1 and p2 < n2:
            if nums1[p1] == nums2[p2]: return nums1[p1]
            if nums1[p1] < nums2[p2]:
                p1 += 1
            else:
                p2 += 1
        return -1
```

### Solution 2: set intersection + O(n) time

```py
class Solution:
    def getCommon(self, nums1: List[int], nums2: List[int]) -> int:
        return min(common) if (common := set(nums1) & set(nums2)) else -1
```

## 2541. Minimum Operations to Make Array Equal II

### Solution 1:  greedy + can't do it if increment != decrement + can only increment a num if it is too small or decrement if too large by a fixed k integer + increment must be reachable by k which means divisible + each individual delta must be divisible by k + O(n) time

```py
class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int], k: int) -> int:
        if k == 0:
            return 0 if nums1 == nums2 else -1
        n = len(nums1)
        increment = decrement = 0
        for i in range(n):
            delta = abs(nums1[i] - nums2[i])
            if delta%k != 0: return -1
            if nums1[i] > nums2[i]:
                increment += delta
            else:
                decrement += delta
        return increment//k if increment%k == 0 and increment == decrement else -1
```

```py
class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int], k: int) -> int:
        if k == 0:
            return 0 if nums1 == nums2 else -1
        n = len(nums1)
        increment = moves = 0
        for i in range(n):
            delta = abs(nums1[i] - nums2[i])
            if delta%k != 0: return -1
            if nums1[i] > nums2[i]:
                increment += delta
                moves += delta//k
            else:
                increment -= delta
        return moves if increment == 0 else -1
```

## 2542. Maximum Subsequence Score

### Solution 1:  min heap + sort to be in decreasing order for the min_value multiplier, that way you will be always considering nums2[i] to be the minimum of the current subsequence + min_heap will allow to remove the smallest integer, and keep the maximum sum along with the current minimum multiplier + O(nlogn) time

```py
class Solution:
    def maxScore(self, nums1: List[int], nums2: List[int], k: int) -> int:
        min_heap = []
        max_score = sum_ = 0
        for x, y in sorted(zip(nums1, nums2), key = lambda p: -p[1]):
            sum_ += x
            heappush(min_heap, x)
            if len(min_heap) == k:
                max_score = max(max_score, y*sum_)
                v = heappop(min_heap)
                sum_ -= v
        return max_score
```

## 2543. Check if Point Is Reachable

### Solution 1:  prime factorization + O(sqrt(max(targetX, targetY))) time + if targetX and targetY have any prime factors in common it is impossible to reach that state

```py
class Solution:
    def isReachable(self, targetX: int, targetY: int) -> bool:
        def prime_factors(num: int) -> Set[int]:
            factors = set()
            while num % 2 == 0:
                num //= 2
            for i in range(3, int(math.sqrt(num)) + 1, 2):
                while num % i == 0:
                    factors.add(i)
                    num //= i
            if num > 2:
                factors.add(num)
            return factors
        f1, f2 = prime_factors(targetX), prime_factors(targetY)
        return len(f1 & f2) == 0
```

### Solution 2:  prime factorization + O(sqrt(max(targetX, targetY))) time + if targetX and targetY have any prime factors in common it is impossible to reach that state + factorize targetX and targetY in the same loop

```py
class Solution:
    def isReachable(self, targetX: int, targetY: int) -> bool:
        def factorize(num: int, factor: int) -> int:
            while num % factor == 0:
                num //= factor
            return num
        targetX, targetY = factorize(targetX, 2), factorize(targetY, 2)
        num = max(targetX, targetY)
        for i in range(3, int(math.sqrt(num)) + 1, 2):
            if targetX%i==0 and targetY%i==0: return False
            targetX, targetY = factorize(targetX, i), factorize(targetY, i)
        return targetX != targetY if targetX > 2 and targetY > 2 else True
```

### Solution 3: number theory + gcd + derivation from input + work backwards

```py

```

## 2544. Alternating Digit Sum

### Solution 1:  sum with conditional + O(n) time

```py
class Solution:
    def alternateDigitSum(self, n: int) -> int:
        return sum(x if i%2==0 else -x for i, x in enumerate(map(int, str(n))))
```

## 2545. Sort the Students by Their Kth Score

### Solution 1:  custom sort with key + O(nlogn) time

```py
class Solution:
    def sortTheStudents(self, score: List[List[int]], k: int) -> List[List[int]]:
        score.sort(key = lambda row: -row[k])
        return score
```

## 2546. Apply Bitwise Operations to Make Strings Equal

### Solution 1:  bitwise operations + O(n) time + if s has no 1s then target must have no ones + if s has at least one 1, than target must have at least a 1 + so eitther both are true or both are false

```py
class Solution:
    def makeStringsEqual(self, s: str, target: str) -> bool:
        return (s.count('1') == 0) == (target.count('1') == 0)
```

## 2547. Minimum Cost to Split an Array

### Solution 1:  recursive dynammic programming + frequency counter for calculate importance value + consider every possible partition + O(n^2) time complexity

```py
class Solution:
    def minCost(self, nums: List[int], k: int) -> int:
        @cache
        def dp(start_index: int) -> int:
            if start_index == len(nums): return 0
            counts_one, best = 0, math.inf
            freq = Counter()
            for end_index in range(start_index, len(nums)):
                val = nums[end_index]
                freq[val] += 1
                if freq[val] == 1:
                    counts_one += 1
                elif freq[val] == 2:
                    counts_one -= 1
                segment_len = end_index - start_index + 1
                importance_val = k + segment_len - counts_one
                best = min(best, dp(end_index + 1) + importance_val)
            return best
        return dp(0)
```

### Solution 2:  iterative dp

```py
class Solution:
    def minCost(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = [math.inf]*(n + 1)
        dp[0] = 0
        for left in range(n):
            freq = [0]*n
            count_ones = 0
            for right in range(left, n):
                val = nums[right]
                freq[val] += 1
                if freq[val] == 1:
                    count_ones += 1
                elif freq[val] == 2:
                    count_ones -= 1
                segment_len = right - left + 1
                importance_val = k + segment_len - count_ones
                dp[right + 1] = min(dp[right + 1], dp[left] + importance_val)
        return dp[n]
```

## 1088. Confusing Number II

### Solution 1:  recursive backtrack + O(5^n) time + store rotated dig in the backtrack by storing the unit

```py
class Solution:
    def confusingNumberII(self, n: int) -> int:
        valid_map = [[0, 0], [1, 1], [6, 9], [8, 8], [9, 6]]
        self.res = 0

        def backtrack(num: int, rotated_num: int, unit: int) -> None:
            if num > n: return 0
            if num != rotated_num: self.res += 1
            for dig, rotated_dig in valid_map:
                if dig == 0 and num == 0: continue # inifinite zero
                backtrack(num*10 + dig, rotated_dig*unit + rotated_num, unit*10)
        backtrack(0, 0, 1)
        return self.res
```

## 131. Palindrome Partitioning

### Solution 1:  is palindrome for string in O(n) time + recursive backtrack + O(n*2^n) time + builds palindromic partitions and finds if it can get to the end with a partition of the string into all partitions that are palindromes

```py
class Solution:
    def is_palindrome(self, part: str) -> bool:
        left, right = 0, len(part) - 1
        while left < right and part[left] == part[right]:
            left += 1
            right -= 1
        return left >= right
    def partition(self, s: str) -> List[List[str]]:
        palindrome_partitions, cur_partitions = [], []
        def backtrack(left: int) -> None:
            if left == len(s):
                palindrome_partitions.append(cur_partitions[:])
                return
            for right in range(left, len(s)):
                cur_part = s[left: right + 1]
                if not self.is_palindrome(cur_part): continue
                cur_partitions.append(cur_part)
                backtrack(right + 1)
                cur_partitions.pop()
        backtrack(0)
        return palindrome_partitions
```

### Solution 2: bitmask

```c++
bool indices[16] = {};
vector<vector<string>> partition(string s) {
    int n = s.size();
    vector<vector<string>> result;
    auto isPalindrome = [&](int i, int j) {
        while (i<j) {
            if (s[i]!=s[j]) {return false;}
            i++;
            j--;
        }
        return true;
    };
    for (int i = 0;i<(1<<(n-1));i++) {
        memset(indices,false,sizeof(indices));
        bool isValidPartition = true;
        for (int j = 0, start = 0;j<n;j++) {
            if ((i>>j)&1 || j==n-1) {
                indices[j]=true;
                if (!isPalindrome(start,j)) {
                    isValidPartition = false;
                    break;
                }
                start = j+1;
            }
        }
        if (isValidPartition) {
            vector<string> partition;
            string pally = "";
            for (int j = 0;j<n;j++) {
                pally+=s[j];
                if (indices[j]) {
                    partition.push_back(pally);
                    pally.clear();
                }
            }
            result.push_back(partition);
        }
    }
    return result;
}
```

### Solution 3: DFS + backtracking

```c++
vector<vector<string>> result;
void dfs(int start, string& s, vector<string>& pally) {
    int n = s.size();
    if (start==n) {
        result.push_back(pally);
        return;
    }
    auto isPalindrome = [&](int i, int j) {
        while (i<j) {
            if (s[i]!=s[j]) {return false;}
            i++;
            j--;
        }
        return true;
    };
    for (int i = start;i<n;i++) {
        if (!isPalindrome(start, i)) continue;
        pally.push_back(s.substr(start,i-start+1));
        dfs(i+1,s,pally);
        pally.pop_back();
    }
}
vector<vector<string>> partition(string s) {
    vector<string> pally;
    dfs(0,s,pally);
    return result;
}
```


```c++
vector<vector<string>> partition(string s) {
    vector<vector<string>> result;
    vector<string> pally;
    int n = s.size();
    auto isPalindrome = [&](int i, int j) {
        while (i<j) {
            if (s[i]!=s[j]) {return false;}
            i++;
            j--;
        }
        return true;
    };
    function<void(int)> dfs = [&](int start) {
        if (start==n) {
            result.push_back(pally);
            return;
        }
        for (int i = start;i<n;i++) {
            if (!isPalindrome(start,i)) continue;
            pally.push_back(s.substr(start,i-start+1));
            dfs(i+1);
            pally.pop_back();
        }
    };
    dfs(0);
    return result;
}
```

### Solution 4: DFS + backtracking with memoization of palindromes

```c++
vector<vector<string>> partition(string s) {
    vector<vector<string>> result;
    vector<string> pally;
    int n = s.size();
    vector<vector<int>> dp(n, vector<int>(n, -1));
    function<int(int,int)> isPalindrome = [&](int i, int j) {
        if (dp[i][j]>=0) {return dp[i][j];}
        if (i==j) {return dp[i][j]=1;}
        if (j-i==1) {return dp[i][j] = s[i]==s[j];}
        return dp[i][j] = s[i]==s[j] && isPalindrome(i+1,j-1);
    };
    function<void(int)> dfs = [&](int start) {
        if (start==n) {
            result.push_back(pally);
            return;
        }
        for (int i = start;i<n;i++) {
            if (isPalindrome(start,i)==0) continue;
            pally.push_back(s.substr(start,i-start+1));
            dfs(i+1);
            pally.pop_back();
        }
    };
    dfs(0);
    return result;
}
```

## 997. Find the Town Judge

### Solution 1: dropwhile + default value for end of iterator + count total degrees of each vertex + no parallel edges + no self loops + the vertex that is called the judge will have n - 1 degrees because there will be n - 1 indegrees and 0 outdegrees

```py
class Solution:
    def findJudge(self, n: int, trust: List[List[int]]) -> int:
        degrees = [0] * (n + 1)
        for u, v in trust:
            degrees[v] += 1
            degrees[u] -= 1
        return next(dropwhile(lambda i: degrees[i] != n - 1, range(1, n + 1)), default := -1)
```

### Solution 2: Count indegrees and outdegrees, judge should have all nodes with indegree on it and have no outdegree edges.

```c++
int findJudge(int n, vector<vector<int>>& trust) {
    vector<int> indegrees(n+1,0), outdegrees(n+1,0);
    for (vector<int>& p : trust) {
        indegrees[p[1]]++;
        outdegrees[p[0]]++;
    }
    for (int i = 1;i<=n;i++) {
        if (indegrees[i]-outdegrees[i]==n-1) {
            return i;
        } 
    }
    return -1;
}
```

## 2359. Find Closest Node to Given Two Nodes

### Solution 1:  linear path through each nodes path + find the first node that is visited from both paths

```py
class Solution:
    def closestMeetingNode(self, edges: List[int], node1: int, node2: int) -> int:
        n = len(edges)
        vis = [[False]*2 for _ in range(n)]
        while True:
            # NODE AT EQUAL DISTANCE FROM INITIAL NODE1 AND NODE2 OR VISIT NODE THAT IS ALREADY VISITED FROM THE OTHER PATH
            if node1 == node2: return node1
            elif vis[node1][1] or vis[node2][0]: return min(node1 if vis[node1][1] else math.inf, node2 if vis[node2][0] else math.inf)
            # CHILD NODES IF UNVISITED
            c1, c2 = edges[node1] if not vis[node1][0] else -1, edges[node2] if not vis[node2][1] else -1
            # MARK NODES AS VISITED FOR RESPECTIVE PATH 0 AND PATH 1
            vis[node1][0] = True
            vis[node2][1] = True
            # IF NO MORE CHILD NODES, BREAK
            if c1 == -1 and c2 == -1: break
            # UPDATE CURRENT NODE1 AND NODE2 VALUES
            node1, node2 = c1 if c1 != -1 else node1, c2 if c2 != -1 else node2
        return -1
```

## 909. Snakes and Ladders

### Solution 1:  bfs + O(n^2) time

```py
class Solution:
    def snakesAndLadders(self, board: List[List[int]]) -> int:
        vis = set({1})
        n = len(board)
        target = n**2
        queue = deque([1])
        moves = 0
        while queue:
            sz = len(queue)
            for _ in range(sz):
                cell = queue.popleft()
                if cell == target: return moves
                for nei in range(cell + 1, min(cell + 6, target) + 1):
                    row = (nei - 1)//n
                    col = (nei - 1)%n
                    if row%2 == 0 and board[~row][col] != -1:
                        new_nei = board[~row][col]
                        if new_nei in vis: continue
                        vis.add(new_nei)
                        queue.append(new_nei)
                    elif row%2 != 0 and board[~row][~col] != -1:
                        new_nei = board[~row][~col]
                        if new_nei in vis: continue
                        vis.add(new_nei)
                        queue.append(new_nei)
                    else:
                        if nei in vis: continue
                        vis.add(nei)
                        queue.append(nei)
            moves += 1
        return -1
```

## 472. Concatenated Words

### Solution 1:  dynammic programming

```py
class Solution:
    def concatenated(self, word: str, dictionary: Set[str]) -> bool:
        n = len(word)
        dp = [False]*(n + 1)
        dp[0] = True
        for right in range(1, n + 1):
            for left in range(right - 1, -1, -1):
                infix_word = word[left:right]
                if infix_word == word: continue
                if infix_word in dictionary:
                    dp[right] = dp[left]
                if dp[right]: break
        return dp[n]
    def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
        words = set(words)
        result = []
        for word in words:
            if self.concatenated(word, words):
                result.append(word)
        return result
```

## 2549. Count Distinct Numbers on Board

### Solution 1:  math + if n > 2 it is true that at each day n % (n - 1) == 1, so it will keep doing this all the way till it reaches 2, 5, 4, 3, 2 + O(1) time

```py
class Solution:
    def distinctIntegers(self, n: int) -> int:
        return n - 1 if n > 2 else 1
```

## 2550. Count Collisions of Monkeys on a Polygon

### Solution 1:  math + observe pattern + pow + fast mod algorithm + O(logn) time

```py
class Solution:
    def monkeyMove(self, n: int) -> int:
        return (pow(2, n, mod := int(1e9) + 7) - 2 + mod)%mod
```

## 2551. Put Marbles in Bags

### Solution 1:  greedy + sort + nlogn time

```py
class Solution:
    def putMarbles(self, weights: List[int], k: int) -> int:
        if k == 1: return 0
        n = len(weights)
        min_pairs = sorted([weights[i] + weights[i-1] for i in range(1, n)])
        min_score = weights[0] + sum(min_pairs[:k - 1]) + weights[-1]
        max_score = weights[0] + sum(min_pairs[-k + 1:]) + weights[-1]
        return max_score - min_score
```

### Solution 2: pairwise + k - 1 cuts + nlargest and nsmallest + heapq + O(nlogk) time

```py
class Solution:
    def putMarbles(self, weights: List[int], k: int) -> int:
        pairs = [x + y for x, y in pairwise(weights)]
        return sum(nlargest(k - 1, pairs)) - sum(nsmallest(k - 1, pairs))
```

## 2552. Count Increasing Quadruplets

### Solution 1:  iterative dp + O(n^2) time + compute count for i < j < k and uses them later when it can consider l index

```py
class Solution:
    def countQuadruplets(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0]*n
        res = 0
        for j in range(n):
            prev_smaller = 0
            for i in range(j):
                if nums[i] < nums[j]: # imagine i = k and j = l, so this is when k < l and nums[k] < nums[l]
                    prev_smaller += 1
                    res += dp[i] # where i = k and j = l, so this is k < l and nums[k] < nums[l]
                else: # imagine i = k and j = j, so nums[k] > nums[j], so add this so can use it later
                    dp[i] += prev_smaller
        return res
```

### Solution 2: precompute count of smaller and count of larger elements for a range + iterate through each j, k pair that satisfies condition of j < k and nums[j] > nums[k] + multiple the count of smaller elements and larger elements in proper ranges + O(n^2) time

```cpp
class Solution {
public:
    long long countQuadruplets(vector<int>& nums) {
        long long res = 0;
        int n = nums.size();
        // construct count_smaller which is a precomputed count array  of the count of smaller elements in a 
        // range [i, j], where all elements in that range are smaller than nums[i] and i > j
        vector<vector<int>> countSmaller(n, vector<int>(n, 0));
        for (int i = n-1;i>=0;i--) {
            int cnt = 0;
            for (int j = i-1;j>=0;j--) {
                if (nums[j] < nums[i]) {
                    countSmaller[i][j] = ++cnt;
                } else {
                    countSmaller[i][j] = cnt;
                }
            }
        }
        // construct count_larger which is precomputed count array of the count of larger elements in a
        // range [i, j], where all elements in that range are greater than nums[i] and i < j
        vector<vector<int>> countLarger(n, vector<int>(n, 0));
        for (int i = 0;i<n;i++) {
            int cnt = 0;
            for (int j = i+1;j<n;j++) {
                if (nums[i] < nums[j]) {
                    countLarger[i][j] = ++cnt;
                } else {
                    countLarger[i][j] = cnt;
                }
            }
        }
        for (int j = 1;j<n-2;j++) {
            for (int k=j+1;k<n-1;k++) {
                if (nums[j] > nums[k]) {
                    // elements smaller than nums[k] in the range [0, k] and subtract the elments that are smaller on range [j, k] because want elements smaller than k in range [0, j]
                    long long count_smaller = countSmaller[k][0] - countSmaller[k][j];
                    // elements larger than nums[j] in the range [j, n-1] and subtract the elements that are larger in the range [j, k] because want elements larger than j in range [k, n-1
                    long long count_larger = countLarger[j][n-1] - countLarger[j][k];
                    res += count_smaller * count_larger;
                }
            }
        }
        return res;
    }
};
```

# 460. LFU Cache

### Solution 1: hashmaps + minfreq quick eviction => (freq -> doubly linked lists) => (key -> node)

```c++
struct Node {
    int key, val, freq;
    list<int>::iterator it;
};
class LFUCache {
public:
    unordered_map<int,Node*> vmap;
    unordered_map<int, list<int>> D;
    int cap, minFreq;
    LFUCache(int capacity) {
        cap=capacity;
        minFreq = 0;
    }
    
    int get(int key) {
        if (vmap.find(key)==vmap.end()) {return -1;}
        Node *node = vmap[key];
        int f = node->freq;
        D[f].erase(node->it);
        if (f==minFreq && D[f].empty()) {
            minFreq++;
        }
        node->freq++;
        D[f+1].push_front(key);
        node->it = D[f+1].begin();
        return node->val;
    }
    
    void put(int key, int value) {
        if (cap==0 && minFreq==0) return;
        if (vmap.find(key)==vmap.end()) {
            if (cap==0) {
                int rmk = D[minFreq].back();
                D[minFreq].pop_back();
                vmap.erase(rmk);
            } else {
                cap--;
            }
            Node *node = new Node();
            node->key = key;
            node->val = value;
            node->freq = 1;
            D[1].push_front(key);
            node->it = D[1].begin();
            vmap[key]=node;
            minFreq = 1;
        } else {
            Node *node = vmap[key];
            int f = node->freq;
            D[f].erase(node->it);
            if (f==minFreq && D[f].empty()) {
                minFreq++;
            }
            node->val = value;
            node->freq++;
            D[f+1].push_front(key);
            node->it = D[f+1].begin();
        }
    }
};
```

## Solution 2:  min heap + hash table for charge + hash table for counter + hash table for values

```py
class LFUCache:

    def __init__(self, capacity: int):
        self.minheap = []
        self.charge_table = {}
        self.table = {}
        self.counter_table = defaultdict(int)
        self.cap = capacity
        self.charge = 0

    def get(self, key: int) -> int:
        if key not in self.table: return -1
        
        self.charge_table[key] = self.charge
        self.counter_table[key] += 1
        heappush(self.minheap, (self.counter_table[key], self.charge, key))
        self.charge += 1
        
        return self.table[key]

    def put(self, key: int, value: int) -> None:
        if key in self.table:
            self.table[key] = value
            self.get(key)
            return
        if self.cap == 0: return
        while len(self.table) == self.cap:
            cnt, chrg, k = heappop(self.minheap)
            if chrg == self.charge_table[k]:
                del self.table[k]
                del self.charge_table[k]
                del self.counter_table[k]
        self.charge_table[key] = self.charge
        self.counter_table[key] += 1
        heappush(self.minheap, (self.counter_table[key], self.charge, key))
        self.charge += 1
        self.table[key] = value
```

### Solution 3:  lru cache with infinite capacity + frequency dictionary and frequency lru cache dictionary + pointers to min frequency

```py
class LRUCache:

    def __init__(self, capacity: int = math.inf):
        self.cache = OrderedDict()
        self.cap = capacity

    def get(self, key: int) -> int:
        if key not in self.cache: return -1
        self.put(key, self.cache[key])
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.cap:
            self.cache.popitem(last = False)

    def discard(self, key: int) -> None:
        if key not in self.cache: return
        self.cache.move_to_end(key)
        self.cache.popitem() # pops item from the end

    """
    returns key of lru item or -1 if lru cache is empty
    """
    def pop(self) -> int:
        try:
            return self.cache.popitem(last = False)[0] # pops item from start and returns just the key
        except:
            return -1

    def __repr__(self):
        return str(self.cache)
    
    def __len__(self):
        return len(self.cache)

    def __getitem__(self, key):
        return self.cache[key]

class LFUCache:

    def __init__(self, capacity: int):
        self.freq_key = dict() # points to the frequency of this key in lfu cache
        self.freq_lru_cache = defaultdict(LRUCache) # frequency cache construct of lru caches with infinite capacities
        self.min_freq_ptr = 0 # pointer to the least frequency with key or keys
        self.total_count = 0 # count of objects in lfu cache
        self.capacity = capacity # capacity of the lfu

    def get(self, key: int) -> int:
        if key not in self.freq_key: return -1
        freq = self.freq_key[key]
        val = self.freq_lru_cache[freq][key]
        self.put(key, val)
        return val

    def put(self, key: int, value: int) -> None:
        if self.capacity == 0: return
        freq = self.freq_key.get(key, 0)
        if freq == 0:
            self.total_count += 1
        if self.total_count > self.capacity:
            lfu_key = self.freq_lru_cache[self.min_freq_ptr].pop()
            self.freq_key.pop(lfu_key)
            self.total_count -= 1
        self.freq_lru_cache[freq].discard(key)
        if freq == 0 or (freq == self.min_freq_ptr and len(self.freq_lru_cache[freq]) == 0):
            self.min_freq_ptr = freq + 1
        self.freq_lru_cache[freq + 1].put(key, value)
        self.freq_key[key] = freq + 1
```

## 1908. Game of Nim

### Solution 1:  nim game theory solution + xor + xor sum greater than 0 is winning state, and xor sum equal to 0 is losing state + O(n) time

```py
class Solution:
    def nimGame(self, piles: List[int]) -> bool:
        return reduce(operator.xor, piles) > 0
```

### Solution 2:  sprague grundy game theory

```py

```

## 953. Verifying an Alien Dictionary

### Solution 1:  recursion + groupby + custom comparator for groupby + dropwhile to remove 1 length words from front + O(m) time

```py
class Solution:
    def isAlienSorted(self, words: List[str], order: str) -> bool:
        if len(words) == 1: return True
        char_index = 0
        for key, grp in groupby(words, key = lambda word: word[0]):
            while char_index < len(order) and key != order[char_index]:
                char_index += 1
            if char_index == len(order): return False
            next_words = []
            for word in dropwhile(lambda word: len(word) == 1, grp):
                if len(word) == 1: return False # SHOULDN'T SEE ANOTHER 1 LENGTH WORD
                if len(word) > 1:
                    next_words.append(word[1:])
            if not self.isAlienSorted(next_words, order): return False
        return True
```

### Solution 2:  convert characters to index values + lexicographical ordering of the list of integers +  O(m) time, where m = count_chars_in(words)

```py
class Solution:
    def isAlienSorted(self, words: List[str], order: str) -> bool:
        order_index_map = {ch: i for i, ch in enumerate(order)}
        words = [[order_index_map[ch] for ch in word] for word in words]
        return not any(x > y for x, y in zip(words, words[1:]))
```

## 1071. Greatest Common Divisor of Strings

### Solution 1:  gcd + math + euclidean algorithm to find gcd O(logn) + O(n1 + n2) time

```py
class Solution:
    def gcdOfStrings(self, s1: str, s2: str) -> str:
        return s1[:math.gcd(len(s1), len(s2))] if s1 + s2 == s2 + s1 else ''
```

## 734. Sentence Similarity

### Solution 1:  adjacency list + graph

```py
class Solution:
    def areSentencesSimilar(self, sentence1: List[str], sentence2: List[str], similarPairs: List[List[str]]) -> bool:
        if len(sentence1) != len(sentence2): return False
        adj_list = defaultdict(set)
        for u, v in similarPairs:
            adj_list[u].add(v)
            adj_list[v].add(u)
        for s1, s2 in zip(sentence1, sentence2):
            if s1 != s2 and s1 not in adj_list[s2]: return False
        return True
```

## 1626. Best Team With No Conflicts

### Solution 1:  iterative dp + maximum sum increasing subsequence + O(n^2) time + O(n) extra space

```py
class Solution:
    def bestTeamScore(self, scores: List[int], ages: List[int]) -> int:
        n = len(scores)
        dp = [0]*n
        scores = [scores[i] for i in sorted(range(n), key = lambda i: (ages[i], scores[i]))]
        for i in range(n):
            dp[i] = scores[i]
            for j in range(i):
                if scores[i] >= scores[j]:
                    dp[i] = max(dp[i], dp[j] + scores[i])
        return max(dp)
```

## 567. Permutation in String

### Solution 1:  sliding window + O(n) time + counter

```py
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        unicode = lambda ch: ord(ch) - ord('a')
        freq, window_freq = [0]*26, [0]*26
        for ch in s1:
            freq[unicode(ch)] += 1
        n1 = len(s1)
        for i in range(len(s2)):
            window_freq[unicode(s2[i])] += 1
            if i >= n1 - 1:
                if freq == window_freq: return True
                window_freq[unicode(s2[i - n1 + 1])] -= 1
        return False
```

## 2553. Separate the Digits in an Array

### Solution 1:  map + chain from iterables 

```py
class Solution:
    def separateDigits(self, nums: List[int]) -> List[int]:
        return [dig for dig in map(int, chain.from_iterable(map(str, nums)))]
```

## 2554. Maximum Number of Integers to Choose From a Range I

### Solution 1:  sort + greedy 

```py
class Solution:
    def maxCount(self, banned: List[int], n: int, maxSum: int) -> int:
        banned.sort()
        banned.append(math.inf)
        res = cur_sum = ptr = 0
        for i in range(1, n + 1):
            while banned[ptr] < i:
                ptr += 1
            if cur_sum + i > maxSum: break
            if i != banned[ptr]:
                res += 1
                cur_sum += i
        return res
```

## 2555. Maximize Win From Two Segments

### Solution 1:  max heap + greedy + O(nlogn) time + O(n) extra space

```py
class Solution:
    def maximizeWin(self, pos: List[int], k: int) -> int:
        maxheap = []
        left = 0
        n = len(pos)
        for right in range(n):
            while pos[right] - pos[left] > k:
                left += 1
            val = right - left + 1
            heappush(maxheap, (-val, pos[left]))
        left += 1
        while left < n:
            val = n - left
            heappush(maxheap, (-val, pos[left]))
            left += 1
        res = left = 0
        for right in range(n):
            while pos[right] - pos[left] > k:
                left += 1
            left_seg_val = right - left + 1
            while maxheap and maxheap[0][-1] <= pos[right]:
                heappop(maxheap)
            right_seg_val = abs(maxheap[0][0]) if maxheap else 0
            res = max(res, left_seg_val + right_seg_val)
        return res
```

## 2556. Disconnect Path in a Binary Matrix by at Most One Flip

### Solution 1:

```py

```

## 1470. Shuffle the Array

### Solution 1:  bit manipulation + using first and second 10 bits to represent the pair (xi, yi), put them in the first n places, and then place the ones in the end first + O(n) time + O(1) space

```py
class Solution:
    def shuffle(self, nums: List[int], n: int) -> List[int]:
        shift = 10
        # PACK UP THE PAIRS (XI, YI)
        for i in range(n):
            nums[n - i - 1] |= (nums[~i] << shift)
        # UNPACK THE PAIRS INTO CORRECT LOCATION IN ARRAY
        for i in range(n):
            num = nums[n - i - 1]
            x, y = num & ((1 << shift) - 1), num >> 10
            nums[~(2*i) - 1], nums[~(2*i)] = x, y
        return nums
```

```py
class Solution:
    def shuffle(self, nums: List[int], n: int) -> List[int]:
        shift = 10
        # PACK UP THE PAIRS (XI, YI)
        # STORE first 10 bits are for y and last 10 bits are for x if looking at bits from left to right with msb being on the left
        for i in range(n, 2*n):
            y = nums[i] << 10
            nums[i - n] |= y
        mask_for_x = ((1 << shift) - 1)
        # UNPACK THE PAIRS INTO CORRECT LOCATION IN ARRAY
        for i in reversed(range(n)):
            x = nums[i] & mask_for_x
            y = nums[i] >> shift
            nums[2*i], nums[2*i + 1] = x, y
        return nums
```

## 904. Fruit into Baskets

### Solution 1: count + sliding window

```py
class Solution:
    def totalFruit(self, fruits: List[int]) -> int:
        cnta = cntb = 0
        fruita = fruitb = -1
        j = 0
        maxFruit = 0
        for i, fruit in enumerate(fruits):
            if fruita == -1 or fruit==fruita: 
                fruita = fruit
                cnta += 1
            elif fruitb == -1 or fruit==fruitb:
                fruitb = fruit
                cntb += 1
            else:
                while cnta > 0 and cntb > 0:
                    cnta -= int(fruits[j]==fruita)
                    cntb -= int(fruits[j]==fruitb)
                    j+=1
                if cnta == 0:
                    fruita = fruit
                    cnta += 1
                else:
                    fruitb = fruit
                    cntb += 1
            maxFruit = max(maxFruit, i - j + 1)
        return maxFruit
```

## 2556. Disconnect Path in a Binary Matrix by at Most One Flip

### Solution 1:

```py

```

## 280. Wiggle Sorts

### Solution 1:  bucket sort + extra space + two pointers, take from smallest then largest + O(n) time + O(n) space

```py
class Solution:
    def wiggleSort(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        def bucket_sort(nums: List[int]) -> List[int]:
            m = max(nums)
            bucket = [0] * (m + 1)
            for num in nums:
                bucket[num] += 1
            return bucket
        buckets = bucket_sort(nums)
        n, m = len(nums), len(buckets)
        left, right = 0, m - 1
        for i in range(n):
            if i%2 == 0:
                left = next(dropwhile(lambda idx: buckets[idx] == 0, range(left, m)))
                nums[i] = left
                buckets[left] -= 1
            else:
                right = next(dropwhile(lambda idx: buckets[idx] == 0, range(right, -1, -1)))
                nums[i] = right
                buckets[right] -= 1
```

### Solution 2:  greedy, sort when wiggle sort invariant is false + wiggle sort + O(n) time + O(1) space

```py
class Solution:
    def wiggleSort(self, nums: List[int]) -> None:
        for i in range(len(nums) - 1):
            if i%2 == 0 and nums[i] > nums[i + 1]:
                nums[i], nums[i + 1] = nums[i + 1], nums[i]
            elif i%2 == 1 and nums[i] < nums[i + 1]:
                nums[i], nums[i + 1] = nums[i + 1], nums[i]
```

## 1129. Shortest Path with Alternating Colors

### Solution 1:  bfs + memoization + shortest path

```py
class Solution:
    def shortestAlternatingPaths(self, n: int, redEdges: List[List[int]], blueEdges: List[List[int]]) -> List[int]:
        dist = [[math.inf]*n for _ in range(2)]
        red, blue = 0, 1
        dist[red][0] = dist[blue][0] = 0
        adj_list = [[[] for _ in range(n)] for _ in range(2)]
        for u, v in redEdges:
            adj_list[red][u].append(v)
        for u, v in blueEdges:
            adj_list[blue][u].append(v)
        queue = deque([(0, red), (0, blue)])
        while queue:
            node, prev_edge_color = queue.popleft()
            color = prev_edge_color ^ 1
            for nei in adj_list[color][node]:
                ndist = dist[prev_edge_color][node] + 1
                if ndist < dist[color][nei]:
                    dist[color][nei] = ndist
                    queue.append((nei, color))
        return [min(rd, bd) if rd != math.inf or bd != math.inf else -1 for rd, bd in zip(dist[red], dist[blue])]
```

## 138. Copy List with Random Pointer

### Solution 1:  Hash Table + iterate

```py
class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        if not head: return head
        sentinel = Node(0,head)
        copy_dict = {sentinel: Node(0)}
        current = sentinel
        while current:
            # CREATE THE NEW NODE AND SET OLD EQUAL TO IT. 
            if current.next:
                if current.next not in copy_dict:
                    copy_dict[current.next] = Node(current.next.val)
                copy_dict[current].next = copy_dict[current.next]
            # CREATE THE NEW NODE AND SET OLD RANDOM EQUAL TO IT.
            if current.random:
                if current.random not in copy_dict:
                    copy_dict[current.random] = Node(current.random.val)
                copy_dict[current].random = copy_dict[current.random]
            current=current.next
        return copy_dict[sentinel.next]
```

### Solution 2: Recursion + dfs + hash table + graph

```py
class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        self.visited = {}
        
        def dfs(head):
            if not head: return head
            if head in self.visited:
                return self.visited[head]
            
            node = Node(head.val)
            self.visited[head] = node
            node.next, node.random = dfs(head.next), dfs(head.random)
            return node
            
        return dfs(head)
```

### Solution 3:  Iterate + Interweaved Linked List

```py
class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        current = head
        # CREATE THE INTERWEAVED LINKED LIST
        while current:
            node = Node(current.val, current.next)
            current.next = node
            current = node.next
        current = head
        # SET RANDOM POINTERS FROM INTERWEAVED LIST
        while current:
            nnode = current.next
            if current.random:
                nnode.random = current.random.next
            current = nnode.next
        sentinel = Node(0,head)
        current = sentinel
        # CREATE NEW LINKED LIST FROM INTERWEAVED LINKED LIST
        while current.next:
            current.next = current.next.next
            current=current.next
        return sentinel.next
```

## 2562. Find the Array Concatenation Value

### Solution 1:  two pointers

```py
class Solution:
    def findTheArrayConcVal(self, nums: List[int]) -> int:
        n = len(nums)
        left, right = 0, n - 1
        res = 0
        while left <= right:
            if left == right:
                num = nums[left]
            else:
                num = int(str(nums[left]) + str(nums[right]))
            res += num
            left += 1
            right -= 1
        return res
            
```

## 2563. Count the Number of Fair Pairs

### Solution 1:  sorted list + binary searching 

```py
from sortedcontainers import SortedList
class Solution:
    def countFairPairs(self, nums: List[int], lower: int, upper: int) -> int:
        arr = SortedList()
        res = 0
        for num in nums:
            lower_bound = lower - num
            upper_bound = upper - num
            left, right = arr.bisect_left(lower_bound), arr.bisect_right(upper_bound)
            res += right - left
            arr.add(num)
        return res
```

## 2564. Substring XOR Queries

### Solution 1:  bit manipulation + bit math + dictionary + creating all substrings up to length of mx which largest mx can be is 30

```py
class Solution:
    def substringXorQueries(self, s: str, queries: List[List[int]]) -> List[List[int]]:
        n, q = len(s), len(queries)
        ans = [[-1]*2 for _ in range(q)]
        query_index = defaultdict(list)
        mx = 0
        for i, (first, second) in enumerate(queries):
            num = first ^ second
            query_index[num].append(i)
            mx = max(mx, len(bin(num)[2:]))
        for i in range(n + 1):
            left = max(0, i - mx)
            for j in reversed(range(left, i)):
                val = int(s[j:i], 2)
                if val not in query_index: continue
                for k in query_index[val]:
                    ans[k] = [j, i - 1]
                query_index.pop(val)
        return ans
```

## 2565. Subsequence With the Minimum Score

### Solution 1:  prefix + suffix + two pointers

```py
class Solution:
    def minimumScore(self, s: str, t: str) -> int:
        n1, n2 = len(s), len(t)
        prefix, suffix = [0]*n1, [0]*n1
        left, right = 0 , n2 - 1
        for i, ch in enumerate(s):
            if left < n2 and ch == t[left]:
                left += 1
            prefix[i] = left
        for i in reversed(range(n1)):
            ch = s[i]
            if right >= 0 and ch == t[right]:
                right -= 1
            suffix[i] = right
        res = math.inf
        for i, (ri, le) in enumerate(zip(suffix, prefix)):
            if i > 0 and i < n1 - 1 and prefix[i - 1] < prefix[i] and suffix[i] < suffix[i + 1]:
                res = min(res, ri - le + 2)
            else:
                res = min(res, ri - le + 1)
        return max(0, res)
```

## 1523. Count Odd Numbers in an Interval Range

### Solution 1:  math + add one for it if odd number

```py
class Solution:
    def countOdds(self, low: int, high: int) -> int:
        return (high - low + (low&1) + (high&1))//2
```

## 989. Add to Array-Form of Integer

### Solution 1:  zip_longest + math add integers + carry

```py
class Solution:
    def addToArrayForm(self, num: List[int], k: int) -> List[int]:
        carry = 0
        res = []
        for x, y in zip_longest(map(int, reversed(num)), map(int, reversed(str(k))), fillvalue = 0):
            carry, cur = divmod(x + y + carry, 10)
            res.append(cur)
        if carry:
            res.append(carry)
        return reversed(res)
```

## 67. Add Binary

### Solution 1:

```py
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        res, carry = int(a, 2), int(b, 2)
        while carry:
            res, carry = res ^ carry, (res & carry) << 1
        return bin(res)[2:]
```

```py
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        res = []
        carry = 0
        for x, y in zip_longest(map(int, reversed(a)), map(int, reversed(b)), fillvalue = 0):
            carry, cur = divmod(x + y + carry, 2)
            res.append(cur)
        if carry:
            res.append(carry)
        return ''.join(map(str, reversed(res)))
```

## 2566. Maximum Difference by Remapping a Digit

### Solution 1:  try mapping every digit to 0 and 9 to get the max difference + O(1) time since there are only 10 digits possible always

```py
class Solution:
    def minMaxDifference(self, num: int) -> int:
        lo, hi = '0', '9'
        min_val, max_val = math.inf, -math.inf
        for d in string.digits:
            min_val = min(min_val, int(str(num).replace(d, lo)))
            max_val = max(max_val, int(str(num).replace(d, hi)))
        return max_val - min_val
```

## 2568. Minimum Impossible OR

### Solution 1:  smallest power of two that doesn't exist in nums

```py
class Solution:
    def minImpossibleOR(self, nums: List[int]) -> int:
        nums = set(nums)
        return next(1 << i for i in range(32) if (1<<i) not in nums)
```

## 2567. Minimum Score by Changing Two Elements

### Solution 1:  3 cases, always lowest score will be 0 from duplicates + case 1 difference between second largest and second smallest + case 2 difference between third largest and smallest + case 3 difference between largest and third smallest + O(nlogn)

```py
class Solution:
    def minimizeSum(self, nums: List[int]) -> int:
        nums.sort()
        return min(nums[-2] - nums[1], nums[-3] - nums[0], nums[-1] - nums[2])
```

## 2569. Handling Sum Queries After Update

### Solution 1:  lazy segment tree + range queries + range updates + keeps track of count of ones on a segment and stores the number of flips on a segment + O(nlogn) time 

```py
class LazySegmentTree:
    def __init__(self, n: int, neutral: int, noop: int, arr: List[int]):
        self.neutral = neutral
        self.size = 1
        self.noop = noop
        self.n = n 
        self.arr = arr
        while self.size<n:
            self.size*=2
        self.num_flips = [noop for _ in range(self.size*2)] # number of flips in a segment
        self.counts = [neutral for _ in range(self.size*2)] # count of ones in a segment
        self.build()
        
    def build(self):
        for segment_idx in range(self.n):
            v = self.arr[segment_idx]
            segment_idx += self.size - 1
            self.counts[segment_idx] = v
            self.ascend(segment_idx)

    def is_leaf_node(self, segment_right_bound, segment_left_bound) -> bool:
        return segment_right_bound - segment_left_bound == 1
    
    def calc_op(self, x: int, y: int) -> int:
        return x + y

    def modify_op(self, x: int, y: int) -> int:
        return x ^ y
    
    """
    Gives the count of a bit in a segment, which is a range. and the length of that range is represented by the segment_len.
    And it flips all the bits such as 0000110 -> 1111001, the number of 1s are now segment_len - cnt, where cnt is the current number of 1s
    So it goes from 2 -> 7 - 2 = 5
    """
    def flip_op(self, segment_len: int, cnt: int) -> int:
        return segment_len - cnt

    def propagate(self, segment_idx: int, segment_left_bound: int, segment_right_bound: int) -> None:
        # do not want to propagate if it is a leaf node or if it is no operation (means there are no updates stored there).
        if self.is_leaf_node(segment_right_bound, segment_left_bound) or self.num_flips[segment_idx] == self.noop: return
        left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
        children_segment_len = (segment_right_bound - segment_left_bound) >> 1
        self.num_flips[left_segment_idx] = self.modify_op(self.num_flips[left_segment_idx], self.num_flips[segment_idx])
        self.num_flips[right_segment_idx] = self.modify_op(self.num_flips[right_segment_idx], self.num_flips[segment_idx])
        self.counts[left_segment_idx] = self.flip_op(children_segment_len, self.counts[left_segment_idx])
        self.counts[right_segment_idx] = self.flip_op(children_segment_len, self.counts[right_segment_idx])
        self.num_flips[segment_idx] = self.noop
    
    def ascend(self, segment_idx: int) -> None:
        while segment_idx > 0:
            segment_idx -= 1
            segment_idx >>= 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.counts[segment_idx] = self.calc_op(self.counts[left_segment_idx], self.counts[right_segment_idx])
    
    def update(self, left: int, right: int, val: int) -> None:
        stack = [(0, self.size, 0)]
        segments = []
        while stack:
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # COMPLETE OVERLAP
            if segment_left_bound >= left and segment_right_bound <= right:
                self.num_flips[segment_idx] = self.modify_op(self.num_flips[segment_idx], val)
                segment_len = segment_right_bound - segment_left_bound
                self.counts[segment_idx] = self.flip_op(segment_len, self.counts[segment_idx])
                segments.append(segment_idx)
                continue
            # PARTIAL OVERLAP
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.propagate(segment_idx, segment_left_bound, segment_right_bound)
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
        for segment_idx in segments:
            self.ascend(segment_idx)

    def query(self, left: int, right: int) -> int:
        stack = [(0, self.size, 0)]
        result = self.neutral
        while stack:
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # LEAF NODE
            if segment_left_bound >= left and segment_right_bound <= right:
                result = self.calc_op(result, self.counts[segment_idx])
                continue
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2    
            self.propagate(segment_idx, segment_left_bound, segment_right_bound)      
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
        return result
    
    def __repr__(self) -> str:
        return f"counts: {self.counts}, num_flips: {self.num_flips}"
    
class Solution:
    def handleQuery(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        lazy_seg_tree = LazySegmentTree(n, 0, 0, nums1)
        base, addl = sum(nums2), 0
        ans = []
        for query in queries:
            if query[0] == 1:
                left, right = query[1:]
                lazy_seg_tree.update(left, right + 1, 1)
            elif query[0] == 2:
                p = query[1]
                count_ones = lazy_seg_tree.query(0, n)
                addl += p*count_ones
            else:
                ans.append(base + addl)
        return ans
```

### Solution 2:  range query is not needed, just store the total_sum for the segment tree at all times 

```py
class LazySegmentTree:
    def __init__(self, n: int, neutral: int, noop: int, arr: List[int]):
        self.neutral = neutral
        self.size = 1
        self.noop = noop
        self.n = n 
        self.arr = arr
        while self.size<n:
            self.size*=2
        self.num_flips = [noop for _ in range(self.size*2)] # number of flips in a segment
        self.counts = [neutral for _ in range(self.size*2)] # count of ones in a segment
        self.total_sum = sum(arr)
        self.build()
        
    def build(self):
        for segment_idx in range(self.n):
            v = self.arr[segment_idx]
            segment_idx += self.size - 1
            self.counts[segment_idx] = v
            self.ascend(segment_idx)

    def is_leaf_node(self, segment_right_bound, segment_left_bound) -> bool:
        return segment_right_bound - segment_left_bound == 1
    
    def calc_op(self, x: int, y: int) -> int:
        return x + y

    def modify_op(self, x: int, y: int) -> int:
        return x ^ y
    
    """
    Gives the count of a bit in a segment, which is a range. and the length of that range is represented by the segment_len.
    And it flips all the bits such as 0000110 -> 1111001, the number of 1s are now segment_len - cnt, where cnt is the current number of 1s
    So it goes from 2 -> 7 - 2 = 5
    """
    def flip_op(self, segment_len: int, cnt: int) -> int:
        return segment_len - cnt

    def propagate(self, segment_idx: int, segment_left_bound: int, segment_right_bound: int) -> None:
        # do not want to propagate if it is a leaf node or if it is no operation (means there are no updates stored there).
        if self.is_leaf_node(segment_right_bound, segment_left_bound) or self.num_flips[segment_idx] == self.noop: return
        left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
        children_segment_len = (segment_right_bound - segment_left_bound) >> 1
        self.num_flips[left_segment_idx] = self.modify_op(self.num_flips[left_segment_idx], self.num_flips[segment_idx])
        self.num_flips[right_segment_idx] =### Solution 1:  3 cases, always lowest score will be 0 from duplicates + case 1 difference between second largest and second smallest + case 2 difference between third largest and smallest + case 3 difference between largest and third smallest + O(nlogn)
 self.modify_op(self.num_flips[right_segment_idx], self.num_flips[segment_idx])
        self.counts[left_segment_idx] = self.flip_op(children_segment_len, self.counts[left_segment_idx])
        self.counts[right_segment_idx] = self.flip_op(children_segment_len, self.counts[right_segment_idx])
        self.num_flips[segment_idx] = self.noop
    
    def ascend(self, segment_idx: int) -> None:
        while segment_idx > 0:
            segment_idx -= 1
            segment_idx >>= 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.counts[segment_idx] = self.calc_op(self.counts[left_segment_idx], self.counts[right_segment_idx])
    
    def update(self, left: int, right: int, val: int) -> None:
        stack = [(0, self.size, 0)]
        segments = []
        while stack:
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # COMPLETE OVERLAP
            if segment_left_bound >= left and segment_right_bound <= right:
                self.num_flips[segment_idx] = self.modify_op(self.num_flips[segment_idx], val)
                segment_len = segment_right_bound - segment_left_bound
                self.total_sum -= self.counts[segment_idx]
                self.counts[segment_idx] = self.flip_op(segment_len, self.counts[segment_idx])
                self.total_sum += self.counts[segment_idx]
                segments.append(segment_idx)
                continue
            # PARTIAL OVERLAP
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.propagate(segment_idx, segment_left_bound, segment_right_bound)
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
        for segment_idx in segments:
            self.ascend(segment_idx)
    
    def __repr__(self) -> str:
        return f"counts: {self.counts}, num_flips: {self.num_flips}"
    
class Solution:
    def handleQuery(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        lazy_seg_tree = LazySegmentTree(n, 0, 0, nums1)
        base, addl = sum(nums2), 0
        ans = []
        for query in queries:
            if query[0] == 1:
                left, right = query[1:]
                lazy_seg_tree.update(left, right + 1, 1)
            elif query[0] == 2:
                p = query[1]
                count_ones = lazy_seg_tree.total_sum
                addl += p*count_ones
            else:
                ans.append(base + addl)
        return ans
```

## 35. Search Insert Position

### Solution 1:  binary search with bisect_left + log(n)

```py
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        return bisect.bisect_left(nums, target)
```

## 1110. Delete Nodes And Return Forest

### Solution 1:  postorder traversal on binary tree + sentinel node to get the root of the tree as forest + if node is in delete list + it's left and righ children will be added to forest + this needs to return None, because  parent node above it will have it's left or right set to None, if this was left or right node for it. 

```py
class Solution:
    def delNodes(self, root: Optional[TreeNode], to_delete: List[int]) -> List[TreeNode]:
        sentinel_node = TreeNode(0, root)
        self.forest = []
        to_delete = set([0] + to_delete)
        def postorder(node: Optional[TreeNode]) -> Optional[TreeNode]:
            if not node: return node
            left, right = postorder(node.left), postorder(node.right)
            if node.val in to_delete:
                if left:
                    self.forest.append(left)
                if right:
                    self.forest.append(right)
                return None
            node.left, node.right = left, right
            return node
        postorder(sentinel_node)
        return self.forest
```

## 540. Single Element in a Sorted Array

### Solution 1:  binary search + xor + move to right segment if nums[mid] == nums[mid^1] because then you know it is not on left segment + log(n)

```py
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        left, right = 0, len(nums) - 1
        while left < right:
            mid = (left + right) >> 1
            if nums[mid] == nums[mid ^ 1]:
                left = mid + 1
            else:
                right = mid
        return nums[left]
```

### Solution 2: bisect_left + this creates an array that will be something like [F, F, T, T, T], and so you want the first T in here + use bisect_left

```py
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        return nums[bisect.bisect_left(range(len(nums) - 1), True, key = lambda i: nums[i] != nums[i ^ 1])]
```

##

### Solution 1:  binary search with bisect_left + custom key + FFFTTT, return first True + O(nlogn)

```py
class Solution:
    def shipWithinDays(self, weights: List[int], days: int) -> int:
        n = len(weights)
        def possible(capacity: int) -> bool:
            i = 0
            for _ in range(days):
                cur_cap = 0
                while i < n and cur_cap + weights[i] <= capacity:
                    cur_cap += weights[i]
                    i += 1
                if i == n: return True
            return False
        return bisect.bisect_left(range(500_000), True, key = lambda capacity: possible(capacity))
```

```py
class Solution:
    def shipWithinDays(self, weights: List[int], days: int) -> int:
        n = len(weights)
        left, right = max(weights), sum(weights)
        def possible(capacity: int) -> bool:
            i = 0
            for _ in range(days):
                cur_cap = 0
                while i < n and cur_cap + weights[i] <= capacity:
                    cur_cap += weights[i]
                    i += 1
                if i == n: return True
            return False
        return bisect.bisect_left(range(left, right + 1), True, key = lambda capacity: possible(capacity)) + left
```

## 1259. Handshakes That Don't Cross

### Solution 1: analytical formula + constant space iterative dp + O(n) time

```py
class Solution:
    def numberOfWays(self, numPeople: int) -> int:
        cn, mod = 1, int(1e9) + 7
        for i in range(1, numPeople//2 + 1):
            cn = (2*(2*i - 1)*cn)//(i + 1)
        return cn%mod
```

### Solution 2: dynamic programming with count for N people

![visualization](images/handshakes_n_people.png)

We only need to consider the number of ways for even number of people.

TC: O(N^2)

```c++
const int MOD = 1e9+7;
class Solution {
public:
    int numberOfWays(int numPeople) {
        vector<long long> dp(numPeople+1,-1);
        function<long long(int)> dfs = [&](int n) {
            if (n<=2) return 1LL;
            if (dp[n]!=-1) return dp[n];
            long long cnt = 0;
            for (int i = 0;i<=n-2;i+=2) {
                cnt = (cnt+dfs(i)*dfs(n-i-2))%MOD;
            }
            return dp[n]=cnt;
        };
        return dfs(numPeople);
    }
};
```

### Solution 3: dynamic programming with count for N/2 pairs of people 

![visualization](images/handshakes_pairs.png)

```c++
const int MOD = 1e9+7;
class Solution {
public:
    int numberOfWays(int numPeople) {
        vector<long long> dp(numPeople/2+1,-1);
        function<long long(int)> dfs = [&](int n) {
            if (n<2) return 1LL;
            if (dp[n]!=-1) return dp[n];
            long long cnt = 0;
            for (int i = 0;i<n;i++) {
                cnt = (cnt+dfs(i)*dfs(n-i-1))%MOD;
            }
            return dp[n]=cnt;
        };
        return dfs(numPeople/2);
    }
};
```

### Solution 4: iterative DP with count for N/2 pairs of people

```c++
const int MOD = 1e9+7;
class Solution {
public:
    int numberOfWays(int numPeople) {
        int n = numPeople/2;
        vector<long long> dp(n+1,0);
        dp[0]=1;
        for (int i = 0;i<=n;i++) {
            for (int j = 0;j<i;j++) {
                dp[i] = (dp[i] + dp[j]*dp[i-j-1])%MOD;
            }
        }
        return dp[n];
    }
};
```

### Solution 5:  math.comb + binomial coefficient + number of combinations of (2n, n)

```py
class Solution:
    def numberOfWays(self, numPeople: int) -> int:
        n, mod = numPeople//2, int(1e9) + 7
        return (math.comb(2*n, n)//(n + 1))%mod
```

## 502. IPO

### Solution 1:  max heap + sort with zip and custom key + pairs

```py
class Solution:
    def findMaximizedCapital(self, k: int, w: int, profits: List[int], capital: List[int]) -> int:
        projects = sorted(zip(profits, capital), key = lambda p: p[1])
        n, i = len(projects), 0
        maxheap = []
        for _ in range(k):
            while i < n and projects[i][1] <= w:
                heappush(maxheap, -projects[i][0])
                i += 1
            if not maxheap: break
            w -= heappop(maxheap)
        return w
```

## 1675. Minimize Deviation in Array


### Solution 1: Greedy with min and max heap datastructure

```py
from heapq import heappush, heappop, heapify
class Solution:
    def minimumDeviation(self, nums: List[int]) -> int:
        
        # BUILD MIN AND MAX HEAP DATASTRUCTURES
        min_heap, max_heap = nums, [-num for num in nums]
        heapify(min_heap)
        heapify(max_heap)
        best = math.inf

        # INITIAL DEVIATION
        minv, maxv= min_heap[0], abs(max_heap[0])
        best = min(best, maxv-minv)
        
        # DOUBLE ODD ELEMENTS FROM MIN HEAP AS LONG IT MINIMIZE DEVIATION
        while min_heap[0]%2!=0 and abs(2*min_heap[0]-abs(max_heap[0])) <= abs(max_heap[0])-min_heap[0]:
            minv = min_heap[0]
            heappush(min_heap, minv*2)
            heappush(max_heap, -minv*2)
            # print(f"maximize the odd value in the array : minv={minv}, maxv={abs(max_heap[0])}")
            heappop(min_heap)
            best = min(best, abs(max_heap[0]) - min_heap[0])

        # HALF EVEN ELEMENTS FROM MAX HEAP AS LONG IT MINIMIZE DEVIATION
        while abs(max_heap[0])%2==0 and abs(min_heap[0]-(abs(max_heap[0])//2)) <= abs(max_heap[0])-min_heap[0]:
            maxv= abs(max_heap[0])
            heappush(min_heap, maxv//2)
            heappush(max_heap, -maxv//2)
            heappop(max_heap)
            # print(f"minimize the even value in the array : maxv={maxv}, minv={min_heap[0]}")
            best = min(best, abs(max_heap[0]) - min_heap[0])
            
        return best
```

### Solution 2: min heap + divide all the 2 factors from each integer

```py
class Solution:
    def minimumDeviation(self, nums: List[int]) -> int:
        def shrink(start):
            while start%2 == 0:
                start >>= 1
            return start
        heapify(minheap := [(shrink(num), -num) for num in nums])
        max_val, res = max(shrink(num) for num in nums), math.inf
        while len(minheap) > 1:
            v, num = heappop(minheap)
            res = min(res, max(max_val - v, 0))
            if v%2 == 0 and v >= abs(num): break
            v <<= 1
            max_val = max(max_val, v)
            heappush(minheap, (v, num))
        return res
```

## 652. Find Duplicate Subtrees

### Solution 1:  preorder traversal encoding of tree with string and null for empty nodes + hashmap for subtrees to find if seen more than once + recursion + O(n^2) time

```py
class Solution:
    def findDuplicateSubtrees(self, root: Optional[TreeNode]) -> List[Optional[TreeNode]]:  
        subtrees = defaultdict(list)
        def encoding_tree(node) -> str:
            if not node: return 'null'
            tree_struct = f'{node.val}, {encoding_tree(node.left)}, {encoding_tree(node.right)}'
            subtrees[tree_struct].append(node)
            return tree_struct
        encoding_tree(root)
        return [nodes[0] for nodes in subtrees.values() if len(nodes) > 1]
```

## 2574. Left and Right Sum Differences

### Solution 1:  prefix + suffix sum 

```py
class Solution:
    def leftRigthDifference(self, nums: List[int]) -> List[int]:
        n = len(nums)
        left_sum, right_sum = 0, sum(nums)
        ans = [0]*n
        for i, num in enumerate(nums):
            right_sum -= num
            ans[i] = abs(right_sum - left_sum)
            left_sum += num
        return ans
```

## 2575. Find the Divisibility Array of a String

### Solution 1:  math + modular arithmetic + remainder theorem + n = q*m + r

```py
class Solution:
    def divisibilityArray(self, word: str, m: int) -> List[int]:
        n = len(word)
        rem = 0
        ans = [0]*n
        for i, dig in enumerate(map(int, word)):
            rem = (10*rem + dig)%m
            if rem == 0: ans[i] = 1
        return ans
```

## 2576. Find the Maximum Number of Marked Indices

### Solution 1:  greedy + sort + always best to take the smallest half numbers and mark them together with largest number at jth index.  Best it will do is mark all the numbers

```py
class Solution:
    def maxNumOfMarkedIndices(self, nums: List[int]) -> int:
        n, res = len(nums), 0
        j = n - 1
        nums.sort()
        for i in reversed(range(n//2)):
            if 2*nums[i] <= nums[j]:
                res += 2
                j -= 1
        return res
```

## 2577. Minimum Time to Visit a Cell In a Grid

### Solution 1:  ping pong dijkstra algorithm + minheap + single source shortest path

```py
class Solution:
    def minimumTime(self, grid: List[List[int]]) -> int:
        if grid[1][0] > 1 and grid[0][1] > 1: return -1
        R, C = len(grid), len(grid[0])
        vis = set([(0, 0)])
        minheap = [(0, 0, 0)]
        in_bounds = lambda r, c: 0 <= r < R and 0 <= c < C
        neighborhood = lambda r, c: ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1))
        while minheap:
            time, r, c = heappop(minheap)
            if (r, c) == (R - 1, C - 1): return time
            for nr, nc in neighborhood(r, c):
                if not in_bounds(nr, nc) or (nr, nc) in vis: continue
                delta = max(0, grid[nr][nc] - time)
                vis.add((nr, nc))
                ntime = max(time + 1, grid[nr][nc] + (delta%2 == 0))
                heappush(minheap, (ntime, nr, nc))
        return -1
```

## 427. Construct Quad Tree

### Solution 1:  brute force + recursion + O(n^4) time

```py
"""
# Definition for a QuadTree node.
class Node:
    def __init__(self, val, isLeaf, topLeft, topRight, bottomLeft, bottomRight):
        self.val = val
        self.isLeaf = isLeaf
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomLeft = bottomLeft
        self.bottomRight = bottomRight
"""

"""
quad tree from a grid

isLeaf = True when all the values in the grid are the same else it is false
val = 1 if all values are 1
val = 0 if all values are 0

it will be O(n^4) time if you just brute force, that is dividing into each quad recursively can be O(n^2) operations and then checking the value of each quad can take O(n^2) time again.

Can we make the checking if the grid contains all 0s or 1s faster?  Probably if we precompute the sum over that sub grid, think 2d prefix sum for example, than can compute it in O(1) time the sum and if we know the size of the sub grid, we know if the sum == size then it is all 1s
if it is 0 then it is all 0s.  That one is easy.  But yeah

This will allow a O(n^2) solve
"""

class Solution:
    def construct(self, grid: List[List[int]]) -> 'Node':
        n = len(grid)
        grid_sum = sum(map(sum, grid))
        if grid_sum == 0 or grid_sum == n*n:
            val = 0 if grid_sum == 0 else 1
            return Node(val, True)
        quad_n = n//2
        top, bottom = grid[:quad_n], grid[quad_n:]
        topLeft, topRight, bottomLeft, bottomRight = map(lambda x: [[] for _ in range(quad_n)], range(4))
        for r, c in product(range(quad_n), range(n)):
            if c < quad_n: # LEFT SIDE
                topLeft[r].append(top[r][c])
                bottomLeft[r].append(bottom[r][c])
            else: # RIGHT SIDE
                topRight[r].append(top[r][c])
                bottomRight[r].append(bottom[r][c])
        return Node(0, False, self.construct(topLeft), self.construct(topRight), self.construct(bottomLeft), self.construct(bottomRight))
```

### Solution 2: precompute the grid sums in a 2d prefix sum + math to split each grid into quads using just bounding box 4 integer variables to define the top left row, col and the bottom right row, col + O(n^2) time

![quad tree](images/quad_tree.png)

```py
"""
# Definition for a QuadTree node.
class Node:
    def __init__(self, val, isLeaf, topLeft, topRight, bottomLeft, bottomRight):
        self.val = val
        self.isLeaf = isLeaf
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomLeft = bottomLeft
        self.bottomRight = bottomRight
"""

class Solution:
    def recurse(self, min_row: int, min_col: int, max_row: int, max_col: int) -> 'Node':
        delta = max_row - min_row
        grid_sum = self.ps[max_row][max_col] - self.ps[max_row][min_col] - self.ps[min_row][max_col] + self.ps[min_row][min_col]
        if grid_sum == 0 or grid_sum == delta*delta:
            val = 0 if grid_sum == 0 else 1
            return Node(val, True)
        mid_row, mid_col = min_row + delta//2, min_col + delta//2
        topLeft, topRight, bottomLeft, bottomRight = (min_row, min_col, mid_row, mid_col), (min_row, mid_col, mid_row, max_col), (mid_row, min_col, max_row, mid_col), (mid_row, mid_col, max_row, max_col)
        return Node(0, False, self.recurse(*topLeft), self.recurse(*topRight), self.recurse(*bottomLeft), self.recurse(*bottomRight))

    def construct(self, grid: List[List[int]]) -> 'Node':
        n = len(grid)
        self.ps = [[0]*(n+1) for _ in range(n+1)]
        # BUILD 2D PREFIX SUM
        for r, c in product(range(1,n+1),range(1,n+1)):
            self.ps[r][c] = self.ps[r-1][c] + self.ps[r][c-1] + grid[r-1][c-1] - self.ps[r-1][c-1]
        return self.recurse(0, 0, n, n)
```

## 443. String Compression

### Solution 1:  two pointers + O(n) time

```py
class Solution:
    def compress(self, chars: List[str]) -> int:
        self.idx, prev_char, cnt = 0, chars[0], 0
        def compress_char():
            chars[self.idx] = prev_char
            self.idx += 1
            if cnt > 1:
                for dig in str(cnt):
                    chars[self.idx] = dig
                    self.idx += 1
        for ch in chars:
            if ch != prev_char:
                compress_char()
                cnt, prev_char = 0, ch
            cnt += 1
        compress_char()
        return self.idx
```

## 422. Valid Word Square

### Solution 1:  iterating over a matrix + checking edge cases + O(nm)

```py
class Solution:
    def validWordSquare(self, words: List[str]) -> bool:
        n = len(words)
        for i in range(n):
            for j in range(len(words[i])):
                if j >= n or i >= len(words[j]): return False
                if words[i][j] != words[j][i]: return False
        return True
```

## 912. Sort an Array

### Solution 1:  counting sort

```py
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        min_val, max_val = min(nums), max(nums)
        counts_arr = [0]*(max_val - min_val + 1)
        for num in nums:
            counts_arr[num - min_val] += 1
        idx = 0
        for v, cnt in enumerate(counts_arr):
            v += min_val
            for _ in range(cnt):
                nums[idx] = v 
                idx += 1
        return nums
```

## 2582. Pass the Pillow

### Solution 1:  modular arithmetic + math + O(1) time

Write out an example and it is easy to see that it follows a pattern
for n = 4, as time increases from 0 to some number, you would get these locations
1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 3, 2, ....
----------

This sequence of consecutive natural numbers from 1, n  repeats every so often, but if you compare some examples
You can see that it repeats after ever gap where the gap is the number of decreases it will need to get back to person 1
But it is evident that there will be n - 2 for the size of the gap, cause they will need to give it to n - 2 people to get it
back to person 1
so if you take the result mod this n + gap, you can see that if it is smaller than n it is basically going to be part of that
increasing segment.  While if it is greater it will be decreasing, which can be model by a formula

Another way to state it is
Going forward and back to the original position,
take n * 2 - 2 steps.

```py
class Solution:
    def passThePillow(self, n: int, time: int) -> int:
        gap = n - 2
        mod = n + gap
        return time%mod + 1 if time%mod < n else 2*n - time%mod - 1
```

## 2583. Kth Largest Sum in a Binary Tree

### Solution 1:  min heap + bfs + level order traversal + deque + O(nlogk) time

```py
class Solution:
    def kthLargestLevelSum(self, root: Optional[TreeNode], k: int) -> int:
        minheap = []
        queue = deque([root])
        while queue:
            level_sum = 0
            for _ in range(len(queue)):
                node = queue.popleft()
                level_sum += node.val
                queue.extend(filter(None, (node.left, node.right)))
            heappush(minheap, level_sum)
            if len(minheap) > k:
                heappop(minheap)
        return minheap[0] if len(minheap) == k else -1
```

## 2584. Split the Array to Make Coprime Products

### Solution 1:  prime factorization + use memory to save the prime factorization of a number + O(nsqrt(m)) time

Using the idea of tracking the prime factors in the prefix product and suffix product. 
So as you encounter a number you take the prime factors of it and remove it from the count of 
those prime factors in suffix.  That is there is a suffix counter that is tracking the count
of all the prime factors of every integer in the suffix.  Basically once you encounter the prime factor
it is now both in the prefix and suffix, until you have reached an integer where all of the
instances of that prime factor are now in the prefix and no longer in the suffix indicated
by he count being 0 in suffix counter. 

If there are no prime factors that are both in prefix and still in suffix then you will have a gcd of 1
and they are coprime, so can return that index. 

Idea simplified, you are finding when no prime factors are both in prefix product and suffix product when splitting
the array into two parts. 

Here, we cannot actually compute the products as they can become astronomically large.
Products of prefix and suffix elements are co-prime if prefix elements do not share any prime factors with suffix elements.

```py
class Solution:
    def findValidSplit(self, nums: List[int]) -> int:
        n = len(nums)
        suffix_prime_counter = Counter()
        def prime_factors(num: int) -> List[int]:
                factors = []
                while num % 2 == 0:
                    factors.append(2)
                    num //= 2
                for i in range(3, int(math.sqrt(num)) + 1, 2):
                    while num % i == 0:
                        factors.append(i)
                        num //= i
                if num > 2:
                    factors.append(num)
                return factors
        factors = {}
        for num in nums:
            factors[num] = factors.get(num, prime_factors(num))
            for prime_factor in factors[num]:
                suffix_prime_counter[prime_factor] += 1
        shared_primes = set()
        for i in range(n - 1):
            for prime_factor in factors[nums[i]]:
                suffix_prime_counter[prime_factor] -= 1
                if suffix_prime_counter[prime_factor] > 0:
                    shared_primes.add(prime_factor)
                else:
                    shared_primes.discard(prime_factor)
            if len(shared_primes) == 0: return i
        return -1
```

### Solution 2: prime sieve + O(n + mloglogm) time

There are 78,498 prime factors in the range of 1 to 1,000,000. 

This solution TLE, I guess the prime sieve is too slow being that it is going to be mloglogm

```py
class Solution:
    def findValidSplit(self, nums: List[int]) -> int:
        n = len(nums)
        max_val = max(nums)
        suffix_prime_counter = Counter()
        def prime_sieve(lim):
            sieve,primes = [[] for _ in range(lim)], []
            for integer in range(2,lim):
                if not len(sieve[integer]):
                    primes.append(integer)
                    for possibly_divisible_integer in range(integer,lim,integer):
                        current_integer = possibly_divisible_integer
                        while not current_integer%integer:
                            sieve[possibly_divisible_integer].append(integer)
                            current_integer //= integer
            return sieve
        sieve_prime_factors = prime_sieve(max_val + 1)
        for num in nums:
            for prime_factor in sieve_prime_factors[num]:
                suffix_prime_counter[prime_factor] += 1
        shared_primes = set()
        for i in range(n - 1):
            for prime_factor in sieve_prime_factors[nums[i]]:
                suffix_prime_counter[prime_factor] -= 1
                if suffix_prime_counter[prime_factor] > 0:
                    shared_primes.add(prime_factor)
                else:
                    shared_primes.discard(prime_factor)
            if len(shared_primes) == 0: return i
        return -1
```

## 2585. Number of Ways to Earn Points

### Solution 1: recursive dp + knapsack dp + O(target*n^2) time, where n = 50, target = 1000

classic dp problem

look at [cses](https://cses.fi/problemset/task/1636)

```py
class Solution:
    def waysToReachTarget(self, target: int, types: List[List[int]]) -> int:
        n, mod = len(types), int(1e9) + 7
        @cache
        def dp(sum_marks: int, idx: int) -> int:
            if sum_marks == target: return 1
            if sum_marks > target or idx == n: return 0
            counts, mark = types[idx]
            ways = 0
            for take in range(counts + 1):
                ways = (ways + dp(sum_marks + take*mark, idx + 1))%mod
            return ways
        return dp(0, 0)
```

### Solution 2:  iterative dp + knapsack dp

build dp[i] to store the number of distinct ways to achieve i marks. 
for each item you need to iterate from dp[big] to dp[small] so that
it doesn't update based on previous value. 
Then just run through the number of this item can take

```py
class Solution:
    def waysToReachTarget(self, target: int, types: List[List[int]]) -> int:
        n, mod = len(types), int(1e9) + 7
        dp = [0]*(target + 1)
        dp[0] = 1
        for count, mark in types:
            for sum_ in range(target, 0, -1):
                for take in range(1, count + 1):
                    if take*mark > sum_: break
                    dp[sum_] = (dp[sum_] + dp[sum_ - take*mark])%mod
        return dp[target]
```

## 2578. Split With Minimum Sum

### Solution 1:  sort + greedy + O(nlogn) time, where n = number of digits which is n = 10

sort the digits so like 1234567 and then altneratively assign them to the two numbers to be added.
1357 + 246, this is better than 
This creates the smallest sum of digits possible for each place, so for the 4th place smallest is 1
for the 3rd place smallest is the next two smaller digits, 2, 3
and for the 2nd place it is 4, 5 and so on. 

```py
class Solution:
    def splitNum(self, num: int) -> int:
        s = ''.join(sorted(str(num)))
        return int(s[::2]) + int(s[1::2])
```

## 2579. Count Total Number of Colored Cells

### Solution 1:  math

Always adding 4 more to each increment, so it is 4*(1+2+3+4+...+n-1)

```py
class Solution:
    def coloredCells(self, n: int) -> int:
        return 1 + 4*((n - 1)*n//2)
```

## 2580. Count Ways to Group Overlapping Ranges

### Solution 1:  sort + store max reachable end to find when new groups start + O(nlogn)

power of 2 with a mod, is a fast way to compute the x**y%m with a mod. 

Also storing the max reachable for the current group, start a new group when you exceed max reachable for current group.

```py
class Solution:
    def countWays(self, ranges: List[List[int]]) -> int:
        num_groups, reachable, mod= 0, -1, int(1e9) + 7
        for s, e in sorted(ranges):
            num_groups += (s > reachable)
            reachable = max(reachable, e)
        return pow(2, num_groups, mod)
```

## 2581. Count Number of Possible Root Nodes

### Solution 1:  two dfs + reroot tree + O(n) time

This is an example of a rerooting algorithm.  The first dfs will set an arbitrary root for the tree, such as 0.
And construct the number of correct guesses when that is the root of the tree.  

A correct guess is when the tree with that root will have that edge in that order from parent node to node.

The second dfs will reroot the tree at each node, and count the number of correct guesses when that is the root of the tree. The transitions are easy because it is just
if you have are going to reroot tree to the nei_node, what you need to do is remove the guess that was saying that node -> nei_node, where this is 
parent_node -> node.  The reason for subtracting is when rerooting to nei_node, that will no longer be a true guess, but now you have a new guess that nei_node -> node. 

When it finishes a depth first search it will come back to the current node, and extend down another neighbor node.  In this case it will be like, it needs to undo it's decision to swap the nodes, because now again node -> nei_node will work, if you reroot to another nei_node. This is easy to visualize in your mind. 

```py
class Solution:
    def rootCount(self, edges: List[List[int]], guesses: List[List[int]], k: int) -> int:
        n = len(edges) + 1
        guess_counts = Counter()
        for parent, node in guesses:
            guess_counts[(parent, node)] += 1
        adj_list = [[] for _ in range(n)]
        for u, v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
        def dfs(node: int, parent_node: int) -> int:
            correct_guesses = 0
            for nei_node in adj_list[node]:
                if nei_node == parent_node: continue
                correct_guesses += guess_counts[(node, nei_node)]
                correct_guesses += dfs(nei_node, node)
            return correct_guesses
        def reroot_tree_dfs(node: int, parent_node: int) -> int:
            root_count = int(self.correct_guesses >= k)
            for nei_node in adj_list[node]:
                if nei_node == parent_node: continue
                self.correct_guesses -= guess_counts[(node, nei_node)]
                self.correct_guesses += guess_counts[(nei_node, node)]
                root_count += reroot_tree_dfs(nei_node, node)
                self.correct_guesses += guess_counts[(node, nei_node)]
                self.correct_guesses -= guess_counts[(nei_node, node)]
            return root_count
        self.correct_guesses = dfs(0, -1) # choosing arbitrary root node 0
        return reroot_tree_dfs(0, -1) 
```

## 1539. Kth Missing Positive Number

### Solution 1:

```py
class Solution:
    def findKthPositive(self, arr: List[int], k: int) -> int:
        n = len(arr)
        left, right = 0, 2_001
        def possible(target: int) -> bool:
            nonmissing_ints = bisect.bisect_left(arr, target)
            missing_ints = target - nonmissing_ints
            return missing_ints <= k
        while left < right:
            mid = (left + right + 1) >> 1
            if possible(mid):
                left = mid
            else:
                right = mid - 1
        return left
```

### Solution 2:  binary search + greedy

```py
class Solution:
    def findKthPositive(self, arr: List[int], k: int) -> int:
        n = len(arr)
        left, right = 0, n - 1
        def possible(idx: int) -> bool:
            return arr[idx] - idx - 1 < k
        while left < right:
            mid = (left + right) >> 1
            if possible(mid):
                left = mid + 1
            else:
                right = mid
        return k + left
```

## 109. Convert Sorted List to Binary Search Tree

### Solution 1:  fast and slower pointer + recursion + divide and conquer + O(nlogn) time 

This is a cool divide and conquer problem where you can find the mid pointer and then divide it into to.  Choosing the mid point to be the root of a binary search tree makes sense because it is the middle value in sorted list.  so you will place smaller elements to the left and larger elements to the right.  And this will result distributing the same number to the left and right with at most a difference of 1 node. 

It is important to be careful to set fast = head.next.next cause it needs to be advanced a little cause you want to set middle to slow.next so that you can set slow.next = None, so that the first part of th elinked list is modified, needs to be divided after all.  

```py
class Solution:
    def sortedListToBST(self, head: Optional[ListNode]) -> Optional[TreeNode]:
        if not head: return
        if not head.next: return TreeNode(head.val)
        # SLOW AND FAST POINTER, MID WILL BE AT SLOW POINTER AFTER ITERESTING TO END OF FAST POINTER
        slow, fast = head, head.next.next
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
        middle = slow.next
        slow.next = None
        root = TreeNode(middle.val)
        root.left, root.right = self.sortedListToBST(head), self.sortedListToBST(middle.next)
        return root
```

### Solution 2:  Inorder traversal to create binary search tree

It is known that an inorder traversal of a binary search tree will give a sorted list.  Thus in this case we have a sorted list, if we consider that it was created by an inorder traversal of a binary search tree.  It is possible to solve it in that manner? Basically you get the size of the linked list, and you then can consider breaking it into halves again.  Dividing it with two pointers left and right.  And find a mid pointer.  Then you can create a root node but it is inorder traversal of the sorted list we are using.  So you compute the left, then you set the value of the root node, and then you set the value of traverse right

inorder pattern
left subtree
root
right subtree

```py
class Solution:
    def getsize(self, head: Optional[ListNode]) -> int:
        cnt = 0
        while head:
            head = head.next
            cnt += 1
        return cnt
    def sortedListToBST(self, head: Optional[ListNode]) -> Optional[TreeNode]:

        size = self.getsize(head)

        def convert(left: int, right: int) -> Optional[TreeNode]:
            nonlocal head
            if left > right: return
            mid = (left + right) >> 1
            left = convert(left, mid - 1)
            root = TreeNode(head.val)
            head = head.next
            root.left, root.right = left, convert(mid + 1, right)
            return root
        
        return convert(0, size - 1)
```

### Solution 3:  DSW algorithm + tree rotations

The idea is to convert the tree into a vine (like linked list) using left rotations, and then balance it using right rotations. You can look online for the full description of the DSW algorithm.

right rotation in tree is the following

right rotation about node A, where B is left child means you do this Set A.left = B.right, so the left child of A will be what was the right child of B now.  And set B as parent of A in form of B.right = A, so A is right child of B.  And everything else stays the same. That is a right rotation, and maintains the binary search tree invariant.  

left rotation is inverse to the right rotation.  

```py
class Solution:
    def balanceBST(self, root: TreeNode) -> TreeNode:
        # PHASE 1: CREATE THE RIGHT LEANING VINE/BACKBONE 
        def right_rotation(node):
            prev_node = node
            node = node.left
            node_left_right = node.right
            node.right = prev_node
            prev_node.left = node_left_right
            return node
        def create_vine(grand):
            tmp = grand.right
            cnt = 0
            while tmp:
                if tmp.left:
                    tmp = right_rotation(tmp)
                    grand.right = tmp
                else:
                    cnt += 1
                    grand = grand.right
                    tmp = tmp.right
            return cnt
        grand_parent = TreeNode()
        grand_parent.right = root
        # count number of nodes
        n = create_vine(grand_parent)
        # PHASE 2: LEFT ROTATIONS TO GET BALANCED BINARY SEARCH TREE
        # height_perfect_balanced_tree
        h = int(math.log2(n + 1))
        # needed_nodes_perfect_balanced_tree
        m = pow(2, h) - 1
        excess = n - m 
        def left_rotation(node):
            prev_node = node
            node = node.right
            prev_node.right = node.left
            node.left = prev_node
            return node
        def compress(grand_parent, cnt):
            node = grand_parent.right
            while cnt > 0:
                cnt -= 1
                node = left_rotation(node)
                grand_parent.right = node
                grand_parent = node
                node = node.right
        compress(grand_parent, excess)
        while m > 0:
            m >>= 1
            compress(grand_parent, m)
        return grand_parent.right
```

## 2589. Minimum Time to Complete All Tasks

### Solution 1:  greedy + O(n^2) time

sort by end time and run that cpu task with the latest available times that the cpu is not already running until you have no more duration.  by running
the cpu at the latest time for a task, it increases the odd that it will contribute to other task that has a later end time but earlier start time. 

```py
class Solution:
    def findMinimumTime(self, tasks: List[List[int]]) -> int:
        tasks.sort(key = lambda task: task[1])
        running_time = [0]*2001 # each index correspond to time, and if it is running at that time it is set to 1
        total_time = 0
        for start, end, duration in tasks:
            # determine how much duration remains for this task
            for i in range(start, end + 1):
                duration -= (running_time[i])
            # complete this tasking using the latest available times, so that it increases chance of assisting with another task    
            for i in reversed(range(start, end + 1)):
                if duration <= 0: break
                if running_time[i]: continue
                duration -= 1
                running_time[i] = 1
        return sum(running_time)
```

## 2587. Rearrange Array to Maximize Prefix Score

### Solution 1:  sort array and go until prefix sum is less than or equal to 0 + O(n) time

```py
class Solution:
    def maxScore(self, nums: List[int]) -> int:
        prefix_sum = 0
        for i, num in enumerate(sorted(nums, reverse = True)):
            prefix_sum += num
            if prefix_sum <= 0: return i
        return len(nums)
```

## 2588. Count the Number of Beautiful Subarrays

### Solution 1:  prefix + bit manipulation + bitmask + O(n) time

The idea is that the prefix bit mask represents if that prefix has an odd or even number for that specific 2^i, where i is the index in the bit string representation. 
so like 0101, has odd number of bits in 0 and 2 location and even number in 1 and 3 location.  Whenever you encounter a number you just need to take the xor to update what is even and odd. Because they just flip everything.  All the current bits will flip bits in the prefix mask.  

so it is the case that a subarray is beautiful if it can be represented by 0000, that is all are even.  Now you just need to find the previous prefix mask that xor with current prefix max will give you the 0000, and with a counter table can track the number of times those prefixes occurred, each of those represent the left end points of a range. and you can compute the total count of subarrays. 

```py
class Solution:
    def beautifulSubarrays(self, nums: List[int]) -> int:
        prefix_mask = res = 0
        prefix_counts = Counter({0: 1})
        for num in nums:
            prefix_mask ^= num
            res += prefix_counts[prefix_mask]
            prefix_counts[prefix_mask] += 1
        return res
```

## 2586. Count the Number of Vowel Strings in Range

### Solution 1:  string slice + string

```py
class Solution:
    def vowelStrings(self, words: List[str], left: int, right: int) -> int:
        vowels = 'aeiou'
        res = 0
        for word in words[left:right + 1]:
            res += (word[0] in vowels and word[-1] in vowels)
        return res
```

## 23. Merge k Sorted Lists

### Solution 1:  merge with divide and conquer + O(nlogk) time

suppose you have an array of lists = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
merge the adjacent lists initially to get 
merged pairs (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12)
and stored in [0, 2, 4, 6, 8, 10, 12]
now let's merge these pairs
(0, 2), (4, 6), (8, 10), (12)
stored in [0, 4, 8, 12]
now let's merge these pairs (0, 4), (8, 12)
stored in [0, 8]
merge these pairs (0, 8)
stored in [0]

this will be logk of merges.

notice that at each step you are incrementing by 
2, 4, 8, 16, 32
and merging pairs that this distance away at each of these increment levels
1, 2, 4, 8, 16


```py
class Solution:
    def merge_two_lists(self, list1: List[Optional[ListNode]], list2: List[Optional[ListNode]]) -> List[Optional[ListNode]]:
        sentinel_node = ListNode(0)
        cur = sentinel_node
        while list1 and list2:
            val = min(list1.val, list2.val)
            cur.next = ListNode(val)
            cur = cur.next
            if list1.val == val:
                list1 = list1.next
            else:
                list2 = list2.next
        cur.next = list1 or list2
        return sentinel_node.next
        
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        k = len(lists)
        interval = 1
        while interval < k:
            for i in range(0, k - interval, 2*interval):
                lists[i] = self.merge_two_lists(lists[i], lists[i + interval])
            interval *= 2
        return lists[0] if k > 0 else None
```

## Solution 2: minheap with custom comparator

TC: O(Nlog(k)) where k is number of lists, and N is total number nodes in all lists

SC: O(K) 

```py
class Solution:
    ListNode.__lt__ = lambda self, other: self.val < other.val
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        minheap = [node for i, node in enumerate(lists) if node]
        heapify(minheap)
        sentinel_node = ListNode()
        head = sentinel_node
        while minheap:
            node = heappop(minheap)
            head.next = node
            head = head.next
            node = node.next
            if node:
                heappush(minheap, node)
        return sentinel_node.next
```


```py
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        if not lists:
            return None
        if len(lists)==1:
            return lists[0]
        mid = len(lists)//2
        leftList, rightList = self.mergeKLists(lists[:mid]), self.mergeKLists(lists[mid:])
        return self.merge(leftList,rightList)
            
    def merge(self, left, right):
        head = ListNode()
        cur = head
        while left and right:
            if left.val<right.val:
                cur.next=left
                left=left.next
            else:
                cur.next=right
                right=right.next
            cur=cur.next
        cur.next=left or right
        return head.next
```

## 382. Linked List Random Node

### Solution 1:  reservoir sampling with k = 1 + O(n) time for each getRandom

proof of reservoir sampling:

suppose you have looked at n nodes and you are considering the n + 1 node.  What is the probability of remaining on node 1
so the transition from node 1 -> node 1, what is the probability.  1/n * n/(n+1) = 1/(n+1), what does this mean
the 1/n means the probability of picking node 1 out of n nodes is 1/n if you have uniform distribution and so on.  This seems to make sense, but what is n/(n+1), will there are n/(n+1) chance you remain on node 1 and don't pick another node.  Will let's prove this statement first

proof that n/(n+1) is chance to remain for array size n
if there are n+1 elements to pick from, what are the chance you pick the current element, well let's get to an example of reservoir sampling

if you have [1,2,3,4,5,6], and I ask what is the probability that I end up with element 1 at the end of the sampling.  
first step 
1/1 chance of pick element 1
second step
there is 1/2 chance of not picking element 1
third step
there is a 2/3 chance of not picking element 1
fourth step
there is a 3/4 chance of not picking element 1, and staying in reservoir

so the n/(n+1) represents the chance of not picking node 1 and it remains in reservoir.  So it remains in reservoir in this problem. so that means for the n+1 it has 1/(n+1) chance of staying in node 1

For node 2 you have two possiblities there is a chance node 1 -> node 2 or node 2 -> node 2
node 1 -> node 2 is 1/n*(1/n + 1) = 1/(n*(n+1))
node 2 -> node 2 is 1/n*(n-1)/(n+1) = (n-1)/(n*(n+1))
summed together gives (n-1+1)/(n*(n+1)) = 1/(n*(n+1)) = 1/(n+1), so it has the same probability and that is correct it should be that prob to be at node 2

for node k+1
node k -> node k + 1 is 1/n*(k/(n+1)) = k/(n*(n+1))
node k+1 -> node k+1 is 1/n*(n-k)*(n+1) = (n-k)/(n*(n+1))
summed together is again 1/(n+1).  So that shows that all nodes have same chance of being picked in reservoir sampling algorithm.  Which is what you want complete randomness.

```py
class Solution:

    def __init__(self, head: Optional[ListNode]):
        self.head = head

    def getRandom(self) -> int:
        head, res = self.head, self.head
        n = k = 1
        while head.next:
            n += 1
            head = head.next
            if random.random() < k/n:
                res = res.next
                k += 1
        return res.val
```

### Solution 2: vector store values and rand()

```c++
int n;
vector<int> vals;
Solution(ListNode* head) {
    n=0;
    ListNode *cur = head;
    for (ListNode *cur = head;cur;cur=cur->next,n++) {
        vals.push_back(cur->val);
    }
}

int getRandom() {
    return vals[rand()%n];
}
```

### Solution 3: reservoir sampling with k=1
This algorithm can efficiently solve the problem to find a random element when dealing with unkown size of the data.  Or basically a stream
of data.  

Known limitation of rand() is that it will only work for a size of 32k integer because of RAND_MAX

Proposed solution uniform_int_distribution<int> dist(0,n-1);

```c++
ListNode *front;
Solution(ListNode* head) {
    front = head;
}

int getRandom() {
    ListNode *cur = front;
    int value = 0;
    for (int i = 1;cur;cur=cur->next,i++) {
        if (rand()%i==0) {
            value = cur->val;
        }
    }
    return value;
}
```

## 875. Koko Eating Bananas

### Solution 1:  bisect + greedy + binary search

```py
class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        return bisect.bisect_left(range(1, max(piles) + 1), True, key = lambda spd: sum((bananas+spd-1)//spd for bananas in piles) <= h) + 1
```

### Solution 2: binary search

```c++
int minEatingSpeed(vector<int>& piles, int h) {
    int lo = 1, hi = 1e9;
    while (lo<hi) {
        int mid = (lo+hi)>>1;
        int time = 0;
        for (int& bananas : piles) {
            time += ((bananas+mid-1)/mid);
        }
        if (time<=h) {
            hi = mid;
        } else {
            lo = mid+1;
        }
    }
    return lo;
}
```

```py
def minEatingSpeed(self, piles: List[int], h: int) -> int:
    lo, hi = 1, int(1e9)
    while lo<hi:
        mid = (lo+hi)//2
        if sum((bananas+mid-1)//mid for bananas in piles)<=h:
            hi = mid
        else:
            lo = mid+1
    return lo
```

## Solution 3:  binary search + greedy

```py
class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        left, right = 1, max(piles)
        possible = lambda spd: sum((bananas+spd-1)//spd for bananas in piles) <= h
        while left < right:
            mid = (left+right)>>1
            if possible(mid):
                right = mid
            else:
                left = mid+1
        return left
```

## 142. Linked List Cycle II

### Solution 1:  floyd's tortoise and hare + slow and fast pointer + math 

proof:

break the problem down into a tail and a cycle, tail leads up to the cycle and there is an entrance node that starts the cycle.  Suppose the length of the tail is F and the length of the cycle is C.  Number the nodes in the cycle starting from 0 to C - 1, 0 node is the entrance node and so on. So you will have nodes 0, 1, 2, 3, ..., C - 1 in a cycle

When the slow pointer reaches the entrance node of the cycle it wll have traveled a distance of F and the fast pointer will have travelled 2*F. In addition, it travels F distance in the cycle. Using division algorithm you can get this equation F = nC + r, where 0 <= r < C.  So n is just the number of complete laps the fast pointer makes in the cycle.  And the remainder is the current node that it will be sitting at from node 0.  So call this node r, This is current location of the fast pointer in the cycle.  Also you know that F = r (modC), where = is modular congruence operator.  

Now you know the distance of separation between slow and fast pointer at this moment.  It is C - r, because slow pointer is at node 0 and fast pointer at node r and cycle length is C.  Since fast pointer travels twice as fast, you also know it will catch up in C - r movements and intersect.  So can we prove that they intersect with mathematics. 

fast pointer distance = r + 2*(C - r) = 2*C - r which is modular congruence to (C - r)(mod C), thus they intersect in this math proof as well.  The reason why is because a congruent b (modm) => a + a congruent b (modm) Here just adding a C, C + C - r congruent to C - r (mod C).  Thus they are intersecting at same point on the cycle. 

Now we know the intersection location of slow and fast pointer to be at C - r, set the slow pointer back to the start of the linked list on the tail. and now the fast pointer will move same speed as slow pointer.  

This last phase will proof that they will intersect at the entrance node to the cycle or node 0. 
slow pointer will travel F distance to get to node 0
while fast pointer will travel F distance as well, but F = nC + r, so take current position on cycle
node that it lands on after moving F = C - r + F = C - r + nC + r = (n + 1)C.  But this is modular congruence to 0 (mod C), so it will land on node 0.  In addition you are going to complete one more lap then the laps completed in distance F. 

```py
class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow, fast = head, head
        # PHASE 1: FIND INTERSECTION AND CYCLE EXISTS
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
            if slow == fast: break
        if not fast or not fast.next: # NO CYCLE
            return None
        # PHASE 2: RESET SLOW TO START AND MOVE BOTH POINTERS AT SAME SPEED UNTIL THEY ARE PONITING TO SAME NODE
        slow = head
        while slow != fast:
            slow, fast = slow.next, fast.next
        return slow
```

### Solution 2: floyd's tortoise and hare or fast and slow pointer

```c++
ListNode *detectCycle(ListNode *head) {
    ListNode *fast = head, *slow = head;
    auto isCycle = [&]() {
        while (fast && fast->next) {
            slow=slow->next;
            fast=fast->next->next;
            if (slow==fast) {return true;}
        }
        return false;
    };
    if (!isCycle()) {
        return nullptr;
    }
    while (head!=slow) {
        slow=slow->next;
        head=head->next;
    }
    return head;
}
```

## 958. Check Completeness of a Binary Tree

### Solution 1:  recursion + count number of nodes in binary tree + use the max label + O(n)

```py
class Solution:
    def isCompleteTree(self, root: Optional[TreeNode]) -> bool:
        max_label = cnt = 0
        def dfs(node, label):
            nonlocal max_label, cnt
            max_label = max(max_label, label)
            cnt += 1
            if node.left:
                dfs(node.left, 2*label)
            if node.right:
                dfs(node.right, 2*label + 1)
        dfs(root, 1)
        return max_label == cnt
```

## 1472. Design Browser History

### Solution 1:  stacks

```py
class BrowserHistory:

    def __init__(self, homepage: str):
        self.bstk, self.fstk = [homepage], []

    def visit(self, url: str) -> None:
        self.bstk.append(url)
        self.fstk = []

    def back(self, steps: int) -> str:
        for _ in range(steps):
            if len(self.bstk) == 1: break
            self.fstk.append(self.bstk.pop())
        return self.bstk[-1]

    def forward(self, steps: int) -> str:
        for _ in range(steps):
            if not self.fstk: break
            self.bstk.append(self.fstk.pop())
        return self.bstk[-1]
```

```py
class BrowserHistory:

    def __init__(self, homepage: str):
        self.index = 0
        self.history = [homepage]

    def visit(self, url: str) -> None:
        self.history = self.history[:self.index + 1]
        self.history.append(url)
        self.index += 1

    def back(self, steps: int) -> str:
        self.index = max(0, self.index - steps)
        return self.history[self.index]

    def forward(self, steps: int) -> str:
        self.index = min(len(self.history) - 1, self.index + steps)
        return self.history[self.index]
```

### Solution 2:  dynamic array + replace values in array + store right boundary + O(1) for visit, back, forward, and initialize of object

```py
class BrowserHistory:

    def __init__(self, homepage: str):
        self.index = self.right = 0
        self.history = [homepage]

    def visit(self, url: str) -> None:
        self.index += 1
        if len(self.history) > self.index:
            self.history[self.index] = url
        else:
            self.history.append(url)
        self.right = self.index

    def back(self, steps: int) -> str:
        self.index = max(0, self.index - steps)
        return self.history[self.index]

    def forward(self, steps: int) -> str:
        self.index = min(self.right, self.index + steps)
        return self.history[self.index]
```

## 2590. Design a Todo List

### Solution 1: 3 hash tables + sorting

```py
class TodoList:

    def __init__(self):
        self.task_id = 0
        self.user_tasks = defaultdict(set) # user_id -> task_id
        self.tasks = defaultdict(list) # task_id -> (task_descript, due_date)
        self.tags_tasks = defaultdict(set) # tag -> task_id

    def addTask(self, userId: int, taskDescription: str, dueDate: int, tags: List[str]) -> int:
        self.task_id += 1
        self.user_tasks[userId].add(self.task_id)
        self.tasks[self.task_id] = (dueDate, taskDescription)
        for tag in tags:
            self.tags_tasks[tag].add(self.task_id)
        return self.task_id

    def getAllTasks(self, userId: int) -> List[str]:
        # if task id is in self the user tasks than it exists, so can add it to tasks for this user
        all_tasks = sorted([task for task in map(lambda task_id: self.tasks[task_id], self.user_tasks[userId])])
        return [task_descript for _, task_descript in all_tasks]


    def getTasksForTag(self, userId: int, tag: str) -> List[str]:
        all_tasks = sorted([self.tasks[task_id] for task_id in self.tags_tasks[tag] if task_id in self.user_tasks[userId]])
        return [task_descript for _, task_descript in all_tasks]

    def completeTask(self, userId: int, taskId: int) -> None:
        self.user_tasks[userId].discard(taskId)
```

## 605. Can Place Flowers

### Solution 1:  array + greedy

```c++
bool canPlaceFlowers(vector<int>& flowerbed, int n) {
    flowerbed.insert(flowerbed.begin(),0);
    flowerbed.push_back(0);
    for (int i = 1;i<flowerbed.size()-1 && n>0;i++) {
        if (flowerbed[i-1]+flowerbed[i]+flowerbed[i+1]==0) {
            n--;
            flowerbed[i]=1;
        }
    }
    return n==0;
}
```

```py
class Solution:
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        num_flowers = len(flowerbed)
        is_empty = lambda i: not in_bounds(i) or flowerbed[i] == 0
        in_bounds = lambda i: 0 <= i < num_flowers
        def mark(i):
            if in_bounds(i):
                flowerbed[i] = 1
        i = 0
        while i < num_flowers:
            if is_empty(i - 1) and is_empty(i) and is_empty(i + 1):
                flowerbed[i] = 1
                n -= 1
                i += 2
            else:
                i += 1
        return n <= 0
```

## 2360. Longest Cycle in a Graph

### Solution 1:  iterative search + track current visited index in current search + track globally visited vertex 

```py
class Solution:
    def longestCycle(self, edges: List[int]) -> int:
        n = len(edges)
        vis = [0]*n
        longest_cycle = -1
        def search(node):
            vis[node] = 1
            while edges[node] != -1:
                nei = edges[node]
                # visited in current search
                if vis_index[nei]: return vis_index[node] - vis_index[nei] + 1
                if vis[nei]: break # visited in a previous search
                vis[nei] = 1
                vis_index[nei] = vis_index[node] + 1
                node = nei
                if not vis[nei]:
                    vis_index[nei] = vis_index[node] + 1
                    vis[nei] = 1
                    nod = nei
            return -1
        for node in range(n):
            if vis[node]: continue
            vis_index = Counter({node: 1})
            longest_cycle = max(longest_cycle, search(node))
        return longest_cycle
```

### Solution 2:  Kahn's algorithm + Topological sort to remove the vertex not part of cycle + all that remains is cycles + find largest cycle

topological sort kind of removes these leaf nodes but they are leaf in terms of indegree is 0, and just keep pruning them

```py
class Solution:
    def longestCycle(self, edges: List[int]) -> int:
        n = len(edges)
        indegrees = [0]*n
        vis = [0]*n
        res = -1
        for nei_node in edges:
            if nei_node == -1: continue
            indegrees[nei_node] += 1
        frontier_stack = []
        for node in range(n):
            if indegrees[node] == 0: frontier_stack.append(node)
        while frontier_stack:
            node = frontier_stack.pop()
            vis[node] = 1
            nei = edges[node]
            if nei != -1:
                indegrees[nei] -= 1
                if indegrees[nei] == 0:
                    frontier_stack.append(nei)
        for node in range(n):
            if vis[node]: continue
            cnt = 0
            while not vis[node]:
                vis[node] = 1
                node = edges[node]
                cnt += 1
            res = max(res, cnt)
        return res
```

### Solution 3:  Tarjan's algorithm + find strongly connected components + size of strongly connected component is same as length of cycle in this instance

```py
class Solution:
    def longestCycle(self, edges: List[int]) -> int:
        n = len(edges)
        adj_list = [[] for _ in range(n)]
        for u in range(n):
            v = edges[u]
            if v != -1:
                adj_list[u].append(v)
        res = time = 0
        disc, low, on_stack = [0]*n, [0]*n, [0]*n
        stack = []
        def dfs(node):
            nonlocal res, time
            time += 1
            disc[node] = time
            low[node] = disc[node]
            on_stack[node] = 1
            stack.append(node)
            for nei in adj_list[node]:
                if not disc[nei]: dfs(nei)
                if on_stack[nei]: low[node] = min(low[node], low[nei])
            # found scc
            if disc[node] == low[node]:
                size = 0
                while stack:
                    snode = stack.pop()
                    size += 1
                    on_stack[snode] = 0
                    low[snode] = low[node]
                    if snode == node: break
                res = max(res, size)
        for i in range(n):
            if disc[i]: continue
            dfs(i)
        return res if res > 1 else -1
```

## 2316. Count Unreachable Pairs of Nodes in an Undirected Graph

### Solution 1:  union find + iterate through the size of each disjoint set + multiple it with all nodes in other disjoint sets

```py
class UnionFind:
    def __init__(self,n):
        self.size = [1]*n
        self.parent = list(range(n))
    
    def find(self,i):
        if i==self.parent[i]:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self,i,j):
        i, j = self.find(i), self.find(j)
        if i!=j:
            if self.size[i] < self.size[j]:
                i,j=j,i
            self.parent[j] = i
            self.size[i] += self.size[j]
            return True
        return False
    
class Solution:
    def countPairs(self, n: int, edges: List[List[int]]) -> int:
        dsu = UnionFind(n)
        for u, v in edges:
            dsu.union(u,v)
        seen = set()
        sizes = []
        for i in range(n):
            root = dsu.find(i)
            if root in seen: continue
            sizes.append(dsu.size[root])
            seen.add(root)
        sizes.sort()
        result = 0
        for sz in sizes:
            n -= sz
            result += sz*n
        return result
```

## 704. Binary Search

### Solution 1:  binary search

```py
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        i = bisect_left(nums, target)
        return i if i < len(nums) and nums[i] == target else -1
```

## 245. Shortest Word Distance III

### Solution 1:  two pointers

```py
class Solution:
    def shortestWordDistance(self, wordsDict: List[str], word1: str, word2: str) -> int:
        res = math.inf
        last_index1 = last_index2 = -math.inf
        for i, word in enumerate(wordsDict):
            if word == word1:
                last_index1 = i
                res = min(res, last_index1 - last_index2)
            if word == word2:
                last_index2 = i
                if last_index2 != last_index1:
                    res = min(res, last_index2 - last_index1)
        return res
```

## 881. Boats to Save People

### Solution:  sort + two pointers

```py
class Solution:
    def numRescueBoats(self, people: List[int], limit: int) -> int:
        n = len(people)
        people.sort()
        left, right, boats = 0, n - 1, 0
        while left <= right:
            if people[left] + people[right] <= limit:
                left += 1
            right -= 1
            boats += 1
        return boats
```

## 651. 4 Keys Keyboard

### Solution 1:  recursive dp + copy or past if buffer > 1 no reason to perform the print

```py
class Solution:
    def maxA(self, n: int) -> int:
        # 3 A is same as A, C, V
        # so for 6 it becomes better to add the A, C, V
        # can also keep using V 
        @cache
        def dp(i, buffer, size):
            if i == n: return 0
            copy = dp(i + 2, size, size) if i < n - 2 else 0
            paste = dp(i + 1, buffer, size + buffer) + buffer
            if buffer > 1:
                return max(copy, paste)
            prnt = dp(i + 1, buffer, size + 1) + 1
            return max(copy, paste, prnt)
        return dp(0, 0, 0)
```

## 87. Scramble String

### Solution 1:  iterative dp 

solve for subproblem for when length is 1, which is going to be based on if s1(i) is equal to s2(j).  This is a hard problem to understand actually. 

```py
class Solution:
    def isScramble(self, s1: str, s2: str) -> bool:
        n = len(s1)
        # (p1, p2, length)
        dp = [[[False]*(n + 1) for _ in range(n)] for _ in range(n)]
        for i, j in product(range(n), repeat = 2):
            dp[i][j][1] = s1[i] == s2[j]
        for length in range(2, n + 1):
            for k, i, j in product(range(1, length), range(n - length + 1), range(n - length + 1)):
                # current split is both scrambled of each other
                left_len, right_len = k, length - k
                prefix_scrambled = dp[i][j][left_len]
                suffix_scrambled = dp[i + left_len][j + left_len][right_len]
                dp[i][j][length] |= (prefix_scrambled & suffix_scrambled)
                # or scramble current segments
                prefix_scrambled = dp[i][j + length - left_len][left_len]
                suffix_scrambled = dp[i + left_len][j][right_len]
                dp[i][j][length] |= (prefix_scrambled & suffix_scrambled)
        return dp[0][0][n]
```

## 1402. Reducing Dishes

### Solution 1:  iterative dp

```py
class Solution:
    def maxSatisfaction(self, satisfaction: List[int]) -> int:
        n = len(satisfaction)
        satisfaction.sort()
        dp = [-math.inf]*(n + 1)
        dp[0] = 0
        for i, val in enumerate(satisfaction):
            for t in range(i + 1, 0, -1):
                dp[t] = max(dp[t], dp[t - 1] + t*val)
        return max(dp)
```

## 1444. Number of Ways of Cutting a Pizza

### Solution 1: 

```py

```

## 1254. Number of Closed Islands

### Solution 1:  dfs

```py
class Solution:
    def closedIsland(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        in_bounds = lambda r, c: 0 <= r < R and 0 <= c < C
        res = 0
        def dfs(row, col):
            stack = [(row, col)]
            grid[row][col] = 2
            is_closed = True
            while stack:
                r, c = stack.pop()
                for nr, nc in [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]:
                    if not in_bounds(nr, nc): 
                        is_closed = False
                        continue
                    if grid[nr][nc] != 0: continue
                    grid[nr][nc] = 2
                    stack.append((nr, nc))
            return is_closed
        for r, c in product(range(R), range(C)):
            if grid[r][c] != 0: continue
            res += dfs(r, c)
        return res
```

## 760. Find Anagram Mappings

### Solution 1:  deque + defaultdict + hash table

```py
class Solution:
    def anagramMappings(self, nums1: List[int], nums2: List[int]) -> List[int]:
        anagram_index = defaultdict(deque)
        n = len(nums1)
        mapping = [0]*n
        for i, num in enumerate(nums2):
            anagram_index[num].append(i)
        for i, num in enumerate(nums1):
            index = anagram_index[num].popleft()
            mapping[i] = index
        return mapping
```

# 20. Valid Parentheses

## Solution: recursive pattern with stack

```py
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        opened = {')': '(', ']': '[', '}': '{'}
        for ch in s:
            if ch in opened:
                if not stack or stack[-1] != opened[ch]: return False
                stack.pop()
            else: stack.append(ch)
        return len(stack)==0
```

## 2218. Maximum Value of K Coins From Piles

### Solution 1:  recursive dynammic programming

states are (index, remain), you want to maximize the value for at a specific index with some remain coins you can take. 
Thinking about it you can realize that you want the maximum profit for each index and remain, Then The best if if you start from index = 0 and remain = k.  That will give optimal answer for that state. 

```py
class Solution:
    def maxValueOfCoins(self, piles: List[List[int]], k: int) -> int:
        n = len(piles)
        @cache
        def dp(index, remain):
            if remain == 0 or index == n: return 0
            psum = 0
            best = dp(index + 1, remain)
            for i in range(len(piles[index])):
                if i == remain: break
                psum += piles[index][i]
                best = max(best, dp(index + 1, remain - i - 1) + psum)
            return best
        return dp(0, k)
```

## 1416. Restore The Array

### Solution 1:  space optimized + deque + O(n) time + iterative dp

so all you need is to write out an example and the pattern becomes obvious, take this as example
s = 19930613 and k = 1000
first start the empty array to be equal to 1
then you have this window = [1]
at the first 3 you have solution of 1 and store in array window = [1,1]
at 13, window = [2, 1, 1]
at 613 window = [4, 2, 1, 1]
at 0613 window = [0, 4, 2, 1, 1]
at 30613 window = [6, 0, 4, 2], pop two times from end because only 306 <= k
at 9306 window = [10, 6, 0, 4]
at 9930 window = [16, 10, 6, 0]
at 1993 window = [32, 16, 10, 6]

think of it like this if you are given 930613, you know at the current step you can take the 9, 30613 : 93, 0613 : 930, 613 those are only you can take currently because must be under k.  Now you already have computed the number of ways to form the 30613, 0613, and 613.  Those are subproblems that are already solved and you know the number of ways for the current will just be the summation of the number of ways for those 3 subproblems. 

```py
class Solution:
    def numberOfArrays(self, s: str, k: int) -> int:
        n = len(s)
        window = deque([1])
        window_sum = sum(window)
        m = 1
        cur_val = res = 0
        mod = int(1e9) + 7
        for left in reversed(range(n)):
            cur_val = (int(s[left])*m + cur_val)
            m *= 10
            while cur_val > k:
                cur_val //= 10
                m //= 10
                window_sum = (window_sum - window.pop() + mod)%mod
            if s[left] != '0':
                window.appendleft(windsow_sum)
                res = window_sum
                window_sum = (2*window_sum)%mod
            else:
                window.appendleft(0)
        return res
```

## 1046. Last Stone Weight

### Solution 1: max heap datastructure

NlogN

```c++
int lastStoneWeight(vector<int>& stones) {
    priority_queue<int> maxHeap(stones.begin(),stones.end());
    while (maxHeap.size()>1) {
        int a = maxHeap.top();
        maxHeap.pop();
        int b = maxHeap.top();
        maxHeap.pop();
        if (a==b) continue;
        maxHeap.push(abs(a-b));
    }
    return !maxHeap.empty() ? maxHeap.top() : 0;
}
```

```py
class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        for i in range(len(stones)):
            stones[i]*=-1
        heapify(stones)
        while len(stones)>1:
            x, y = abs(heappop(stones)), abs(heappop(stones))
            z = x-y
            if z>0:
                heappush(stones, -z)
        return abs(heappop(stones)) if stones else 0
```

### solution 2: bucket sort

Works if the maximum weight value is smaller than the number of elements in the array

O(N+W)

```py
class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        W = max(stones)
        buckets = [0]*(W+1)
        for s in stones:
            buckets[s]+=1
        i = lastj = W
        while i > 0:
            if buckets[i]>1:
                buckets[i] -= 2*(buckets[i]//2)
            elif buckets[i]==1:
                j = min(i-1,lastj)
                while not buckets[j]:
                    j-=1
                if j<0: return i
                lastj = j
                buckets[i-j] += 1
                buckets[i] -= 1
                buckets[j] -= 1
                i = max(i-j,j)
            else:
                i-=1
        return 0
```

## 1491. Average Salary Excluding the Minimum and Maximum Salary

### Solution 1:  min + max + sum

```py
class Solution:
    def average(self, salary: List[int]) -> float:
        return (sum(salary) - min(salary) - max(salary)) / (len(salary) - 2)
```

## 1065. Index Pairs of a String

### Solution 1:  z-algorithm + z_array + pattern searching in string + O(nm) n is the length of the string and m is the length of the dictionary

Using z-array to find all the occurences of the words in the dictionary in the given string.  z-array allows to find the index of each substring in the text that matches the pattern in O(n) time. 

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
    def indexPairs(self, text: str, words: List[str]) -> List[List[int]]:
        res = []
        for pattern in words:
            composite = f"{pattern}${text}"
            z_array = z_algorithm(composite)
            n = len(pattern)
            for i, z_val in enumerate(z_array[n + 1:]):
                if z_val == n:
                    res.append([i, i + n - 1])
        return sorted(res)
```

## 1822. Sign of the Product of an Array

### Solution 1: any

```py
class Solution:
    def arraySign(self, nums: List[int]) -> int:
        if any(num == 0 for num in nums): return 0
        neg_count = sum(1 for num in nums if num < 0)
        return 1 if neg_count%2 == 0 else -1
```

## 1498. Number of Subsequences That Satisfy the Given Sum Condition

### Solution 1:  sort + two pointers

array = [3, 3, 6, 8], target = 9
you take two pointers, and start like this
3 3 6 8
^     ^
decrement the right  pointer while the summation of the two integers is greater than target
3 3 6 8
^   ^
then you have these possible configurations that satisfy the target, representing as bits, to signify if it is in the set or not.  So for example
3 _ 6 = 101
so possible configurations is
100
110
101
111
basically there are 4 configurations, because the first bit is always active, so it is just right - left pointer
3 3 6 8
  ^ ^ 
This also satisfies the constraint so take 
10
11, two possible configurations
can see from this that it is 2^(right_index - left_index)

```py
class Solution:
    def numSubseq(self, nums: List[int], target: int) -> int:
        n = len(nums)
        mod = int(1e9) + 7
        right = n - 1
        res = 0
        nums.sort()
        for left in range(n):
            while left <= right and nums[left] + nums[right] > target:
                right -= 1
            if right < left: break
            res += pow(2, right - left, mod)
            res %= mod
        return res
```

## 1456. Maximum Number of Vowels in a Substring of Given Length

### Solution 1:  constant size sliding window

```py
class Solution:
    def maxVowels(self, s: str, k: int) -> int:
        n = len(s)
        left = res = cur = 0
        vowels = 'aeiou'
        for right in range(n):
            cur += (s[right] in vowels)
            if right >= k - 1:
                res = max(res, cur)
                cur -= (s[left] in vowels)
                left += 1
        return res
```

## 649. Dota2 Senate

### Solution 1: queue + banned counter + O(n) votes maximum so time complexity is O(n)

```py
class Solution:
    def predictPartyVictory(self, senate: str) -> str:
        banned = [0]*2
        parties = [0]*2
        queue = deque()
        for party in senate:
            x = party == 'D'
            queue.append(x)
            parties[x] += 1
        while all(parties):
            x = queue.popleft()
            if banned[x] > 0:
                banned[x] -= 1
                parties[x] -= 1
            else:
                queue.append(x)
                banned[x^1] += 1
        return "Radiant" if parties[0] else "Dire"

```

## 1538. Guess the Majority in a Hidden Array

### Solution 1: 

```py

```

## 1964. Find the Longest Valid Obstacle Course at Each Position

### Solution 1:  longest non-decreasing subsequence + dynamic programming + binary search + O(nlogn) time

main observation is that at each index you are just trying to find the longest non-decreasing subsequence that uses the current element at index i. 

Using a single array lengths works, because if you find a integer that is smaller than current integer that is at some length of longest non-decreasing subsequence, you can just replace it with the current integer, because it is smaller and you can use it to build a longer non-decreasing subsequence., right cause if you have 
[1, 1, 3], if you get a 2, it's better to replace it with that. [1, 1, 2], because now another 2 can increase this subsequence, but it wouldn't if it was still a 3.

```py
class Solution:
    def longestObstacleCourseAtEachPosition(self, obstacles: List[int]) -> List[int]:
        n = len(obstacles)
        ans = [0]*n
        lengths = [0]
        for i, v in enumerate(obstacles):
            j = bisect.bisect_right(lengths, v)
            if j == len(lengths):
                lengths.append(v)
            else:
                lengths[j] = v
            ans[i] = j
        return ans
```

## 1572. Matrix Diagonal Sum

### Solution 1:  loops + index math

need to skip the element on secondary diagonal sum if i == n - i - 1, 
-i - 1 = ~i

```py
class Solution:
    def diagonalSum(self, mat: List[List[int]]) -> int:
        n = len(mat)
        primary_diag_sum = sum(mat[i][i] for i in range(n))
        secondary_diag_sum = sum(mat[i][~i] for i in range(n) if i != n + ~i)
        return primary_diag_sum + secondary_diag_sum
```

## 311. Sparse Matrix Multiplication

### Solution 1:  brute force algorithm multiply two matrices

the time complexity of this algorithm is O(NMK), but if we have square matrices of N x N, 
then we can see it is O(N^3), so it is going to be tricky to do this in an efficient manner.


```py
class Solution:
    def multiply(self, mat1: List[List[int]], mat2: List[List[int]]) -> List[List[int]]:
        m, n, k = len(mat1), len(mat2[0]), len(mat2)
        M = [[0]*n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                M[i][j] = sum(mat1[i][ii]*mat2[ii][j] for ii in range(k))
        return M
```

## Solution 2: Optimize the matrix multiplication by skipping if the current element is 0 in the row, no reason to iterate over the column of matrix 2 and multiply

```py
class Solution:
    def multiply(self, mat1: List[List[int]], mat2: List[List[int]]) -> List[List[int]]:
        m, n, k = len(mat1), len(mat2[0]), len(mat2)
        M = [[0]*n for _ in range(m)]
        for i in range(m):
            for ii in range(k):
                if mat1[i][ii] == 0: continue
                for j in range(n):
                    M[i][j] += mat1[i][ii]*mat2[ii][j]
        return M
```

### Solution 3:  Sparse matrices + yale format + compressed sparse row (CSR) + compressed sparse column (CSC)

You can represent a sparse matrix with three 1-dimensional arrays, for values, row_indices, col_indices.  With these representation you can save on memory and number of operations on average for sparse matrices. Consider matrices where the number of nonzero elements is about the same as the count of rows or columns in the matrix. 

There are more names such as CRS (compressed row storage)

```py
class CompressedSparseRowMatrix:
    def __init__(self, matrix):
        R, C = len(matrix), len(matrix[0])
        self.values, self.col_indices, self.row_indices = [], [], [0]
        for r in range(R):
            for c in range(C):
                if matrix[r][c] == 0: continue
                self.values.append(matrix[r][c])
                self.col_indices.append(c)
            self.row_indices.append(len(self.values))

class CompressedSparseColumnMatrix:
    def __init__(self, matrix):
        R, C = len(matrix), len(matrix[0])
        self.values, self.col_indices, self.row_indices = [], [0], []
        for c in range(C):
            for r in range(R):
                if matrix[r][c] == 0: continue
                self.values.append(matrix[r][c])
                self.row_indices.append(r)
            self.col_indices.append(len(self.values))

class Solution:
    def multiply(self, mat1: List[List[int]], mat2: List[List[int]]) -> List[List[int]]:
        R, M, C = len(mat1), len(mat1[0]), len(mat2[0])
        left_matrix, right_matrix = CompressedSparseRowMatrix(mat1), CompressedSparseColumnMatrix(mat2)
        res = [[0]*C for _ in range(R)]
        for r, c in product(range(R), range(C)):
            left_col_ptr = left_matrix.row_indices[r]
            left_col_end = left_matrix.row_indices[r + 1]
            right_row_ptr = right_matrix.col_indices[c]
            right_row_end = right_matrix.col_indices[c + 1]
            while left_col_ptr < left_col_end and right_row_ptr < right_row_end:
                left_col_index = left_matrix.col_indices[left_col_ptr]
                right_row_index = right_matrix.row_indices[right_row_ptr]
                if left_col_index < right_row_index:
                    left_col_ptr += 1
                elif left_col_index > right_row_index:
                    right_row_ptr += 1
                else:
                    res[r][c] += left_matrix.values[left_col_ptr]*right_matrix.values[right_row_ptr]
                    left_col_ptr += 1
                    right_row_ptr += 1
        return res
```

## 54. Spiral Matrix

### Solution 1:  many loops

```cpp
class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        int R = matrix.size(), C = matrix[0].size();
        vector<int> spiral;
        int top = 0, left = 0, bot = R-1, right = C-1;
        while (right>=left) {
            // go to the right on the topmost row
            for (int i = left;i<=right;i++) {
                spiral.push_back(matrix[top][i]);
            }
            top++;
            if (top>bot) {
                break;
            }
            // go to the bottom on the righmost column
            for (int i = top;i<=bot;i++) {
                spiral.push_back(matrix[i][right]);
            }
            right--;
            if (right<left) {
                break;
            }
            // go to the left on the bottomost row
            for (int i = right;i>=left;i--) {
                spiral.push_back(matrix[bot][i]);
            }
            bot--;
            if (bot<top) {
                break;
            }
            // go to the top on the leftmost column
            for (int i = bot;i>=top;i--) {
                spiral.push_back(matrix[i][left]);
            }
            left++;
        }
        return spiral;
    }
};
```

## 1035. Uncrossed Lines

### Solution 1:  iterative dynamic programming + O(n) memory optimized + O(n^2) time

two loops, but at each index i and index j, consider the possiblity of connecting those two index if they match.   For that you want to take the value from the previous i - 1 iteration and look back one and see how much uncrossed connections were possible up to that j - 1 point, then update j with the addition of the crossing.  And if they are not equal then just want to update based on i - 1 or just look at j - 1 on current i.  

That way dp(i) is the maximum number of uncrossed connections betweent he two integer arrays up to the ith index. dp(2) = 1, means can have 1 uncrossed connection at index 2 in the nums2 integer array, doesn't matter, can be either integer array.  

so dp(0) is always equal to 0, because can always connect line with the 0th element and it should always set it to 1 but for the dp(1) state. 

![uncrossed lines](images/uncrossing_lines.png)

```py
class Solution:
    def maxUncrossedLines(self, nums1: List[int], nums2: List[int]) -> int:
        n1, n2 = len(nums1), len(nums2)
        dp = [0]*(n2 + 1)
        for i in range(n1):
            ndp = [0]*(n2 + 1)
            for j in range(n2):
                if nums1[i] == nums2[j]:
                    ndp[j + 1] = max(dp[j + 1], dp[j] + 1)
                else:
                    ndp[j + 1] = max(dp[j + 1], ndp[j])
            dp = ndp
        return dp[-1]
```

## 1799. Maximize Score After N Operations

### Solution 1:  dp bitmask + backtrack + memoization + O(2^(2n)*n^2*log(A)) where A is largest integer in nums array

This dp bitmask starts from the end_mask which will have a 1 set for every single bit to say that all integers have been picked.
so 0 => not picked, 1 => picked
Then you set the dp[end_Mask] = 0, cause 0 is the score for having all picked (counter intuitive but we are going backwards). 
Then it decreases the bitmask, and if the used is odd, then that means you have something like 1110, which is not feasible because picking in pairs. So skip these states, since they are impossible. 
Then it will reach a state where 1100, two bits are 0, Now we are going to use this to find all the pairs of integers that we can pick, which will just be the last two integers, and then it will add from that previous state 1111, to get the value for the state 1100, supposing that already formed 1 pair, so it will use 2*gcd(x, y), So now we are supposing if we moved from 1111 -> 1100 state this is the maximum amount of score that could be achieved, and we are really saying the maximum.  This is kinda weird to be honest, going backwards, but think when you get to 0000, it will get the max based on all the previous states with two bits set to 1. Just think in order it is supposing if you pick those two first you know.  So eally it is saying if you picked the two bits, last this is the best you could get for current state. 

```py
class Solution:
    def maxScore(self, nums: List[int]) -> int:
        n = len(nums)
        end_mask = (1 << n) - 1
        dp = [-math.inf]*(1 << n)
        for bitmask in range(end_mask, -1, -1):
            if bitmask == end_mask: # base case
                dp[bitmask] = 0
                continue
            used = bitmask.bit_count()
            if used&1: continue # impossible state, must be even
            pairs_formed = used // 2
            for i in range(n):
                if (bitmask >> i) & 1: continue # skip if it is set to a 1, only want 0s
                for j in range(i + 1, n):
                    if (bitmask >> j) & 1: continue
                    nmask = bitmask | (1 << i) | (1 << j)
                    dp[bitmask] = max(dp[bitmask], dp[nmask] + (pairs_formed + 1)*math.gcd(nums[i], nums[j]))
        return dp[0]
```

### Solution 2:  dp bitmask + forward

This one is more straightforward since it moves forward, and considers the first, second, and so on pairs formed. 
but because it is increasing bitmask, you need to look at a smaller bitmask so you need to set 1s to 0s to get a smaller bitmask. 

```py
class Solution:
    def maxScore(self, nums: List[int]) -> int:
        n = len(nums)
        end_mask = (1 << n) - 1
        dp = [-math.inf]*(1 << n)
        dp[0] = 0
        for bitmask in range(1, 1 << n):
            used = bitmask.bit_count()
            if used & 1: continue
            pairs_formed = used // 2
            for i in range(n):
                if not ((bitmask >> i) & 1): continue # skip if set to a 0
                for j in range(i + 1, n):
                    if not ((bitmask >> j) & 1): continue
                    nmask = bitmask ^ (1 << i) ^ (1 << j)
                    dp[bitmask] = max(dp[bitmask], dp[nmask] + pairs_formed*math.gcd(nums[i], nums[j]))
        return dp[end_mask]
```

## 1721. Swapping Nodes in a Linked List

### Solution 1:  single pass + two pointers

one pointer is current node as traverse to the last node The other pointer is one that is k nodes behind the current node, that way it will give the kth node from the last node in the linked list.  

![example](images/swapping_nodes.png)

```py
class Solution:
    def swapNodes(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        cur_node = head
        list_len = 0
        while cur_node:
            list_len += 1
            if list_len == k:
                front_node = cur_node
                end_node = head
            if list_len > k:
                end_node = end_node.next
            cur_node = cur_node.next
        front_node.val, end_node.val = end_node.val, front_node.val
        return head
```

## 1044. Longest Duplicate Substring

### Solution 1:  suffix array + longest common prefix array + O(n) time

```py
from typing import List
def radix_sort(leaderboard: List[int], equivalence_class: List[int]) -> List[int]:
    n = len(leaderboard)
    bucket_size = [0]*n
    for eq_class in equivalence_class:
        bucket_size[eq_class] += 1
    bucket_pos = [0]*n
    for i in range(1, n):
        bucket_pos[i] = bucket_pos[i-1] + bucket_size[i-1]
    updated_leaderboard = [0]*n
    for i in range(n):
        eq_class = equivalence_class[leaderboard[i]]
        pos = bucket_pos[eq_class]
        updated_leaderboard[pos] = leaderboard[i]
        bucket_pos[eq_class] += 1
    return updated_leaderboard

def suffix_array(s: str) -> List[int]:
    n = len(s)
    arr = [None]*n
    for i, ch in enumerate(s):
        arr[i] = (ch, i)
    arr.sort()
    leaderboard = [0]*n
    equivalence_class = [0]*n
    for i, (_, j) in enumerate(arr):
        leaderboard[i] = j
    equivalence_class[leaderboard[0]] = 0
    for i in range(1, n):
        left_segment = arr[i-1][0]
        right_segment = arr[i][0]
        equivalence_class[leaderboard[i]] = equivalence_class[leaderboard[i-1]] + (left_segment != right_segment)
    is_finished = False
    k = 1
    while k < n and not is_finished:
        for i in range(n):
            leaderboard[i] = (leaderboard[i] - k + n)%n # create left segment, keeps sort of the right segment
        leaderboard = radix_sort(leaderboard, equivalence_class) # radix sort for the left segment
        updated_equivalence_class = [0]*n
        updated_equivalence_class[leaderboard[0]] = 0
        for i in range(1, n):
            left_segment = (equivalence_class[leaderboard[i-1]], equivalence_class[(leaderboard[i-1]+k)%n])
            right_segment = (equivalence_class[leaderboard[i]], equivalence_class[(leaderboard[i]+k)%n])
            updated_equivalence_class[leaderboard[i]] = updated_equivalence_class[leaderboard[i-1]] + (left_segment != right_segment)
            is_finished &= (updated_equivalence_class[leaderboard[i]] != updated_equivalence_class[leaderboard[i-1]])
        k <<= 1
        equivalence_class = updated_equivalence_class
    return leaderboard, equivalence_class

def lcp(leaderboard: List[int], equivalence_class: List[int], s: str) -> List[int]:
    n = len(s)
    lcp = [0]*(n-1)
    k = 0
    for i in range(n-1):
        pos_i = equivalence_class[i]
        j = leaderboard[pos_i - 1]
        while s[i + k] == s[j + k]: k += 1
        lcp[pos_i-1] = k
        k = max(k - 1, 0)
    return lcp

class Solution:
    def longestDupSubstring(self, s: str) -> str:
        s += '$'
        n = len(s)
        p, c = suffix_array(s)
        lcp_arr = lcp(p, c, s)
        idx = max(range(n - 1), key = lambda i: lcp_arr[i])
        len_ = lcp_arr[idx]
        suffix_index = p[idx]
        return s[suffix_index: suffix_index + len_]
```

## 253. Meeting Rooms II

### Solution 1:  line sweep + sort

```py
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        events = []
        for s, e in intervals:
            events.append((s, 1))
            events.append((e, -1))
        events.sort()
        num_rooms = res = 0
        for ev, delta in events:
            num_rooms += delta
            res = max(res, num_rooms)
        return res
```

## 347. Top K Frequent Elements

### Solution 1:  sort + group + O(nlogn) time

```py
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        nums.sort()
        freq = [(len(list(grp)), elem) for elem, grp in groupby(nums)]
        freq.sort(reverse=True)
        return [elem for _, elem in freq[:k]]
```

### Solution 2:  counter sort for frequency array + hash table to store elements with specific frequency + O(n) time

This solution will scale poorly if the integer range increases vastly such as it allows 10**9, A better solution for that if you still have a reasonable amount of elements is to use dictionary at that point, instead of lists.

```py
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        delta = 10**4
        sz = 2 * delta + 1
        freq, freq_freq = [0]*sz, [0]*sz
        freq_map = defaultdict(list)
        for num in nums:
            freq[num + delta] += 1
        # counting sort for frequency
        # map to get elements that had that frequency for recovery
        for i in range(sz):
            if freq[i] == 0: continue
            freq_freq[freq[i]] += 1
            freq_map[freq[i]].append(i - delta)
        res = []
        for i in reversed(range(sz)):
            if freq_freq[i] == 0: continue
            for _ in range(freq_freq[i]):
                if k == 0: break
                res.append(freq_map[i].pop())
                k -= 1
            if k == 0: break
        return res
```

## 703. Kth Largest Element in a Stream

### Solution 1: min heap datastructure of size k

This can store the kth largest element, it will be the smallest element in the min heap of size k

nlogn + mlogk, m=calls to add, n=len(nums)

```py
class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        self.heap = nums
        heapify(self.heap)
        while len(self.heap)>k:
            heappop(self.heap)
        self.k = k

    def add(self, val: int) -> int:
        heappush(self.heap,val)
        if len(self.heap)>self.k:
            heappop(self.heap)
        return self.heap[0]
```

## 837. New 21 Game

### Solution 1:  dp + sliding window + probability

![image](images/new_21_game.PNG)

```py
class Solution:
    def new21Game(self, n: int, k: int, m: int) -> float:
        if k == 0 or n >= k + m - 1: return 1.0
        window_sum = 1.0
        dp = [1.0] + [0.0]*n
        for i in range(1, n + 1):
            dp[i] = window_sum/m
            if i < k:
                window_sum += dp[i]
            if i - m  >= 0:
                window_sum -= dp[i - m]
        return sum(dp[k:])
```

## 1406. Stone Game III

### Solution 1:  dynamic programming + recurrence relation

dp[i] = max score that current player can attain with taking of the ith to i+3th stone. 

score of current player is there score minus score of other player

```py
class Solution:
    def stoneGameIII(self, stoneValue: List[int]) -> str:
        n = len(stoneValue)
        dp = [-math.inf]*n + [0]
        for i in range(n - 1, -1, -1):
            score = 0
            for j in range(i, min(i + 3, n)):
                score += stoneValue[j]
                dp[i] = max(dp[i], score - dp[j + 1])
        return 'Alice' if dp[0] > 0 else 'Bob' if dp[0] < 0 else 'Tie'
```

## 1140. Stone Game II

### Solution 1: dynamic programming + suffix sum + maximize stones for each player

dp[0][1] is the player that takes first and must take when M = 1 initially

![images](images/stone_game_2_1.png)
![images](images/stone_game_2_2.png)

```py
class Solution:
    def stoneGameII(self, piles: List[int]) -> int:
        n = len(piles)
        dp = [[0]*(n + 1) for _ in range(n + 1)]
        suffix_sum = [0]*(n + 1)
        for i in range(n - 1, -1, -1):
            suffix_sum[i] = piles[i] + suffix_sum[i + 1]
        for i in range(n + 1):
            dp[i][-1] = suffix_sum[i]
        for i, j in product(reversed(range(n)), repeat = 2):
            # j is M
                for x in range(1, 2*j + 1): # integers can take
                    if i + x > n: break
                    dp[i][j] = max(dp[i][j], suffix_sum[i] - dp[i + x][max(j, x)])
        return dp[0][1] # 0th index, M = 1
```

## 1547. Minimum Cost to Cut a Stick

### Solution 1:  dynamic programming + O(n^3)

the length is more of the number of cut points, so for instance
length = 2 means a stick with two endpoints that are cut points
length = 3 means a stick with two endpoints and middle cut point
length = 4, now you have two cut options between it, and so on want the minimum value, solve the recurrence relation

![images](images/minimum_stick_cuts_1.png)
![images](images/minimum_stick_cuts_2.png)

```py
class Solution:
    def minCost(self, n: int, cuts: List[int]) -> int:
        cuts.extend([0, n])
        cuts.sort()
        m = len(cuts)
        dp = [[math.inf] * m for _ in range(m)]
        for i in range(m - 1):
            dp[i][i + 1] = 0 # length = 2 stick does not need be cut
        for i in range(m - 2):
            dp[i][i + 2] = cuts[i + 2] - cuts[i] # only cut in middle
        for len_ in range(4, m + 1):
            for i in range(m):
                j = i + len_ - 1
                if j >= m: break
                for k in range(i + 1, j):
                    dp[i][j] = min(dp[i][j], dp[i][k] + dp[k][j] + cuts[j] - cuts[i])
        return dp[0][m - 1] # cost of cutting the entire stick
```

## 1603. Design Parking System

### Solution 1:  array + counter

```py
class ParkingSystem:

    def __init__(self, big: int, medium: int, small: int):
        self.free = [big, medium, small]

    def addCar(self, carType: int) -> bool:
        if self.free[carType - 1] == 0: return False
        self.free[carType - 1] -= 1
        return True
```

## 348. Design Tic-Tac-Toe

### Solution 1:  counter + hash table + O(1) for each move

```py
class TicTacToe:

    def __init__(self, n: int):
        self.n = n
        self.row_count, self.col_count = [0] * n, [0] * n
        self.diags = [0] * 2
        
    def winner(self, row: int, col: int, player: int) -> bool:
        size = self.n if player == 1 else -self.n
        return self.row_count[row] == size or self.col_count[col] == size or size in self.diags

    def move(self, row: int, col: int, player: int) -> int:
        delta = 1 if player == 1 else -1
        self.row_count[row] += delta
        self.col_count[col] += delta
        if row - col == 0:
            self.diags[0] += delta
        if row + col == self.n - 1:
            self.diags[1] += delta
        return player if self.winner(row, col, player) else 0
```

## 1230. Toss Strange Coins

### Solution 1:  dynamic programming + mathematics + probabilities + addition and multiplication rule of probabilities

The addition rule of probability is used when you are trying to find the probability of one event or the other happening. It is also known as the "or" rule. For example, if you wanted to know the probability of rolling a 1 or a 2 on a fair die, you would use the addition rule.The multiplication rule of probability is used when you are trying to find the probability of one event happening and another event happening. It is also known as the "and" rule. For example, if you wanted to know the probability of rolling a 1 on the first roll and a 2 on the second roll of a fair die, you would use the multiplication rule.In general, you use the addition rule when the events are mutually exclusive, which means that they cannot happen at the same time, and you use the multiplication rule when the events are dependent, which means that one event affects the probability of the other event happening.

```py
class Solution:
    def probabilityOfHeads(self, prob: List[float], target: int) -> float:
        n = len(prob)

        @cache
        def dp(index, heads):
            if index == n: return heads == target
            skip = dp(index + 1, heads) * (1 - prob[index])
            take = dp(index + 1, heads + 1) * prob[index]
            return skip + take
            
        return dp(0, 0)
```

### Solution 2:  iterative dynamic programming

```py
class Solution:
    def probabilityOfHeads(self, prob: List[float], target: int) -> float:
        n = len(prob)
        dp = [[0.0] * (target + 1) for _ in range(n + 1)]
        dp[0][0] = 1.0
        for i, j in product(range(1, n + 1), range(target + 1)):
            tails = dp[i - 1][j] * (1 - prob[i - 1])
            heads = dp[i - 1][j - 1] * prob[i - 1] if j > 0 else 0.0
            dp[i][j] = heads + tails
        return dp[n][target]
```

### Solution 3:  space optimized + O(target) space

```py
class Solution:
    def probabilityOfHeads(self, prob: List[float], target: int) -> float:
        n = len(prob)
        dp = [1.0] + [0.0] * target
        for i in range(n):
            ndp = [0.0] * (target + 1)
            for j in range(target + 1):
                if j > i + 1: break
                tails = dp[j] * (1 - prob[i])
                heads = dp[j - 1] * prob[i] if j > 0 else 0.0
                ndp[j] = heads + tails
            dp = ndp
        return dp[-1]
```

## 2101. Detonate the Maximum Bombs

### Solution 1:  multisource bfs from each bomb + directed graph 

Therefore, the original problem can be transformed into a graph traversal problem where we calculate the total number of reachable nodes from each node i

Find all the nodes that can be reached from each node. 

find maximum number of nodes reachable from any node.

directed edge means from 1 -> 2 means bomb 1 can detonate bomb 2

so if node is within radius of bomb 1 then add directed edge

```py
class Solution:
    def maximumDetonation(self, bombs: List[List[int]]) -> int:
        euclidean_dist = lambda p1, p2: (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2
        n = len(bombs)
        adj_list = [[] for _ in range(n)]
        for i in range(n):
            x1, y1, r1 = bombs[i]
            for j in range(n):
                if i == j: continue
                x2, y2, r2 = bombs[j]
                if euclidean_dist((x1, y1), (x2, y2)) <= r1 * r1:
                    adj_list[i].append(j)
        res = 0
        for start_bomb in range(n):
            visited = [0] * n
            visited[start_bomb] = 1
            queue = deque([start_bomb])
            while queue:
                bomb = queue.popleft()
                for nei in adj_list[bomb]:
                    if visited[nei]: continue
                    visited[nei] = 1
                    queue.append(nei)
            res = max(res, sum(visited))
        return res
```

## 1376. Time Needed to Inform All Employees

### Solution 1:  bfs + directed graph

```py
class Solution:
    def numOfMinutes(self, n: int, headID: int, manager: List[int], informTime: List[int]) -> int:
        adj_list = [[] for _ in range(n)]
        for u, v in enumerate(manager):
            if v == -1: continue
            adj_list[v].append(u)
        res = 0
        queue = deque([(headID, 0)])
        while queue:
            node, time_ = queue.popleft()
            res = max(res, time_)
            for nei in adj_list[node]:
                queue.append((nei, time_ + informTime[node]))
        return res
```

## 547. Number of Provinces

### Solution 1:  dfs + connected components + visited array + adjacency matrix representation + undirected graph + O(n^2) time

```py
class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        # find number of connected components
        n = len(isConnected)
        vis = [0] * n
        res = 0
        for i in range(n):
            if vis[i]: continue
            res += 1
            vis[i] = 1
            stack = [i]
            while stack:
                node = stack.pop()
                for nei, connected in enumerate(isConnected[node]):
                    if not connected or vis[nei]: continue
                    vis[nei] = 1
                    stack.append(nei)
        return res
```

## 1232. Check If It Is a Straight Line

### Solution 1: math + equation of a line + slope of a line

To deal with when vertical or hotizontal line where dx or dy is euqal to 0, then you can use the formula
dy1/dx1 = dy2/dx2
dy1*dx2 = dx1*dy2

```py
class Solution:
    def checkStraightLine(self, coordinates: List[List[int]]) -> bool:
        coordinates.sort()
        n = len(coordinates)
        delta_x = lambda p1, p2: p1[0] - p2[0]
        delta_y = lambda p1, p2: p1[1] - p2[1]
        dx, dy = delta_x(coordinates[0], coordinates[1]), delta_y(coordinates[0], coordinates[1])
        for i in range(2, n):
            dxx, dyy = delta_x(coordinates[0], coordinates[i]), delta_y(coordinates[0], coordinates[i])
            if dx * dyy != dy * dxx: return False
        return True
```

## 1502. Can Make Arithmetic Progression From Sequence

### Solution 1:  difference array + sort + all to determine if all elements are equal in array

```py
class Solution:
    def canMakeArithmeticProgression(self, arr: List[int]) -> bool:
        arr.sort()
        n = len(arr)
        diff = [arr[i] - arr[i - 1] for i in range(1, n)]
        return all(diff[0] == diff[i] for i in range(1, n - 1))
```

## 1318. Minimum Flips to Make a OR b Equal to c

### Solution 1:  bit manipulation + check number flips bit by bit.

```py
class Solution:
    def minFlips(self, a: int, b: int, c: int) -> int:
        res = 0
        for i in range(32):
            if (c >> i) & 1:
                if (((a >> i) & 1) | ((b >> i) & 1)) == 0:
                    res += 1
            else:
                res += ((a >> i) & 1)
                res += ((b >> i) & 1)
        return res
```

## 1351. Count Negative Numbers in a Sorted Matrix

### Solution 1:  binary search + bisect_left + matrix

```py
class Solution:
    def countNegatives(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        res = 0
        for row in grid:
            c = bisect.bisect_left(row, True, key = lambda x: x < 0)
            res += C - c
        return res
```

## 744. Find Smallest Letter Greater Than Target

### Solution 1:  binary search + bisect_right

```py
class Solution:
    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        i = bisect.bisect_right(letters, target)
        return letters[i if i < len(letters) else 0]
```

## 1802. Maximum Value at a Given Index in a Bounded Array

### Solution 1:  binary search + bisect_right + math + sum of 1,2,3,..,n

```py
class Solution:
    def maxValue(self, n: int, index: int, maxSum: int) -> int:
        natural_sum = lambda x: x * (x + 1) // 2
        def possible(target):
            left_segment_len, right_segment_len = index, n - index - 1
            x1, x2, x3, x4 = map(lambda x: natural_sum(x - 1), [target, target, max(1, target - left_segment_len), max(1, target - right_segment_len)])
            sum_ones = max(0, left_segment_len - target + 1) + max(0, right_segment_len - target + 1) 
            sum_ = x1 + x2 - x3 - x4 + sum_ones + target
            return sum_
        return bisect.bisect_right(range(maxSum + 1), maxSum, key = lambda x: possible(x)) - 1
```

## 1146. Snapshot Array

### Solution 1:  dictionary + binary search + memory optimized for sparse arrays

This implementation is memory optimize for sparse arrays, because if the values are initialized at 0 and are never changed, it doesn't do anything unless it has been set. it is optimized for setting values to 0 which is redundant as well.  

This snapshot array can be used when needing to store versions of an array, because need to access historical records of an array. 

```py
class SnapshotArray:

    def __init__(self, length: int):
        self.arr = {}
        self.snapshot_arr = defaultdict(list)
        self.version = 0

    # O(1)
    def set(self, index: int, val: int) -> None:
        if self.snapshot_arr[index] and self.snapshot_arr[index][-1][0] == self.version: 
            self.snapshot_arr[index][-1] = (self.version, val)
        elif self.snapshot_arr[index] and self.snapshot_arr[index][-1][1] == val: 
            return
        elif val != 0 or self.snapshot_arr[index]:
            self.snapshot_arr[index].append((self.version, val))

    # O(1)
    def snap(self) -> int:
        self.version += 1
        return self.version - 1

    # O(log(n))
    def get(self, index: int, snap_id: int) -> int:
        if not self.snapshot_arr[index] or self.snapshot_arr[index][0][0] > snap_id: return 0
        i = bisect.bisect_right(self.snapshot_arr[index], (snap_id, math.inf)) - 1
        return self.snapshot_arr[index][i][1]
```

## 228. Summary Ranges

### Solution 1: loop

```py
class Solution:
    def summaryRanges(self, nums: List[int]) -> List[str]:
        ranges = []
        for i, num in enumerate(nums):
            if len(ranges) == 0 or ranges[-1][-1] + 1 != num:
                ranges.append([num, num])
            else:
                ranges[-1][-1] = num
        return ["->".join(map(str,(x,y))) if x != y else str(x) for x,y in ranges]
```

## 530. Minimum Absolute Difference in BST

### Solution 1:  recursive inorder traversal + binary search tree

```py
class Solution:
    def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
        res, prev = math.inf, -math.inf
        def inorder(node):
            nonlocal res, prev
            if not node: return
            inorder(node.left)
            res = min(res, node.val - prev)
            prev = node.val
            inorder(node.right)
        inorder(root)
        return res
```

## 1161. Maximum Level Sum of a Binary Tree

### Solution 1:  bfs + binary tree + filter

```py
class Solution:
    def maxLevelSum(self, root: Optional[TreeNode]) -> int:
        res, level = -math.inf, 1
        queue = deque([root])
        lv = 0
        while queue:
            level_sum = 0
            lv += 1
            for _ in range(len(queue)):
                node = queue.popleft()
                level_sum += node.val
                queue.extend(filter(None, (node.left, node.right)))
            if level_sum > res:
                res = level_sum
                level = lv
        return level
```

## 163. Missing Ranges

### Solution 1:  loop + decision

```py
class Solution:
    def findMissingRanges(self, nums: List[int], lower: int, upper: int) -> List[List[int]]:
        prev = lower - 1
        res = []
        for num in nums:
            if num - prev > 1:
                res.append([prev + 1, num - 1])
            prev = num
        if upper > prev:
            res.append([prev + 1, upper])
        return res
```

## 1187. Make Array Strictly Increasing

### Solution 1:  sort + dynamic programming + memoization + memory optimized + dictionary + binary search

dp[i] = minimum number of operations to have strictly increasing array ending with ith value
Do this for each strictly increasing array, keep adding one element to it and answer the problem above. 

binary search because you want to find the smallest number that is greater than previous value. 

```py
class Solution:
    def makeArrayIncreasing(self, arr1: List[int], arr2: List[int]) -> int:
        arr2 = sorted(set(arr2))
        n2 = len(arr2)
        dp = {-1: 0}
        for num in arr1:
            ndp = defaultdict(lambda: math.inf)
            for v in dp.keys():
                if num > v:
                    ndp[num] = min(ndp[num], dp[v])
                i = bisect.bisect_right(arr2, v)
                if i < n2:
                    ndp[arr2[i]] = min(ndp[arr2[i]], dp[v] + 1)
            dp = ndp
        return min(dp.values()) if dp else -1
```

## 1569. Number of Ways to Reorder Array to Get Same BST

### Solution 1:  combinatorics + math + left and right subtree

The part that I really couldn't figure out is that the number orderings will be that of the elements that would be in left or right array.  But additionally it is the number of interleavings which can be found by considering the positions, and there are certain many number of ways to choose some left elements to fit in the available positions. 

```py
class Solution:
    def numOfWays(self, nums: List[int]) -> int:
        mod = int(1e9) + 7
        def dfs(arr):
            n = len(arr)
            if n <= 1: return 1
            left_arr = [x for x in arr if x < arr[0]]
            right_arr = [x for x in arr if x > arr[0]]
            left_size, right_size = len(left_arr), len(right_arr)
            left_orderings, right_orderings = dfs(left_arr), dfs(right_arr)
            interleaving_ways = math.comb(left_size + right_size, left_size)
            return left_orderings * right_orderings * interleaving_ways % mod
        return dfs(nums) - 1
```

## 1732. Find the Highest Altitude

### Solution 1:  accumulate for prefix sum + initial value is 0 + max

```py
class Solution:
    def largestAltitude(self, gain: List[int]) -> int:
        return max(accumulate(gain, initial = 0))
```

## 2090. K Radius Subarray Averages

### Solution 1:  sliding window

```py
class Solution:
    def getAverages(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        res = [-1] * n
        window_sum = 0
        for i in range(n):
            window_sum += nums[i]
            if i >= 2*k:
                res[i - k] = window_sum // (2 * k + 1)
                window_sum -= nums[i - 2 * k]
        return res
```

## 1214. Two Sum BSTs

### Solution 1:  dfs stack through one bst + binary search through other bst + O(n^2) worst case, but O(nlogn) average

```py
class Solution:
    def twoSumBSTs(self, root1: Optional[TreeNode], root2: Optional[TreeNode], target: int) -> bool:
        def search(value, root):
            if not root: return False
            if value + root.val > target: return search(value, root.left)
            if value + root.val < target: return search(value, root.right)
            return True
        stack = [root1]
        while stack:
            node1 = stack.pop() 
            if search(node1.val, root2): return True
            stack.extend(filter(None, (node1.left, node1.right)))
        return False
```

## 1027. Longest Arithmetic Subsequence

### Solution 1:  iterative dynamic programming + maximum value of dp

dp[i][j] = maximum length of arithmetic subsequence, with nums[i], and diff = j

So recurrence relation is 
dp[i][nums[i] - nums[j]] = max(dp[i][nums[i]-nums[j]], dp[j][nums[i] - nums[j]] + 1)

```py
class Solution:
    def longestArithSeqLength(self, nums: List[int]) -> int:
        n = len(nums)
        m = 2 * (max(nums) - min(nums))
        dp = [[1] * (m + 1) for _ in range(n)]
        res = 0
        for i in range(1, n):
            for j in range(i):
                delta = nums[i] - nums[j] + m // 2
                dp[i][delta] = max(dp[i][delta], dp[j][delta] + 1)
                res = max(res, dp[i][delta])
        return res
```

also can use dictionary

```py
class Solution:
    def longestArithSeqLength(self, nums: List[int]) -> int:
        n = len(nums)
        dp = defaultdict(lambda: 1)
        for i in range(n):
            for j in range(i):
                delta = nums[i] - nums[j]
                dp[(i, delta)] = max(dp[(j, delta)], dp[(j, delta)] + 1)
        return max(dp.values())
```

## 956. Tallest Billboard

### Solution 1:  meet in the middle + hash table

3^n possibility, 012, use base 3 to represent the states

Only need to represent the left and right sum for each possibility.  The initial sums is that they are both (0, 0).  Then of course there can be the possiblity that current rod is added to left sum, to right sum or neither, so 3 possiblities which gives the 3^n time. 

But it can be done in 3^(n/2) by using meet in the middle, so compute the left and right half of the rods separately with brute force. 

Then you are interested in storing the difference between the left and right sum and storing the maximum value for the left sum.  

Because if you have a diff, then you know the -diff in the other half will make them work together. 
cause 5 - 3 = 2, and 4 - 6 = -2, so together theses work, cause both each to 9.  think about it like if you know the left side is this relative to the right side, then you know for the other half you are looking for the opposite relationship so that it cancels out.  and the total difference is 0.  2 - 2 = 0

proof for why looking for -diff in the second half. 

left1 - right1 = diff
left2 - right2 = -diff
(left1 + left2) - (right1 + right2) = diff - diff = 0
implies that left1 + left2 = right1 + right2 

```py
class Solution:
    def tallestBillboard(self, rods: List[int]) -> int:
        n = len(rods)
        def solve(start, end):
            sums = set([(0, 0)]) # sum for s1, s2
            for i in range(start, end):
                nsums = set()
                for left, right in sums:
                    # add to left sum
                    nsums.add((left + rods[i], right))
                    # add to right sum
                    nsums.add((left, right + rods[i]))
                sums.update(nsums)
            states = defaultdict(lambda: -math.inf)
            for left, right in sums:
                delta = left - right
                states[delta] = max(states[delta], left)
            return states
        left_diffs = solve(0, n//2)
        right_diffs = solve(n//2, n)
        res = 0
        for diff in left_diffs:
            res = max(res, left_diffs[diff] + right_diffs[-diff])
        return res
```

### Solution 2:  dynamic programming 

this is the recurrence relation
dp[diff] = max(dp[diff], taller)
return dp[0]
for each rod, need to consider 3 options for each of the previous dp states. 
1. add rod to taller support
1. add rod to shorter support
1. add rod to neither support

```py
class Solution:
    def tallestBillboard(self, rods: List[int]) -> int:
        n = len(rods)
        # diff: taller
        # diff = taller - shorter
        # shorter = taller - diff
        dp = defaultdict(lambda: -math.inf)
        dp[0] = 0
        for rod in rods:
            # skipping adding the rod
            ndp = dp.copy()
            for diff, taller in dp.items():
                shorter = taller - diff
                # add the rod to the taller support
                ndp[diff + rod] = max(ndp[diff + rod], taller + rod)
                # add the rod to the shorter support
                ndiff = abs(taller - shorter - rod)
                ntaller = max(taller, shorter + rod)
                ndp[ndiff] = max(ndp[ndiff], ntaller)
            dp = ndp
        return dp[0]
```

## 2106. Maximum Fruits Harvested After at Most K Steps

### Solution 1: Sliding window algorithm with prefix and suffix sums

move k steps to the left and then k steps to the right.

For each instance, as you reduce number of steps moved in direction by 1, that allows you to move 2 more steps in the opposite direction. 

This will cover all the possible ranges.

```py
class Solution:
    def maxTotalFruits(self, fruits: List[List[int]], startPos: int, k: int) -> int:
        n = len(fruits)
        mx = max(fruits[-1][0], startPos)
        quantity = [0] * (mx + 1)
        for pos, amt in fruits:
            quantity[pos] = amt
        res = quantity[startPos]
        # move k steps to the left
        left_sum = right_sum = 0
        for pos in range(max(0, startPos - k), startPos):
            left_sum += quantity[pos]
        # moving less steps to left and some to the right with remaining steps
        right_pos = startPos - k
        for left_pos in range(startPos - k, startPos):
            res = max(res, left_sum + right_sum + quantity[startPos])
            if left_pos >= 0:
                left_sum -= quantity[left_pos]
            for _ in range(2):
                right_pos += 1
                if startPos < right_pos <= mx:
                    right_sum += quantity[right_pos]
        # move k steps to the right
        left_sum = right_sum = 0
        for pos in range(startPos + 1, min(startPos + k, mx) + 1):
            right_sum += quantity[pos]
        left_pos = startPos + k
        for right_pos in range(startPos + k, startPos, -1):
            res = max(res, left_sum + right_sum + quantity[startPos])
            if right_pos <= mx:
                right_sum -= quantity[right_pos]
            for _ in range(2):
                left_pos -= 1
                if 0 <= left_pos < startPos:
                    left_sum += quantity[left_pos]
        return res
```

## 373. Find K Pairs with Smallest Sums

### Solution 1:  hash set + min heap

```py
class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        n1, n2 = len(nums1), len(nums2)
        pointers = [0] * n1 # pointer to index in nums2
        heapify(min_heap := [(nums1[i] + nums2[0], i) for i in range(n1)])
        res = []
        for _ in range(k):
            if not min_heap: break
            _, p1 = heappop(min_heap)
            pointers[p1] += 1
            p2 = pointers[p1]
            res.append([nums1[p1], nums2[p2 - 1]])
            if p2 < n2:
                heappush(min_heap, (nums1[p1] + nums2[p2], p1))
        return res
```

## 1514. Path with Maximum Probability

### Solution 1:  math + logarithms + dijkstra + min heap + log sum

Use the property of logarithms
0 <= p <= 1

the best path will be multiplication of probabilities so such as p1*p2*p3*...*pn
log(p1*p2*...*pn) = log(p1) + log(p2) + ... + log(pn),  but you want to maximize probability so you want to maxmize the sum of logarithms of probabilities. 

![image](images/positive_log_probability.png)

As p increases, so does log(p) for given range of p values.  So you can use a max heap and get the one with the largest probability.

![images](images/negative_log_probability.png)

as p increases, -log(p) decreases for given range of p values.  So you can use a min heap cause the minimal sum of negative log of the probabilities corresponds to the maximum sum of positive log of the probabilities and the maximum of the product of probabilities

```py
class Solution:
    def maxProbability(self, n: int, edges: List[List[int]], succProb: List[float], start: int, end: int) -> float:
        adj_list = [[] for _ in range(n)]
        logarize = lambda x: -math.log(x)
        unlogarize = lambda x: math.exp(-x)
        for (u, v), p in zip(edges, succProb):
            adj_list[u].append((v, logarize(p)))
            adj_list[v].append((u, logarize(p)))
        min_heap = [(0, start)]
        dist = [math.inf] * n
        dist[start] = 0
        while min_heap:
            cost, node = heappop(min_heap)
            if dist[node] < cost: continue
            for nei, wei in adj_list[node]:
                ncost = cost + wei
                if ncost < dist[nei]:
                    dist[nei] = ncost
                    heappush(min_heap, (ncost, nei))
        return unlogarize(dist[end]) if dist[end] != math.inf else 0
```

## 250. Count Univalue Subtrees

### Solution 1:  dfs + postorder + binary tree

```py
class Solution:
    def countUnivalSubtrees(self, root: Optional[TreeNode]) -> int:
        res = 0
        def dfs(node):
            nonlocal res
            if not node: return True
            if node.left:
                is_univalued &= (node.val == node.left.val)
                is_univalued &= dfs(node.left)
            if node.right:
                is_univalued &= (node.val == node.right.val)
                is_univalued &= dfs(node.right)
            res += is_univalued
            return is_univalued
        dfs(root)
        return res
```

## 1970. Last Day Where You Can Still Cross

### Solution 1:  binary search for day + bfs over remaining land cells at that day

[t,t,t,t,f,f,f]
at one day you will no longer be able to reach bottom row from top row, and it will be false forever, so can binary search for the last day can reach bottom row.

```py
class Solution:
    def latestDayToCross(self, row: int, col: int, cells: List[List[int]]) -> int:
        cells = [(r - 1, c - 1) for r, c in cells]
        neighborhood = lambda r, c: [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        in_bounds = lambda r, c: 0 <= r < row and 0 <= c < col
        land, water = 0, 1
        def bfs(target):
            grid = [[0] * col for _ in range(row)]
            for r, c in map(lambda i: cells[i], range(target)):
                grid[r][c] = water # which ones will be covered with water by target
            queue = deque()
            for c in range(col):
                if grid[0][c] == land:
                    queue.append((0, c))
                    grid[0][c] = water
            while queue:
                r, c = queue.popleft()
                if r == row - 1: return True # reached bottom row
                for nr, nc in neighborhood(r, c):
                    if not in_bounds(nr, nc) or grid[nr][nc] != land: continue
                    grid[nr][nc] = water
                    queue.append((nr, nc))
            return False
        left, right = 0, row * col
        while left < right:
            mid = (left + right + 1) >> 1
            if bfs(mid):
                left = mid
            else:
                right = mid - 1
        return left
```

### Solution 2:  min heap + dijkstra + undirected graph + time nodes

The update function is interesting, it is not you are adding the sume of the distances, but rather taking the min distance so far.  

negating everything allows to take min heap

so what minimizes the negatives should maximize if they were positives

```py
class Solution:
    def latestDayToCross(self, row: int, col: int, cells: List[List[int]]) -> int:
        cells = [(r - 1, c - 1) for r, c in cells]
        neighborhood = lambda r, c: [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        in_bounds = lambda r, c: 0 <= r < row and 0 <= c < col
        land, water = 0, 1
        grid = [[0] * col for _ in range(row)]
        for i, (r, c) in enumerate(cells, start = 1):
            # negate so that we can take the min heap and apply dijkstra
            grid[r][c] = -i
        min_heap = []
        dist = [[math.inf] * col for _ in range(row)]
        for c in range(col):
            heappush(min_heap, (grid[0][c], 0, c))
            dist[0][c] = grid[0][c]
        while min_heap:
            cost, r, c = heappop(min_heap)
            if r == row - 1: return abs(cost + 1)
            for nr, nc in neighborhood(r, c):
                if not in_bounds(nr, nc): continue
                ncost = max(cost, grid[nr][nc])
                if ncost < dist[nr][nc]:
                    dist[nr][nc] = ncost
                    heappush(min_heap, (ncost, nr, nc))
        return 0
```

## Solution 3:  union find on water

find connected component from left side to right side with water because that will block the path 

```py

```

## 465. Optimal Account Balancing

### Solution 1:  backtrack + recursion + dfs + pruning of dfs tree

```py
class Solution:
    def minTransfers(self, transactions: List[List[int]]) -> int:
        balance = Counter()
        for u, v, amt in transactions:
            balance[u] -= amt
            balance[v] += amt
        balance = list(balance.values())
        n = len(balance)
        def backtrack(i):
            while i < n and balance[i] == 0: i += 1
            res = math.inf
            if i == n: return 0
            for j in range(i + 1, n):
                if balance[i] * balance[j] < 0:
                    balance[j] += balance[i]
                    res = min(res, 1 + backtrack(i + 1))
                    balance[j] -= balance[i]
            return res
        return backtrack(0)
```

## 1601. Maximum Number of Achievable Transfer Requests

### Solution 1:  backtrack + recursion + pruning

```py
class Solution:
    def maximumRequests(self, n: int, requests: List[List[int]]) -> int:
        m = len(requests)
        res = 0
        for mask in range(1, 1 << m):
            degrees = [0] * n
            request_count = mask.bit_count()
            if request_count <= res: continue
            for i in range(m):
                if (mask >> i) & 1:
                    u, v = requests[i]
                    degrees[u] -= 1
                    degrees[v] += 1
            if not any(x != 0 for x in degrees): res = max(res, request_count)
        return res
```

### Solution 2: network flow 

```py

```

## 859. Buddy Strings

### Solution 1:  set + sum + counter

```py
class Solution:
    def buddyStrings(self, s: str, goal: str) -> bool:
        if len(s) != len(goal): return False
        if s == goal and len(set(s)) < len(s): return True
        diff = sum(1 for i in range(len(s)) if s[i] != goal[i])
        return diff == 2 and Counter(s) == Counter(goal)
```

## 1493. Longest Subarray of 1's After Deleting One Element

### Solution 1:  prefix and suffix count of continuous ones

```py
class Solution:
    def longestSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        scount = [0] * (n + 1)
        for i in reversed(range(n)):
            scount[i] = scount[i + 1] + 1 if nums[i] else 0
        pcount = res = 0
        for i in range(n):
            res = max(res, pcount + scount[i + 1])
            pcount = pcount + 1 if nums[i] else 0
        return res
```

### Solution 2: sliding window + maintain one zero + one element deleted in sliding window

```py
class Solution:
    def longestSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        res = zeros = left = 0
        for right in range(n):
            zeros += not nums[right]
            while zeros > 1:
                zeros -= not nums[left]
                left += 1
            res = max(res, right - left)
        return res
```

## 137. Single Number II

### Solution 1:  addition under modulo 3 + bitwise operators + bit manipulation

![images](images/single_number_2.png)

Can induce addition under modulo 3 (a + b) % 3 by using addition under modulo 2 and some logic on each bit. 
Then need three variables to represent the bits that are currently set to 0, 1, 2.  

Logic is that if the bit is not set in twice then it must be set in the zero, so when it encounters it it will be now set to 1. 

```py
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        once = twice = 0
        for num in nums:
            once = (once ^ num & ~twice)
            twice = (twice ^ num & ~once)
        return once
```

## 2024. Maximize the Confusion of an Exam

### Solution 1:  sliding window

Move the left pointer whenever the current window is invalid because both T and F are greater than k so no way to replace k of them.  If one of them is less than or equal to k, then you are capable of flipping that answer key.

```py
class Solution:
    def maxConsecutiveAnswers(self, answerKey: str, k: int) -> int:
        n = len(answerKey)
        left = res = 0
        counts = Counter({'T': 0, 'F': 0})
        for right in range(n):
            counts[answerKey[right]] += 1
            while min(counts.values()) > k:
                counts[answerKey[left]] -= 1
                left += 1
            res = max(res, right - left + 1)
        return res
```

## 340. Longest Substring with At Most K Distinct Characters

### Solution 1:  sliding window + counter

```py
class Solution:
    def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
        n = len(s)
        counts = Counter()
        left = res = count_distinct = 0
        for right in range(n):
            counts[s[right]] += 1
            if counts[s[right]] == 1:
                count_distinct += 1
            while count_distinct > k:
                counts[s[left]] -= 1
                if counts[s[left]] == 0: 
                    count_distinct -= 1
                left += 1
            res = max(res, right - left + 1)
        return res
```

## 111. Minimum Depth of Binary Tree

### Solution 1:  bfs + queue

```py
class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if not root: return 0
        queue = deque([root])
        depth = 1
        while queue:
            for _ in range(len(queue)):
                node = queue.popleft()
                if not node.left and not node.right: return depth
                queue.extend(filter(None, (node.left, node.right)))
            depth += 1
        return -1
```

## 802. Find Eventual Safe States

### Solution 1: directed graph + topological sort + reverse adjacency list + outdegrees

If it has outdegree of a node is equal to 0 that means that it is a safe node because it is either a terminal node or it was only connected to other safe nodes.

```py
class Solution:
    def eventualSafeNodes(self, adj_list: List[List[int]]) -> List[int]:
        n = len(adj_list)
        rev_adj_list = [[] for _ in range(n)]
        outdegrees = [0] * n
        queue = deque()
        for i in range(n):
            outdegrees[i] = len(adj_list[i])
            if outdegrees[i] == 0: queue.append(i)
            for j in adj_list[i]:
                rev_adj_list[j].append(i)
        res = []
        while queue:
            node = queue.popleft()
            res.append(node)
            for nei in rev_adj_list[node]:
                outdegrees[nei] -= 1
                if outdegrees[nei] == 0: queue.append(nei)
        return sorted(res)
```

## 1218. Longest Arithmetic Subsequence of Given Difference

### Solution 1:  dynamic programming + counter

dp[i] = max(dp[i], dp[i - k] + 1), where k is the difference.  So it is either part of previous 

```py
class Solution:
    def longestSubsequence(self, arr: List[int], difference: int) -> int:
        dp = Counter()
        for num in arr:
            dp[num] = max(dp[num], dp[num - difference] + 1)
        return max(dp.values())
```

## 1751. Maximum Number of Events That Can Be Attended II

### Solution 1:  recursive dynamic programming + sort + bisect + binary search

```py
class Solution:
    def maxValue(self, events: List[List[int]], k: int) -> int:
        events = list(map(tuple, events))
        n = len(events)
        events.sort()
        @cache
        def dp(i, j):
            if j == k or i == n: return 0
            # skip event
            res = dp(i + 1, j)
            # take the event
            _, e, v = events[i]
            idx = bisect.bisect_right(events, (e, math.inf, math.inf))
            res = max(res, dp(idx, j + 1) + v)
            return res
        return dp(0, 0)
```

## 1644. Lowest Common Ancestor of a Binary Tree II

### Solution 1:  binary lifting + lowest common ancestor + binary tree

Convert binary tree into into adjacency list representation and as an undirected graph to run in binary lift algorithm

```py
class BinaryLift:
    """
    This binary lift function works on any undirected graph that is composed of
    an adjacency list defined by graph
    """
    def __init__(self, node_count: int, graph: List[List[int]]):
        self.size = node_count
        self.graph = graph # pass in an adjacency list to represent the graph
        self.depth = [0]*node_count
        self.parents = [-1]*node_count
        self.visited = [False]*node_count
        # ITERATE THROUGH EACH POSSIBLE TREE
        for node in range(node_count):
            if self.visited[node]: continue
            self.visited[node] = True
            self.get_parent_depth(node)
        self.maxAncestor = 18 # set it so that only up to 2^18th ancestor can exist for this example
        self.jump = [[-1]*self.maxAncestor for _ in range(self.size)]
        self.build_sparse_table()
        
    def build_sparse_table(self) -> None:
        """
        builds the jump sparse arrays for computing the 2^jth ancestor of ith node in any given query
        """
        for j in range(self.maxAncestor):
            for i in range(self.size):
                if j == 0:
                    self.jump[i][j] = self.parents[i]
                elif self.jump[i][j-1] != -1:
                    prev_ancestor = self.jump[i][j-1]
                    self.jump[i][j] = self.jump[prev_ancestor][j-1]
                    
    def get_parent_depth(self, node: int, parent_node: int = -1, depth: int = 0) -> None:
        """
        Fills out the depth array for each node and the parent array for each node
        """
        self.parents[node] = parent_node
        self.depth[node] = depth
        for nei_node in self.graph[node]:
            if self.visited[nei_node]: continue
            self.visited[nei_node] = True
            self.get_parent_depth(nei_node, node, depth+1)

    def distance(self, p: int, q: int) -> int:
        """
        Computes the distance between two nodes
        """
        lca = self.find_lca(p, q)
        return self.depth[p] + self.depth[q] - 2*self.depth[lca]

    def find_lca(self, p: int, q: int) -> int:
        # ASSUME NODE P IS DEEPER THAN NODE Q   
        if self.depth[p] < self.depth[q]:
            p, q = q, p
        # PUT ON SAME DEPTH BY FINDING THE KTH ANCESTOR
        k = self.depth[p] - self.depth[q]
        p = self.kthAncestor(p, k)
        if p == q: return p
        for j in range(self.maxAncestor)[::-1]:
            if self.jump[p][j] != self.jump[q][j]:
                p, q = self.jump[p][j], self.jump[q][j] # jump to 2^jth ancestor nodes
        return self.jump[p][0]
    
    def kthAncestor(self, node: int, k: int) -> int:
        while node != -1 and k>0:
            i = int(math.log2(k))
            node = self.jump[node][i]
            k-=(1<<i)
        return node

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        node_index = {root: 0}
        index_node = {0: root}
        adj_list = defaultdict(list)
        queue = deque([0])
        while queue:
            index = queue.popleft()
            node = index_node[index]
            if node.left:
                u, v = index, len(node_index)
                node_index[node.left] = v
                index_node[v] = node.left
                adj_list[u].append(v)
                adj_list[v].append(u)
                queue.append(v)
            if node.right:
                u, v = index, len(node_index)
                node_index[node.right] = v
                index_node[v] = node.right
                adj_list[u].append(v)
                adj_list[v].append(u)
                queue.append(v)
        if p not in node_index or q not in node_index: return None
        p_index, q_index = node_index[p], node_index[q]
        lca_algorithm = BinaryLift(len(node_index), adj_list)
        lca_node = index_node[lca_algorithm.find_lca(p_index, q_index)]
        return lca_node
```

## 1125. Smallest Sufficient Team

### Solution 1:  dynamic programming + bitmask + bfs + queue + shortest path in unweighted undirected graph

```py
class Solution:
    def smallestSufficientTeam(self, req_skills: List[str], people: List[List[str]]) -> List[int]:
        n, m = len(req_skills), len(people)
        end_mask = (1 << n) - 1
        dp = [0] * (1 << n)
        dp[0] = 0
        skill_value = {skill: 1 << i for i, skill in enumerate(req_skills)}
        pmasks = [reduce(operator.xor, map(lambda s: skill_value[s], skills), 0) for skills in people]
        queue = deque([0])
        while queue:
            mask = queue.popleft()
            if mask == end_mask: break
            for i, pmask in enumerate(pmasks):
                if pmask == 0: continue
                nmask = mask | pmask
                if dp[nmask]: continue
                dp[nmask] = dp[mask] | (1 << i)
                queue.append(nmask)
        res = [i for i in range(m) if (dp[mask] >> i) & 1]
        return res
```

### Solution 2:  bitmask + memoization + bfs + queue + backtrack to reconstruct shortest path

```py
class Solution:
    def smallestSufficientTeam(self, req_skills: List[str], people: List[List[str]]) -> List[int]:
        n = len(req_skills)
        end_mask = (1 << n) - 1
        vis = [0] * (1 << n)
        vis[0] = 1
        parent_arr = {(-1, 0): None}
        skill_value = {skill: 1 << i for i, skill in enumerate(req_skills)}
        pmasks = [reduce(operator.xor, map(lambda s: skill_value[s], skills), 0) for skills in people]
        queue = deque([(-1, 0)])
        while queue:
            person, mask = queue.popleft()
            if mask == end_mask: break
            for i, pmask in enumerate(pmasks):
                nmask = mask | pmask
                if vis[nmask]: continue
                vis[nmask] = 1
                parent_arr[i, nmask] = (person, mask)
                queue.append((i, nmask))
        res = []
        while mask != 0:
            res.append(person)
            person, mask = parent_arr[(person, mask)]
        return res
```

## Solution 3: BFS to find the shortest path or smallest team with bitmask for skills and bitset for people in team bitset for people in team

```cpp
using People = bitset<60>;
vector<int> smallestSufficientTeam(vector<string>& req_skills, vector<vector<string>>& people) {
    int n = req_skills.size(), m = people.size(), endMask = (1<<n)-1;
    unordered_map<string, int> skillMap;
    for (int i = 0;i<n;i++) {
        skillMap[req_skills[i]]=i;
    }
    int NEUTRAL  = m+1;
    vector<int> dp(1<<n,NEUTRAL);
    dp[0]=0;
    for (int mask = 0;mask<endMask;mask++) {
        if (dp[mask]==NEUTRAL) continue;
        for (int i = 0;i<m;i++) {  
            int nmask = mask;
            for (string& skill : people[i]) {
                int j = skillMap[skill];
                nmask|=(1<<j);
            }
            dp[nmask] = min(dp[nmask],dp[mask]+1);
        }
    }
    vector<int> result, vis(1<<n,0);
    queue<pair<int,People>> q;
    q.emplace(0,0);
    int mask;
    People p;
    while (!q.empty()) {
        tie(mask,p) = q.front();
        q.pop();
        if (mask==endMask) {
            break;
        }
        for (int i = 0;i<m;i++) {  
            int nmask = mask;
            for (string& skill : people[i]) {
                int j = skillMap[skill];
                nmask|=(1<<j);
            }
            if (nmask>mask && !vis[nmask]) {
                People np = p;
                np.set(i);
                q.emplace(nmask,np);
                vis[nmask]=1;
            }
        }
    }
    for (int i = 0;i<m;i++) {
        if (p.test(i)) {
            result.push_back(i);
        }
    }
    return result;
}
```

## 445. Add Two Numbers II

### Solution 1:  reverse linked lists + addition + linked lists + prev

```py
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        def reverse(lst):
            tail = lst
            prev = None 
            while tail:
                nxt = tail.next
                tail.next = prev
                prev = tail
                tail = nxt
            return prev
        l1, l2 = map(reverse, (l1, l2))
        dummy = ListNode()
        cur = dummy
        carry = 0
        while l1 or l2 or carry:
            if l1:
                carry += l1.val
                l1 = l1.next
            if l2:
                carry += l2.val
                l2 = l2.next
            value = carry % 10
            cur.next = ListNode(value)
            cur = cur.next
            carry //= 10
        res = reverse(dummy.next)
        return res
```

### Solution 2:  reverse linked lists + addition + linked lists + prev + reverse result during addition

```py
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        def reverse(lst):
            tail = lst
            prev = None 
            while tail:
                nxt = tail.next
                tail.next = prev
                prev = tail
                tail = nxt
            return prev
        l1, l2 = map(reverse, (l1, l2))
        dummy = ListNode()
        cur = dummy
        tail = prev = None
        carry = 0
        while l1 or l2 or carry:
            if l1:
                carry += l1.val
                l1 = l1.next
            if l2:
                carry += l2.val
                l2 = l2.next
            value = carry % 10
            nxt = ListNode(value)
            if tail:
                tail.next = prev
            prev = tail
            tail = nxt
            carry //= 10
        tail.next = prev
        return tail
```

## 648. Replace Words

### Solution 1:  trie 

```py
class Solution:
    def replaceWords(self, dictionary: List[str], sentence: str) -> str:
        TrieNode = lambda: defaultdict(TrieNode)
        root = TrieNode()
        # adding words into the trie data structure
        for word in dictionary:
            reduce(dict.__getitem__, word, root)['word'] = True
        result = []
        for word in sentence.split():
            cur = root
            n = len(word)
            for i in range(n):
                cur = cur[word[i]]
                if cur['word']:
                    result.append(word[:i + 1])
                    break
            if not cur['word']:
                result.append(word)
        return ' '.join(result)
```

## 435. Non-overlapping Intervals

### Solution 1:  greedy

You want to always greedily take the result that minimizes the end point, because that prevents the number of overlaps. 

So there are two cases as well for a pair of s, e
1. where s >= max_end and e > max_end.  In that case you want to update the max_end, because you will want to take that one. 
2. where s < max_end and e >= max_end.  In this case it is overlapping and you can add it to the result of overlapping intervals to remove. 


```py
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        res, max_end = 0, -math.inf
        for s, e in sorted(intervals, key = lambda pair: pair[1]):
            if s >= max_end:
                max_end = e
            else:
                res += 1
        return res
```

### Solution 2: dynamic programming + coordinate compression

dp[i] = maximum length of non overlapping interval that ends before the ith endpoint. 

```py
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        n = len(intervals)
        ends = defaultdict(list)
        end_points = set()
        index = {}
        index_point, point_index = {}, {}
        for s, e in intervals:
            end_points.update((s, e))
        for ep in sorted(end_points):
            index[ep] = len(index) + 1
        for s, e in intervals:
            ends[index[e]].append(index[s])
        m = len(index)
        dp = [0] * (m + 1)
        for i in range(1, m + 1):
            dp[i] = dp[i - 1]
            for j in sorted(ends[i]):
                dp[i] = max(dp[i], dp[j] + 1)
        return n - dp[-1]
```

## 735. Asteroid Collision

### Solution 1:  stack 

```py
class Solution:
    def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        n = len(asteroids)
        stack = []
        for num in asteroids:
            alive = True
            while stack and stack[-1] > 0 and num < 0:
                prv = stack.pop()
                if prv >= abs(num):
                    alive = False
                    if prv > abs(num):
                        stack.append(prv)
                    break
            if alive:
                stack.append(num)
        return stack
```

## 673. Number of Longest Increasing Subsequence

### Solution 1:  dynamic programming + counter

dp[i][j] = count of number of ways to get to increasing subsequence with length i, and the last element is equal to value j. 

So basically for some i, you want to look at all current processed subsequences of length i - 1, and then add them to the number of ways to attain dp[i][j] if the previous subsequence last element is less than j. 

```py
class Solution:
    def findNumberOfLIS(self, nums: List[int]) -> int:
        arr = []
        dp = [Counter({-math.inf: 1})]
        for num in nums:
            len_subsequence = bisect.bisect_left(arr, num)
            if len_subsequence == len(arr):
                arr.append(num)
                dp.append(Counter())
            else:
                arr[len_subsequence] = num
            for last_element, cnt in dp[len_subsequence].items():
                if last_element < num:
                    dp[len_subsequence + 1][num] += cnt
        return sum(dp[-1].values())
```

## 688. Knight Probability in Chessboard

### Solution 1:  dynamic programming + probability + O(kn^2)

The probability will be added because it is or logic, that is take this sequence or this sequence to end up at a specific cell in chess board.

```py
class Solution:
    def knightProbability(self, n: int, k: int, row: int, column: int) -> float:
        board = [[0.0] * n for _ in range(n)]
        board[row][column] = 1.0
        in_bounds = lambda r, c: 0 <= r < n and 0 <= c < n
        manhattan = lambda r, c: abs(r) + abs(c)
        neighborhood = lambda r, c: [(r + dr, c + dc) for dr, dc in product(range(-2, 3), repeat = 2) if manhattan(dr, dc) == 3]
        for _ in range(k):
            nboard = [[0.0] * n for _ in range(n)]
            for r, c in product(range(n), repeat = 2):
                if board[r][c] == 0.0: continue
                for nr, nc in neighborhood(r, c):
                    if not in_bounds(nr, nc): continue
                    nboard[nr][nc] += board[r][c] / 8
            board = nboard
        return sum(map(sum, board))
```

## 852. Peak Index in a Mountain Array

### Solution 1:  bisect_right + binary search

FFFTTTT, same as 000111, so find the first 1 is the solution

Cause you have it being false when it is increasing and when decreasing it becomes true. 


```py
class Solution:
    def peakIndexInMountainArray(self, arr: List[int]) -> int:
        return bisect.bisect_right(range(1, len(arr)), False, key = lambda i: arr[i] < arr[i - 1])
```

## 50. Pow(x, n)

### Solution 1:  binary exponentation + O(logn) + bit manipulation

```py
class Solution:
    def myPow(self, x: float, n: int) -> float:
        def exponentiation(b, p):
            res = 1
            while p > 0:
                if p & 1:
                    res *= b
                b *= b
                p >>= 1
            return res
        return exponentiation(x, n) if n >= 0 else 1 / exponentiation(x, abs(n))
```

## 1870. Minimum Speed to Arrive on Time

### Solution 1:  binary search + bisect_right

```py
class Solution:
    def minSpeedOnTime(self, dist: List[int], hour: float) -> int:
        def possible(spd):
            cur = 0
            for t in map(lambda dis: dis / spd, dist):
                cur = math.ceil(cur)
                cur += t
                if cur > hour: return False
            return True
        INF = int(1e7) + 5
        i = bisect.bisect_right(range(1, INF), False, key = lambda x: possible(x)) + 1
        return i if i < INF else -1
```

## 439. Ternary Expression Parser

### Solution 1:  reverse polish notation + stack + backwards iteration

postfix notation means 1 2 +, 
you see the operands first and then operator

```py
class Solution:
    def parseTernary(self, expression: str) -> str:
        # reverse polish notation if you iterate from right to left that is a b + , that is you see the operands first and then the operator, in this case the operands are b:c and the operator is ?
        stack = []
        i = len(expression) - 1
        while i >= 0:
            if expression[i] in 'TF0123456789':
                stack.append(expression[i])
            elif expression[i] == '?':
                on_true, on_false = stack.pop(), stack.pop()
                if expression[i - 1] == 'T':
                    stack.append(on_true)
                else:
                    stack.append(on_false)
                i -= 1
            i -= 1
        return stack[0]
```

## 2141. Maximum Running Time of N Computers

### Solution 1:  greedy + binary search + bisect_right

FFFFFTTTT
return first T but actually want last F

```py
class Solution:
    def maxRunTime(self, n: int, batteries: List[int]) -> int:
        batteries.sort(reverse = True)
        hours = sum(batteries) - sum(batteries[:n])
        def possible(target):
            rem = hours
            for i in range(n):
                rem -= max(0, target - batteries[i])
                if rem < 0: return True
            return False
        right = int(1e14) + 1
        i = bisect.bisect_right(range(right), False, key = lambda x: possible(x)) - 1
        return i
```

### Solution 2:  binary search

TTTTTFFFF
return last T

```py
class Solution:
    def maxRunTime(self, n: int, batteries: List[int]) -> int:
        batteries.sort(reverse = True)
        hours = sum(batteries) - sum(batteries[:n])
        def possible(target):
            rem = hours
            for i in range(n):
                rem -= max(0, target - batteries[i])
                if rem < 0: return False
            return True
        left, right = 0, int(1e14) + 1
        while left < right:
            mid = (left + right + 1) >> 1
            if possible(mid):
                left = mid
            else:
                right = mid - 1
        return left
```

### Solution 3:  sort + greedy

```py
class Solution:
    def maxRunTime(self, n: int, batteries: List[int]) -> int:
        batteries.sort(reverse = True)
        hours = sum(batteries) - sum(batteries[:n])
        for i in range(n - 2, -1, -1):
            delta = batteries[i] - batteries[i + 1]
            left_comps = n - i - 1
            if delta * left_comps <= hours:
                hours -= delta * left_comps
            else:
                return batteries[i + 1] + hours // left_comps
        return batteries[0] + hours // n
```

```py
class Solution:
    def maxRunTime(self, n: int, batteries: List[int]) -> int:
        batteries.sort(reverse = True)
        hours = sum(batteries) - sum(batteries[:n])
        i = n - 2
        while i >= 0:
            delta = batteries[i] - batteries[i + 1]
            comps = n - i - 1
            if delta * comps > hours: break
            hours -= delta * comps
            i -= 1
        return batteries[i + 1] + hours // (n - i - 1)
```

## 486. Predict the Winner

### Solution 1:  recursive dynamic programming + O(2^n)

Since n <= 20 this works, 2^20 is around 10^6

```py
class Solution:
    def PredictTheWinner(self, nums: List[int]) -> bool:
        n = len(nums)
        @cache
        def dfs(left, right, score1, score2, player):
            if left == right: return score1 >= score2 if player else score1 < score2
            delta_left1, delta_left2 = nums[left] if player else 0, 0 if player else nums[left]
            delta_right1, delta_right2 = nums[right - 1] if player else 0, 0 if player else nums[right - 1]
            return (
                not dfs(left + 1, right, score1 + delta_left1, score2 + delta_left2, player ^ 1)
                or
                not dfs(left, right - 1, score1 + delta_right1, score2 + delta_right2, player ^ 1)
            )
        return dfs(0, n, 0, 0, 1)
```

### Solution 2:  reduce the size of each state

```py
class Solution:
    def PredictTheWinner(self, nums: List[int]) -> bool:
        n = len(nums)
        @cache
        def dfs(left, right, score):
            player = 0 if (right - left) % 2 == n % 2 else 1
            if left == right: return score >= 0 if not player else score < 0
            return (
                not dfs(left + 1, right, score + (-1)**player * nums[left])
                or
                not dfs(left, right - 1, score + (-1)**player * nums[right - 1])
            )
        return dfs(0, n, 0)
```

O(n^2)

```py
class Solution:
    def PredictTheWinner(self, nums: List[int]) -> bool:
        n = len(nums)
        @cache
        def dfs(left, right):
            if left == right: return nums[left]
            score_left, score_right = nums[left] - dfs(left + 1, right), nums[right] - dfs(left, right - 1)
            return max(score_left, score_right)
        return dfs(0, n - 1) >= 0
```

```py
class Solution:
    def PredictTheWinner(self, nums: List[int]) -> bool:
        n = len(nums)
        dp = [[0] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = nums[i]
        for delta in range(1, n):
            for left in range(n - delta):
                right = left + delta
                dp[left][right] = max(nums[left] - dp[left + 1][right], nums[right] - dp[left][right - 1])
        return dp[0][-1] >= 0
```

## 1060. Missing Element in Sorted Array

### Solution 1:  greedy + binary search + nlogn

FFFTTTT return first T with using bisect_right

```py
class Solution:
    def missingElement(self, nums: List[int], k: int) -> int:
        nums.append(math.inf)
        n = len(nums)
        right = int(1e12)
        def possible(target):
            res = 0
            for i in range(1, n):
                if nums[i] >= target:
                    res += target - nums[i - 1] - (1 if nums[i] == target else 0)
                    break
                res += nums[i] - nums[i - 1] - 1
                if res >= k: return True
            return res >= k
        i = bisect.bisect_right(range(nums[0], right), 0, key = lambda x: possible(x)) + nums[0]
        return i
```

TTTFFF return first F

```py
class Solution:
    def missingElement(self, nums: List[int], k: int) -> int:
        nums.append(math.inf)
        n = len(nums)
        left, right = nums[0], int(1e12)
        def possible(target):
            res = 0
            for i in range(1, n):
                if nums[i] >= target:
                    res += target - nums[i - 1] - (1 if nums[i] == target else 0)
                    break
                res += nums[i] - nums[i - 1] - 1
            return res < k
        while left < right:
            mid = (left + right) >> 1
            if possible(mid):
                left = mid + 1
            else:
                right = mid
        return left
```

## 808. Soup Servings

### Solution 1:  memoization + dynamic programming

```py
class Solution:
    def soupServings(self, n: int) -> float:
        if n >= 5_000: return 1.0
        states = Counter({(n, n): 1.0})
        res = 0
        cnt = 0
        while states:
            nstates = Counter()
            for (a, b), pr in states.items():
                cnt += 1
                for da, db in [(100, 0), (75, 25), (50, 50), (25, 75)]:
                    na, nb = a - da, b - db
                    npr = pr * 0.250
                    if na <= 0 or nb <= 0:
                        if na <= 0 and nb > 0:
                            res += npr
                        if na <= 0 and nb <= 0:
                            res += npr/2
                    else:
                        nstates[(na, nb)] += npr
            states = nstates
        return res
```

### Solution 2:  dynamic programming + probability + law of large numbers + expectation value

you can divide everything by ceil(n / 25) the reason is because each serving is in group of 25, and if it is greater than 0 it needs to be 1, cause you still need one serving

then the problem is about these operations
1. 4 serving a, 0 serving b
1. 3 serving a, 1 serving b
1. 2 serving a, 2 serving b
1. 1 serving a, 3 serving b

without doing the math you can assume that at some large number of servings need for a and b. that it will reach 100% probability that a will finish first because it should have a large expected value, and by law of large numbers it will close in on the expected value.  

So if you just add a factor to check when it becomes within the tolerance 10^-5 you can terminate it earlier,  it turns out it will be around 200 servings.  40,000 operations is fine. 

So what are the base cases and recurrence relation? 

The base case is that when there are 0 servings for a it should have probability = 1.0 and if it is 0 for a and b it is 0.5 and if it is 0 for b it is 0.0.  

So dp[i][j], where i is remaining servings for a, and j is remaining serving for b
dp[0][j] = 1, dp[0][0] = 1/2, dp[i][0] = 0
The recurrence for the rest is 
dp[i][j] = 1 / 4 * (dp[i - 4][j] + dp[i - 3][j - 1] + dp[i - 2][j - 2] + dp[i - 1][j - 3])
it is 1/4th the probability for each possible path which is using or logic, because it use operation 1 or operation 2 etc.

```py

```

## 664. Strange Printer

### Solution 1:  dynamic programming + interval + greedy

increase the length of the ranges that you compute the number of paint turns needed, this means you know previous subproblem is solved since a larger length range is just composed of smaller ranges. 

For each length you take all the left starting points, and you know the range from left to right, so it is a substring of the string.  So for this interval, you want to find the first index at which the characters don't match the rightmost character.  

case 1: all characters match
-------, this is a base case, we return 0 for this
case 2: prefix and suffix match
---xxxxx--
   ^
In this case you want to set the mid pointer to the location where they first beging to not match with the last character.  Then you will consider every possible combination of subinterval 
---xxxxx--
   ^^    ^
   ^ ^   ^
   ^  ^  ^
   ^   ^ ^
   ^    ^^
You already have solved these subproblems, so it will work.  

return the result for the range extending the entire string, and add 1, because you need to set base case to 0, but it really is 1 to paint the base color.  If you made the base case equal to 1 though it would not work. 

I label this as greedy because you greedily take the same character from prefix that match with last character until they are different. 

```py
class Solution:
    def strangePrinter(self, s: str) -> int:
        n = len(s)
        dp = [[n] * n for _ in range(n)] # (left, right)
        for len_ in range(1, n + 1): 
            for left in range(n - len_ + 1):
                right = left + len_ - 1
                mid = None
                for i in range(left, right):
                    if s[i] != s[right] and mid is None:
                        mid = i
                    if mid is not None:
                        # stuck this doesn't work at all actually
                        dp[left][right] = min(dp[left][right], 1 + dp[mid][i] + dp[i + 1][right])
                if mid is None:
                    dp[left][right] = 0 # base case
        return dp[0][-1] + 1
```

## 712. Minimum ASCII Delete Sum for Two Strings

### Solution 1:  dynamic programming + O(n^2)

dp[i][j] = the minimum ascii deletion to get to the s1[...i] and s2[...j] substrings
The result is the full s1 and s2 string. 

The base cases are that you need to know if you delete

```py
class Solution:
    def minimumDeleteSum(self, s1: str, s2: str) -> int:
        n1, n2 = len(s1), len(s2)
        dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]
        for i in range(n1):
            dp[i + 1][0] = dp[i][0] + ord(s1[i])
        for j in range(n2):
            dp[0][j + 1] = dp[0][j] + ord(s2[j])
        for i in range(n1):
            for j in range(n2):
                if s1[i] == s2[j]:
                    dp[i + 1][j + 1] = dp[i][j]
                else:
                    dp[i + 1][j + 1] = min(dp[i][j + 1] + ord(s1[i]), dp[i + 1][j] + ord (s2[j]))
        return dp[-1][-1]
```

## 77. Combinations

### Solution 1:  combinations

```py
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        return combinations(range(1, n + 1), k)
```

```py
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        res = []
        for mask in range(1, 1 << n):
            if mask.bit_count() != k: continue
            cur = [i + 1 for i in range(n) if (mask >> i) & 1]
            res.append(cur)
        return res
```

## 1683. Invalid Tweets

### Solution 1:  string + subset

```py
import pandas as pd

def invalid_tweets(tweets: pd.DataFrame) -> pd.DataFrame:
    tweets = tweets[tweets.content.str.len() > 15]
    return tweets[['tweet_id']]
```

## 1517. Find Users With Valid E-Mails

### Solution 1:  regex + str contains

```py
import pandas as pd

def valid_emails(users: pd.DataFrame) -> pd.DataFrame:
    return users[users.mail.str.contains('^[A-Za-z][A-Za-z0-9_.-]*@leetcode\.com$', regex = True)]
```

## 46. Permutations

### Solution 1:

```py
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        return permutations(nums)
```

```py
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        res, cur = [], []
        def backtrack(i, mask):
            if i == n:
                res.append(cur[:])
            for j in range(n):
                if (mask >> j) & 1: continue
                cur.append(nums[j])
                backtrack(i + 1, mask | (1 << j))
                cur.pop()
        backtrack(0, 0)
        return res
```

## 2082. The Number of Rich Customers

### Solution 1:  nunique + initialize dataframe

```py
import pandas as pd

def count_rich_customers(store: pd.DataFrame) -> pd.DataFrame:
    cnt = store[store.amount > 500].customer_id.nunique()
    return pd.DataFrame({'rich_count': [cnt]})
```

## 596. Classes More Than 5 Students

### Solution 1:  value_counts + reset_index + filter query + column subset dataframe

pandas.series.reset_index() creates a dataframe where the index becomes a column

This is useful when the index needs to be treated as a column, or when the index is meaningless and needs to be reset to the default before another operation.

When drop is False (the default), a DataFrame is returned. The newly created columns will come first in the DataFrame, followed by the original Series values. When drop is True, a Series is returned. In either case, if inplace=True, no value is returned.


```py
import pandas as pd

def find_classes(courses: pd.DataFrame) -> pd.DataFrame:
    # number of unique combinations with column class
    # so like students a, b, c are class = math, then it will be 3 for math
    courses = (
        courses
        .value_counts(subset = 'class')
        .resetindex()
    )
    return courses[courses['count'] >= 5][['class']]
```

## 1907. Count Salary Categories

### Solution 1:  value_countrs + reindexing and reset index with name + apply

```py
import pandas as pd

def count_salary_categories(accounts: pd.DataFrame) -> pd.DataFrame:
    accounts['category'] = accounts['income'].apply(lambda row: "Low Salary" if row < 20_000 else "Average Salary" if row <= 50_000 else "High Salary")
    accounts = (
        accounts
        .value_counts('category')
        .reindex(["Low Salary", "Average Salary", "High Salary"], fill_value = 0)
        .reset_index(name = 'accounts_count')
    )
    return accounts
```

### Solution 2:  Get sum of the boolean series + initialize pandas dataframe from dictionary

```py
import pandas as pd

def count_salary_categories(accounts: pd.DataFrame) -> pd.DataFrame:
    low_count = (accounts.income < 20_000).sum()
    avg_count = ((accounts.income >= 20_000) & (accounts.income <= 50_000)).sum()
    high_count = (accounts.income > 50_000).sum()
    return pd.DataFrame(
        {"category": ["Low Salary", "Average Salary", "High Salary"],
        "accounts_count": [low_count, avg_count, high_count]}
    )
```

## 33. Search in Rotated Sorted Array

### Solution 1:  binary search + if statements to separate the different cases

Break the problem down into three possible scenarios

1. $R <= L <= M$ such as 4, 7, 3 (L, M, R) 
1. $M <= R <= L$ such as 4, 0, 3
1. $L <= M <= R$ such as 0, 3, 5

```py
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        n = len(nums)
        left, right = 0, n - 1
        while left < right:
            mid = (left + right + 1) >> 1
            if nums[right] <= nums[left] <= nums[mid]:
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
        return left if nums[left] == target else -1
```