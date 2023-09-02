# Leetcode Weekly Contest 357

## 2810. Faulty Keyboard

### Solution 1:  deque

```py
class Solution:
    def finalString(self, s: str) -> str:
        n = len(s)
        res = []
        for ch in s:
            if ch == 'i':
                res.reverse()
            else:
                res.append(ch)
        return ''.join(res)
```

### Solution 2:  flipping + deque + O(n) + forward facing and reverse facing

```py
class Solution:
    def finalString(self, s: str) -> str:
        queue = deque()
        forward = False
        for chars in s.split('i'):
            forward = not forward
            if forward:
                queue.extend(chars)
            else:
                queue.extendleft(chars)
        return ''.join(queue) if forward else ''.join(reversed(queue))
```

## 2811. Check if it is Possible to Split Array

### Solution 1:  any + greedy

Observe that you can split any subarray into a single element and the rest, so you just need to do that and have one part for when you get to 3 elements in array, so that a part of it is greater than or equal to m.  That way you can remove the one element.

```py
class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        return len(nums) <= 2 or any(nums[i] + nums[i - 1] >= m for i in range(1, len(nums)))
```

## 2812. Find the Safest Path in a Grid

### Solution 1:  bfs + deque + dijkstra + max heap

Set the grid integers to be equal to the minimum distance to a thief, can do this with a multisource bfs from each thief.  Use those grid integers with a max heap.  Can convert the problem to min heap as well.  

```py
class Solution:
    def maximumSafenessFactor(self, grid: List[List[int]]) -> int:
        n = len(grid)
        thief, empty = 1, 0
        queue = deque([(r, c, 0) for r, c in product(range(n), repeat = 2) if grid[r][c] == thief])
        grid = [[-1] * n for _ in range(n)]
        for r, c, _ in queue:
            grid[r][c] = 0
        neighborhood = lambda r, c: [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        in_bounds = lambda r, c: 0 <= r < n and 0 <= c < n
        while queue:
            r, c, lv = queue.popleft()
            for nr, nc in neighborhood(r, c):
                if not in_bounds(nr, nc) or grid[nr][nc] != -1: continue
                grid[nr][nc] = lv + 1
                queue.append((nr, nc, lv + 1))
        maxheap = [(-grid[0][0], 0, 0)]
        grid[0][0] = -1
        while maxheap:
            lv, r, c = heappop(maxheap)
            lv = abs(lv)
            if r == c == n - 1:
                return lv
            for nr, nc in neighborhood(r, c):
                if not in_bounds(nr, nc) or grid[nr][nc] == -1: continue
                grid[nr][nc] = min(lv, grid[nr][nc])
                heappush(maxheap, (-grid[nr][nc], nr, nc))
                grid[nr][nc] = -1
        return 0
```

### Solution 2:  bfs + binary search + bisect_left + dfs

```py
class Solution:
    def maximumSafenessFactor(self, grid: List[List[int]]) -> int:
        n = len(grid)
        thief, empty = 1, 0
        queue = deque()
        for r, c in product(range(n), repeat = 2):
            if grid[r][c] == thief:
                grid[r][c] = 0
                queue.append((r, c, 0))
            else:
                grid[r][c] = -1
        neighborhood = lambda r, c: [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        in_bounds = lambda r, c: 0 <= r < n and 0 <= c < n
        while queue:
            r, c, lv = queue.popleft()
            for nr, nc in neighborhood(r, c):
                if not in_bounds(nr, nc) or grid[nr][nc] != -1: continue
                grid[nr][nc] = lv + 1
                queue.append((nr, nc, lv + 1))
        def possible(target):
            stack = []
            vis = set()
            if grid[0][0] >= target:
                stack.append((0, 0))
                vis.add((0, 0))
            while stack:
                r, c = stack.pop()
                if r == c == n - 1: return False
                for nr, nc in neighborhood(r, c):
                    if not in_bounds(nr, nc) or grid[nr][nc] < target or (nr, nc) in vis: continue
                    stack.append((nr, nc))
                    vis.add((nr, nc))
            return True
        # FFFTTT, last F
        # false for when you can reach end with the safeness factor
        # true if you can't reach
        return bisect_right(range(2 * n + 1), False, key = lambda x: possible(x)) - 1
```

## 2813. Maximum Elegance of a K-Length Subsequence

### Solution 1:  greedy + sort + monotonic stack

```py
class Solution:
    def findMaximumElegance(self, items: List[List[int]], k: int) -> int:
        n = len(items)
        seen = set()
        res = cur = 0
        # extra will be sorted in non-increasing profits, so the lowest profit at the end of it
        # this will be useful for when you want to add a new category to the list, and you can do this
        # by replacing it. 
        extra = []
        for i, (profit, category) in enumerate(sorted(items, reverse = True)):
            if i < k:
                cur += profit
                if category in seen:
                    extra.append(profit)
            elif category not in seen:
                if not extra: break
                cur += profit - extra.pop()
            seen.add(category)
            res = max(res, cur + len(seen) * len(seen))
        return res
```

### Solution 2: min and max heaps + dictionary

```py
class Solution:
    def findMaximumElegance(self, items: List[List[int]], k: int) -> int:
        n = len(items)
        max_item = Counter()
        index = dict()
        vis = set()
        categories = [0] * (n + 1)
        for i, (profit, cat) in enumerate(items):
            if profit > max_item[cat]:
                max_item[cat] = profit
                index[cat] = i
        res = cur = 0
        min_heap = []
        num_cat = 0
        for cat, pr in sorted(max_item.items(), key = lambda pair: (-pair[1])):
            cur += pr
            vis.add(index[cat])
            categories[cat] = 1
            num_cat += 1
            k -= 1
            heappush(min_heap, (pr, cat))
            if k == 0: break
        res = cur + num_cat * num_cat
        # only add items to the max heap which are not visited
        remain_items = sorted([(profit, cat) for i, (profit, cat) in enumerate(items) if i not in vis], reverse = True)
        for profit, cat in remain_items:
            if k == 0:
                k += 1
                pr, ca = heappop(min_heap)
                cur -= pr
                categories[ca] -= 1
                if categories[ca] == 0:
                    num_cat -= 1
            k -= 1
            cur += profit
            categories[cat] += 1
            if categories[cat] == 1:
                num_cat += 1
            heappush(min_heap, (profit, cat))
            res = max(res, cur + num_cat * num_cat)
        return res
```