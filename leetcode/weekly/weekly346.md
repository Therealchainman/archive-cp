# Leetcode Weekly Contest 346

## 2696. Minimum String Length After Removing Substrings

### Solution 1:  string slicing

```py
class Solution:
    def minLength(self, s: str) -> int:
        a, b = "AB", "CD"
        while a in s or b in s:
            if a in s:
                i = s.index(a)
                s = s[:i] + s[i + 2:]
            elif b in s:
                i = s.index(b)
                s = s[:i] + s[i + 2:]
        return len(s)
```

## 2697. Lexicographically Smallest Palindrome

### Solution 1:  two pointers + reverse second half string

```py
class Solution:
    def makeSmallestPalindrome(self, s: str) -> str:
        n = len(s)
        left_s = s[:n//2]
        right_s = s[n//2 + (1 if n&1 else 0):][::-1]
        s_pal = []
        for i in range(n//2):
            if left_s[i] <= right_s[i]:
                s_pal.append(left_s[i])
            elif left_s[i] > right_s[i]:
                s_pal.append(right_s[i])
        if n & 1:
            s_pal.append(s[n//2])
        for i in range(n//2-1, -1, -1):
            s_pal.append(s_pal[i])
        return ''.join(s_pal)
```

## 2698. Find the Punishment Number of an Integer

### Solution 1:  bit mask

Try all partitions of the integer and check if it works or not.

```py
class Solution:
    def punishmentNumber(self, n: int) -> int:
        def is_valid(i):
            s = str(i * i)
            m = len(s)
            for mask in range(0, 1 << m):
                cur = total = 0
                for j in range(m):
                    if (mask >> j) & 1:
                        total += cur
                        cur = int(s[j])
                    else:
                        cur = cur * 10 + int(s[j])
                total += cur
                if total == i: 
                    return True
            return False
        return sum(i * i for i in range(1, n + 1) if is_valid(i))
```

## 2699. Modify Graph Edge Weights

### Solution 1:  dijkstra + reconstruct path

You have to be careful here because there is a reason this line is needed `edge_weights[edge] = max(target - rev_dists[v] - walked, 1)`.  It guarantees that the path picked as that with minimum distance from source to destination.  If you don't do this you could allow another path to become the path with minimum distance, and could lead to a minimum distance less than target.

This image shows the specific edge case where you need to give specific value at the -1 edge.  If you give edge weight of 4 it guarantees current path is still shortest path, if you were to give it something less than 4, there would be a different path that is shortest. if you give it greater than 4, just need to balance that fact later. 

![image](images/modified_edge_weights.PNG)

```py
class Solution:
    def modifiedGraphEdges(self, n: int, edges: List[List[int]], source: int, destination: int, target: int) -> List[List[int]]:
        adj_list = [[] for _ in range(n)]
        lower_bound, upper_bound = 1, 2*10**9
        for u, v, w in edges:
            adj_list[u].append((v, w))
            adj_list[v].append((u, w))
        def dijkstra(src, skip_mod):
            minheap = [(0, src)]
            dist = [math.inf] * n
            dist[src] = 0
            parent = {src: None}
            while minheap:
                cost, node = heapq.heappop(minheap)
                if cost > dist[node]: continue
                for nei, wei in adj_list[node]:
                    if wei == -1:
                        if skip_mod: continue
                        wei = lower_bound
                    ncost = cost + wei
                    if ncost < dist[nei]:
                        dist[nei] = ncost
                        heapq.heappush(minheap, (ncost, nei))
                        parent[nei] = node
            return dist, parent
        rev_dists, _ = dijkstra(destination, True)
        if rev_dists[source] < target: return []
        dists, parent = dijkstra(source, False)
        if dists[destination] > target: return []
        edge_weights = {(min(u, v), max(u, v)): w for u, v, w in edges}
        path = [destination]
        while path[-1] != source:
            path.append(parent[path[-1]])
        path = path[::-1]
        walked = 0
        for i in range(1, len(path)):
            u, v = path[i-1], path[i]
            edge = (min(u, v), max(u, v))
            if edge_weights[edge] == -1:
                edge_weights[edge] = max(target - rev_dists[v] - walked, 1)
            walked += edge_weights[edge]
        for u, v, w in edges:
            edge = (min(u, v), max(u, v))
            if edge_weights[edge] == -1:
                edge_weights[edge] = upper_bound
        return [[u, v, w] for (u, v), w in edge_weights.items()]
```


