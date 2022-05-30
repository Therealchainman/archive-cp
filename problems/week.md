# Summary 

## 99. Recover Binary Search Tree

### Solution 1: dfs + recursion until no longer swap nodes

```py
class Solution:
    def recoverTree(self, root: Optional[TreeNode]) -> None:
        def swap_nodes(root, lo=TreeNode(-inf), hi=TreeNode(inf)):
            if not root: return False
            if root.val < lo.val:
                root.val, lo.val = lo.val, root.val
                return True
            if root.val > hi.val:
                root.val, hi.val = hi.val, root.val
                return True
            if swap_nodes(root.left, lo, root):
                return True
            if swap_nodes(root.right, root, hi):
                return True
        while swap_nodes(root):
            pass
```

### Solution 2: 

```py
class Solution:
    def recoverTree(self, root: TreeNode):
        def inorder(node):
            if node:
                yield from inorder(node.left)
                yield node.val
                yield from inorder(node.right)
        def swapped_nodes(arr):
            first = second = None
            for i in range(1,len(arr)):
                if arr[i] < arr[i-1]:
                    second = arr[i]
                    if not first:
                        first = arr[i-1]
                    
            return first, second
        def recover(node, count = 2):
            if node:
                if node.val == first or node.val == second:
                    node.val = first if node.val == second else second
                    count -= 1
                    if count == 0: return
                recover(node.left, count)
                recover(node.right, count)
        
        arr = list(inorder(root))
        first, second = swapped_nodes(arr)
        recover(root)
```

### Solution 3: Iterative inorder traversal with stack

```py
class Solution:
    def recoverTree(self, root: TreeNode):
        def inorder(node):
            stack = []
            while node or stack:
                while node:
                    stack.append(node)
                    node = node.left
                node = stack.pop()
                yield node
                node = node.right
            
        prev_node = first_node = second_node = None
        for node in inorder(root):
            if prev_node and prev_node.val > node.val:
                second_node = node
                if not first_node:
                    first_node = prev_node
            prev_node = node
        first_node.val, second_node.val = second_node.val, first_node.val
```

## 1586. Binary Search Tree Iterator II

### Solution 1: Flatten BST + recursive inorder traversal

```py
class BSTIterator:
    
    def inorder(self, root):
        if not root: return []
        return self.inorder(root.left) + [root.val] + self.inorder(root.right)

    def __init__(self, root: Optional[TreeNode]):
        self.arr = [0] + self.inorder(root)
        self.i = 0

    def hasNext(self) -> bool:
        return self.i < len(self.arr) - 1

    def next(self) -> int:
        self.i += 1
        return self.arr[self.i]

    def hasPrev(self) -> bool:
        return self.i > 1

    def prev(self) -> int:
        self.i -= 1
        return self.arr[self.i]
```

### Solution 2: iterative stack based inorder traversal

```py
class BSTIterator:

    def __init__(self, root: Optional[TreeNode]):
        self.node = root
        self.stack, self.arr = [], []
        self.pointer = -1

    def hasNext(self) -> bool:
        return self.stack or self.node or self.pointer + 1 < len(self.arr)

    def next(self) -> int:
        self.pointer += 1
        if self.pointer == len(self.arr):
            while self.node:
                self.stack.append(self.node)
                self.node = self.node.left
            self.node = self.stack.pop()
            self.arr.append(self.node.val)
            self.node = self.node.right
        return self.arr[self.pointer]

    def hasPrev(self) -> bool:
        return self.pointer > 0 

    def prev(self) -> int:
        self.pointer -= 1
        return self.arr[self.pointer]
```

## 285. Inorder Successor in BST

### Solution 1: Iterative stack based inorder traversal

```py
class Solution:
    def inorderSuccessor(self, root: TreeNode, p: TreeNode) -> Optional[TreeNode]:
        def inorder(node):
            stack = []
            while node or stack:
                while node:
                    stack.append(node)
                    node=node.left
                node = stack.pop()
                yield node
                node=node.right
        root_iter = inorder(root)
        while next(root_iter).val != p.val: pass
        try: 
            return next(root_iter)
        except:
            return None
```

## 2237. Count Positions on Street With Required Brightness

### Solution 1: counter + hash table for change in brightness

```py
class Solution:
    def meetRequirement(self, n: int, lights: List[List[int]], requirement: List[int]) -> int:
        change_brightness = [0]*n
        for pos, dist in lights:
            change_brightness[max(0,pos-dist)] += 1
            if pos+dist+1 < n:
                change_brightness[pos+dist+1] -= 1
        brightness = cnt = 0
        for req, delta in zip(requirement, change_brightness):
            brightness += delta
            cnt += (brightness >= req)
        return cnt
```

## 2238. Number of Times a Driver Was a Passenger

### Solution 1: Left Join with distinct driver_id + group by driver_id and aggregate for count of passenger_id

```sql
SELECT 
  d.driver_id, Count(r2.passenger_id) cnt
FROM 
  (SELECT DISTINCT r1.driver_id FROM rides r1) d
  LEFT JOIN rides r2 ON d.driver_id = r2.passenger_id
GROUP BY driver_id
```

## 173. Binary Search Tree Iterator

### Solution 1: iterative inorder BST traversal with stack

```py
class BSTIterator:

    def __init__(self, root: Optional[TreeNode]):
        self.node = root
        self.stack = []

    def next(self) -> int:
        while self.node:
            self.stack.append(self.node)
            self.node = self.node.left
        self.node = self.stack.pop()
        val = self.node.val
        self.node = self.node.right
        return val

    def hasNext(self) -> bool:
        return self.stack or self.node
```

## 705. Design Hashset

### Solution 1:  separate-chaining = Hashset with modulus of prime number and using buckets with linkedlist for collisions

This uses separate chaining because each bucket contains a datastructure that stores all elements that have that bucket index
from the hash function.  So if there is a hash collision it will search through the linked list to add it into that bucket.  

This is still O(n) though because to search through linked list is O(n) but delete, add are O(1)

We could use a list for the buckets and it would be O(n) for search and delete but O(1) for add

We could finally use a self-balancing binary search tree which would give O(logn) for add, search, delete

```py
class MyHashSet:

    def __init__(self):
        self.MOD = 769
        self.values = [Bucket() for _ in range(self.MOD)]
    
    def hash_(self, key: int) -> int:
        return key % self.MOD
    
    def add(self, key: int) -> None:
        i = self.hash_(key)
        self.values[i].add(key)

    def remove(self, key: int) -> None:
        i = self.hash_(key)
        self.values[i].remove(key)

    def contains(self, key: int) -> bool:
        i = self.hash_(key)
        return self.values[i].contains(key)

class Bucket:
    def __init__(self):
        self.linked_list = LinkedList()

    def add(self, value):
        self.linked_list.add(value)

    def remove(self, value):
        self.linked_list.remove(value)

    def contains(self, value):
        return self.linked_list.contains(value)

class Node:
    def __init__(self, val=0, next_node=None):
        self.val = val
        self.next = next_node
    
class LinkedList:
    def __init__(self):
        self.head = Node()
    
    def add(self, val: int) -> None:
        if self.contains(val): return
        node = self.head
        while node.next:
            node = node.next
        node.next = Node(val)
    
    def remove(self, val: int) -> None:
        node = self.head
        while node.next:
            if node.next.val == val:
                node.next = node.next.next
                break
            node=node.next
        
    def contains(self, val: int) -> None:
        node = self.head.next
        while node:
            if node.val == val: return True
            node=node.next
        return False
```

### Solution 2: Separate chaining with buckets but bucket is binary search tree

Example of a facade design pattern

```py
class MyHashSet:

    def __init__(self):
        self.MOD = 769
        self.values = [Bucket() for _ in range(self.MOD)]
    
    def hash_(self, key: int) -> int:
        return key % self.MOD
    
    def add(self, key: int) -> None:
        i = self.hash_(key)
        self.values[i].add(key)

    def remove(self, key: int) -> None:
        i = self.hash_(key)
        self.values[i].remove(key)

    def contains(self, key: int) -> bool:
        i = self.hash_(key)
        return self.values[i].contains(key)

class Bucket:
    def __init__(self):
        self.tree = BSTree()

    def add(self, value):
        self.tree.root = self.tree.insertIntoBST(self.tree.root, value)

    def remove(self, value):
        self.tree.root = self.tree.deleteNode(self.tree.root, value)

    def contains(self, value):
        return (self.tree.searchBST(self.tree.root, value) is not None)

class TreeNode:
    def __init__(self, value):
        self.val = value
        self.left = None
        self.right = None

class BSTree:
    def __init__(self):
        self.root = None

    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        if root is None or val == root.val:
            return root

        return self.searchBST(root.left, val) if val < root.val \
            else self.searchBST(root.right, val)

    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
        if not root:
            return TreeNode(val)

        if val > root.val:
            # insert into the right subtree
            root.right = self.insertIntoBST(root.right, val)
        elif val == root.val:
            return root
        else:
            # insert into the left subtree
            root.left = self.insertIntoBST(root.left, val)
        return root

    def successor(self, root):
        """
        One step right and then always left
        """
        root = root.right
        while root.left:
            root = root.left
        return root.val

    def predecessor(self, root):
        """
        One step left and then always right
        """
        root = root.left
        while root.right:
            root = root.right
        return root.val

    def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
        if not root:
            return None

        # delete from the right subtree
        if key > root.val:
            root.right = self.deleteNode(root.right, key)
        # delete from the left subtree
        elif key < root.val:
            root.left = self.deleteNode(root.left, key)
        # delete the current node
        else:
            # the node is a leaf
            if not (root.left or root.right):
                root = None
            # the node is not a leaf and has a right child
            elif root.right:
                root.val = self.successor(root)
                root.right = self.deleteNode(root.right, root.val)
            # the node is not a leaf, has no right child, and has a left child
            else:
                root.val = self.predecessor(root)
                root.left = self.deleteNode(root.left, root.val)

        return root
```

## 706. Design HashMap

### Solution 1: 

```py
class MyHashMap:

    def __init__(self):
        self.MOD = 2069
        self.buckets = [Bucket() for _ in range(self.MOD)]
        
    def hash_(self, key: int) -> int:
        return key % self.MOD

    def put(self, key: int, value: int) -> None:
        bucket_index = self.hash_(key)
        self.buckets[bucket_index].add(key, value)

    def get(self, key: int) -> int:
        bucket_index = self.hash_(key)
        return self.buckets[bucket_index].search(key)

    def remove(self, key: int) -> None:
        bucket_index = self.hash_(key)
        self.buckets[bucket_index].remove(key)
        
class Node:
    def __init__(self, key=0, val=0,next_node=None):
        self.key = key
        self.val = val
        self.next = next_node
        
class Bucket:
    def __init__(self):
        self.head = Node()
        
    def search(self, key: int) -> int:
        node = self.head.next
        while node:
            if node.key == key: return node.val
            node=node.next
        return -1
    
    def add(self, key: int, val: int) -> None:
        node = self.head
        while node.next:
            if node.next.key == key: 
                node.next.val = val
                return
            node=node.next
        node.next = Node(key,val)
    
    def remove(self, key: int) -> None:
        node = self.head
        while node.next:
            if node.next.key == key:
                node.next = node.next.next
                return
            node=node.next
```

## 687. Longest Univalue Path

### Solution 1: Recursion + postorder dfs binary tree traversal

```py
class Solution:
    def longestUnivaluePath(self, root: Optional[TreeNode]) -> int:
        self.longest_path = 0
        def dfs(node):
            if not node: return 0
            left_len, right_len = dfs(node.left), dfs(node.right)
            left_arrow = right_arrow = 0
            if node.left and node.val==node.left.val:
                left_arrow = left_len + 1
            if node.right and node.val==node.right.val:
                right_arrow = right_len + 1
            self.longest_path = max(self.longest_path, left_arrow + right_arrow)
            return max(left_arrow, right_arrow)
                
        dfs(root)
        return self.longest_path
```

## 535. Encode and Decode TinyURL

### Solution 1: counter + hash table

The problems with this solution is that integer will get very large over time and the tinyurl will be no longy short. 
And in other languages it will overflow, python it won't overflow so that is fine. But there will be performance degredation potentially

```py
class Codec:
    def __init__(self):
        self.cnt = 0
        self.map = {}
    def encode(self, longUrl: str) -> str:
        """Encodes a URL to a shortened URL.
        """
        self.map[self.cnt] = longUrl
        shortUrl = 'http://tinyurl.com/' + str(self.cnt)
        self.cnt += 1
        return shortUrl
    def decode(self, shortUrl: str) -> str:
        """Decodes a shortened URL to its original URL.
        """
        return self.map[int(shortUrl.replace('http://tinyurl.com/', ''))]
```

### Solution 2: variable length encoding

The next level is to use more than just integers to fix the overflow and the fact the short url becomes long quickly

we can use 62 characters if we take integers + alphabet

```py
    def __init__(self):
        self.chars = string.ascii_letters + string.digits
        self.cnt = 0
        self.map = {}
    def encode(self, longUrl: str) -> str:
        """Encodes a URL to a shortened URL.
        """
        count = self.cnt
        encoding = []
        while count > 0:
            encoding.append(self.chars[count%62])
            count //= 62
        encoding_str = "".join(encoding)
        shortUrl = f'http://tinyurl.com/{encoding_str}'
        self.map[encoding_str] = longUrl
        return shortUrl
    def decode(self, shortUrl: str) -> str:
        """Decodes a shortened URL to its original URL.
        """
        return self.map[shortUrl.replace('http://tinyurl.com/', '')]
```

### Solution 3: python inbuilt hash function

```py
class Codec:
    def __init__(self):
        self.map = {}
    def encode(self, longUrl: str) -> str:
        """Encodes a URL to a shortened URL.
        """
        hash_ = hash(longUrl)
        self.map[hash_] = longUrl
        return f'http://tinyurl.com/{hash_}'
    def decode(self, shortUrl: str) -> str:
        """Decodes a shortened URL to its original URL.
        """
        return self.map[int(shortUrl.replace('http://tinyurl.com/', ''))]
```

### Solution 4: random fixed length encoding

62 characters with 6 as fixed size is 62^6

```py

```

## 1396. Design Underground System

### Solution 1: Multiple hash tables 

```py
class UndergroundSystem:

    def __init__(self):
        self.trip_times = Counter()
        self.trip_counts = Counter()
        self.checkedin = {}

    def checkIn(self, id: int, stationName: str, t: int) -> None:
        self.checkedin[id] = (stationName, t)

    def checkOut(self, id: int, stationName: str, t: int) -> None:
        startStation, t1 = self.checkedin[id]
        self.trip_counts[(startStation, stationName)] += 1
        self.trip_times[(startStation, stationName)] += (t-t1)

    def getAverageTime(self, startStation: str, endStation: str) -> float:
        return self.trip_times[(startStation, endStation)] / self.trip_counts[(startStation, endStation)]
```

## 284. Peeking Iterator

### Solution 1: iterator that stores next value for peek, and uses boolean for end so it works with any data type

```py
# Below is the interface for Iterator, which is already defined for you.
#
# class Iterator:
#     def __init__(self, nums):
#         """
#         Initializes an iterator object to the beginning of a list.
#         :type nums: List[int]
#         """
#
#     def hasNext(self):
#         """
#         Returns true if the iteration has more elements.
#         :rtype: bool
#         """
#
#     def next(self):
#         """
#         Returns the next element in the iteration.
#         :rtype: int
#         """

class PeekingIterator:
    def __init__(self, iterator):
        """
        Initialize your data structure here.
        :type iterator: Iterator
        """
        self.iterator = iterator
        self.next_data = self.iterator.next()
        self.not_end = True

    def peek(self):
        """
        Returns the next element in the iteration without advancing the iterator.
        :rtype: int
        """
        if not self.hasNext():
            raise StopIteration
        return self.next_data

    def next(self):
        """
        :rtype: int
        """
        if not self.hasNext():
            raise StopIteration
        data = self.next_data
        self.not_end = False
        if self.iterator.hasNext():
            self.next_data = self.iterator.next()
            self.not_end = True
        return data

    def hasNext(self):
        """
        :rtype: bool
        """
        return self.not_end

# Your PeekingIterator object will be instantiated and called as such:
# iter = PeekingIterator(Iterator(nums))
# while iter.hasNext():
#     val = iter.peek()   # Get the next element but not advance the iterator.
#     iter.next()         # Should return the same value as [val].
```

## 1166. Design File System

### Solution 1: hash table + hash table for tree

```py
class FileSystem:

    def __init__(self):
        self.paths = {}

    def createPath(self, path: str, value: int) -> bool:
        if path in self.paths: return False
        parent = path[:path.rfind('/')]
        if len(parent) > 1 and parent not in self.paths: return False
        self.paths[path] = value
        return True

    def get(self, path: str) -> int:
        return self.paths[path] if path in self.paths else -1
```

### Solution 2: Trie datastructure

```py
class TrieNode:
    def __init__(self, name):
        self.children = defaultdict(TrieNode)
        self.name = name
        self.value = -1
class FileSystem:

    def __init__(self):
        self.root = TrieNode('')

    def createPath(self, path: str, value: int) -> bool:
        components = path.split('/')
        node = self.root
        for i in range(1,len(components)):
            name = components[i]
            if name not in node.children:
                if i==len(components)-1:
                    node.children[name] = TrieNode(name)
                else: 
                    return False
            node=node.children[name]
        if node.value != -1: return False
        node.value = value
        return True

    def get(self, path: str) -> int:
        components = path.split('/')
        node = self.root
        for i in range(1,len(components)):
            name = components[i]
            if name not in node.children: return -1
            node=node.children[name]
        return node.value
```

## 1584. Min Cost to Connect All Points

### Solution 1: Kruskal's Algorithm to find Minimimum Spanning Tree + UnionFind + sort

```py
class UnionFind:
    def __init__(self,n):
        self.size = [0]*n
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
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        n = len(points)
        edges = []
        manhattan = lambda p1, p2: abs(p1[0]-p2[0])+abs(p1[1]-p2[1])
        for i in range(n):
            for j in range(i+1,n):
                edges.append((manhattan(points[i], points[j]), i, j))
        edges.sort()
        dsu = UnionFind(n)
        minCost = 0
        for cost, u, v in edges:
            if dsu.union(u,v):
                minCost += cost
            if dsu.size[u] == n: break
        return minCost
```

## 1202. Smallest String With Swaps

### Solution 1: Union find + minheap for each connected component

```py
class UnionFind:
    pass
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        n = len(s)
        dsu = UnionFind(n)
        for u, v in pairs:
            dsu.union(u,v)
        connected_components = defaultdict(list)
        for i in range(n):
            heappush(connected_components[dsu.find(i)], s[i])
        swapped = []
        for i in range(n):
            swapped.append(heappop(connected_components[dsu.find(i)]))
        return "".join(swapped)
```

### Solution 1: dfs + sorting

```py
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        def dfs(u):
            component.append(u)
            for v in graph[u]:
                if visited[v]: continue
                visited[v] = 1
                dfs(v)
        n = len(s)
        graph = defaultdict(list)
        for u, v in pairs:
            graph[u].append(v)
            graph[v].append(u)
        visited = [0]*n
        result = list(s)
        for i in range(n):
            if visited[i]: continue
            component = []
            visited[i] = 1
            dfs(i)
            component.sort()
            chars = [s[j] for j in component]
            chars.sort()
            for j, k in enumerate(component):
                result[k] = chars[j]
        return "".join(result)
```

## 1631. Path With Minimum Effort

### Solution 1: min heap datastructure + memoize cheapest cost to that cell (similar to dijkstra algorithm)

```py
class Solution:
    def minimumEffortPath(self, heights: List[List[int]]) -> int:
        R, C = len(heights), len(heights[0])
        heap = [(0, 0, 0)] # (cost, row, column)
        diff_matrix = [[inf]*C for _ in range(R)]
        in_boundary = lambda r, c: 0<=r<R and 0<=c<C
        while heap:
            cost, row, col = heappop(heap)
            if row==R-1 and col==C-1:
                return cost
            for r, c in map(lambda x: (row+x[0], col+x[1]), [(1,0),(-1,0),(0,1),(0,-1)]):
                if not in_boundary(r,c): continue
                ncost = max(abs(heights[r][c]-heights[row][col]), cost)
                if ncost < diff_matrix[r][c]:
                    heappush(heap, (ncost, r, c))
                    diff_matrix[r][c] = ncost
        return -1
```

### Solution 2: Binary search + BFS

```py
class Solution:
    def minimumEffortPath(self, heights: List[List[int]]) -> int:
        left, right = 0, max((max(height) for height in heights))-min((min(height) for height in heights))
        R, C = len(heights), len(heights[0])
        def bfs(threshold):
            queue = deque([(0, 0)]) # (row, col)
            visited = set()
            in_boundary = lambda r, c: 0<=r<R and 0<=c<C
            while queue:
                row, col = queue.popleft()
                if row==R-1 and col==C-1:
                    return True
                for nr, nc in map(lambda x: (row+x[0], col+x[1]), [(1,0),(-1,0),(0,1),(0,-1)]):
                    if not in_boundary(nr,nc): continue
                    if (nr,nc) in visited or abs(heights[row][col]-heights[nr][nc]) > threshold: continue
                    visited.add((nr,nc))
                    queue.append((nr,nc))
            return False
        while left < right:
            mid = (left+right)>>1
            if not bfs(mid):
                left = mid+1
            else:
                right = mid
        return left
```

### Solution 3: binary search + dfs

Turns out to be much faster than using bfs too. 

```py
class Solution:
    def minimumEffortPath(self, heights: List[List[int]]) -> int:
        left, right = 0, max((max(height) for height in heights))-min((min(height) for height in heights))
        R, C = len(heights), len(heights[0])
        def dfs(row, col, threshold):
            if row==R-1 and col==C-1: return True
            visited[row][col] = 1
            for nr, nc in map(lambda x: (row+x[0], col+x[1]), [(1,0),(-1,0),(0,1),(0,-1)]):
                if not (0<=nr<R and 0<=nc<C): continue
                if visited[nr][nc] or abs(heights[row][col]-heights[nr][nc]) > threshold: continue
                visited[nr][nc] = 1
                if dfs(nr,nc,threshold): return True
            return False
        while left < right:
            mid = (left+right)>>1
            visited = [[0]*C for _ in range(R)]
            if not dfs(0,0,mid):
                left = mid+1
            else:
                right = mid
        return left
```

### Solution 4: Union Find + it contains the first and last cell in path

```py
class UnionFind:
    pass
class Solution:
    def minimumEffortPath(self, heights: List[List[int]]) -> int:
        R, C = len(heights), len(heights[0])
        dsu = UnionFind(R*C)
        edges = []
        in_boundary = lambda r, c: 0<=r<R and 0<=c<C
        for r, c in product(range(R), range(C)):
            for nr, nc in map(lambda x: (r+x[0],c+x[1]), [(1,0),(-1,0),(0,1),(0,-1)]):
                if not in_boundary(nr,nc): continue
                cost = abs(heights[r][c]-heights[nr][nc])
                node1, node2 = r*C+c, nr*C+nc
                edges.append((cost, node1, node2))
        edges.sort()
        for cost, u, v in edges:
            dsu.union(u,v)
            if dsu.find(0) == dsu.find(R*C-1):
                return cost
        return 0 # single node 
```


## 399. Evaluate Division

### Solution 1: Union Find to check if evaluate division will work + bfs

This is rather brute force, not really saving values, and just recomputing bfs many times. 

```py
class UnionFind:
    pass
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        graph = defaultdict(list)
        compressed = {}
        for i, (a, b) in enumerate(equations):
            if a not in compressed:
                compressed[a] = len(compressed)
            if b not in compressed:
                compressed[b] = len(compressed)
            u, v = compressed[a], compressed[b]
            graph[u].append((v, values[i]))
            graph[v].append((u, 1.0/values[i]))
        n=len(compressed)
        dsu = UnionFind(n)
        for a, b in equations:
            dsu.union(compressed[a], compressed[b])
        def bfs(u, v):
            queue = deque([(u, 1.0)])
            visited = set()
            visited.add(u)
            while queue:
                node, val = queue.popleft()
                if node == v: return val
                for nei, weight in graph[node]:
                    if nei in visited: continue
                    queue.append((nei, val*weight))
                    visited.add(nei)
            return -1.0
        answer = [-1.0]*len(queries)
        for i, (a, b) in enumerate(queries):
            if a not in compressed or b not in compressed: continue
            u, v = compressed[a], compressed[b]
            if dsu.find(u) != dsu.find(v): continue
            answer[i] = bfs(u, v)
        return answer
            
```

### Solution 2: Brute Force DFS

```py
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        graph = defaultdict(defaultdict)
        for (dividend, divisor), val in zip(equations, values):
            graph[dividend][divisor] = val
            graph[divisor][dividend] = 1/val
        def dfs(current_node, target_node):
            if current_node == target_node: return 1.0
            prod = inf
            for nei in graph[current_node]:
                if nei in visited: continue
                visited.add(nei)
                prod = (graph[current_node][nei]*dfs(nei, target_node))
                if prod != inf: return prod
            return prod 
        answer = [-1.0]*len(queries)
        for i, (a, b) in enumerate(queries):
            if a not in graph or b not in graph: continue
            visited = set()
            res = dfs(a,b)
            answer[i] = res if res != inf else -1.0
        return answer
```

### Solution 3: Floyd Warshall Algorithm

works because we have small number of vertices, the algorithm takes O(V^3) time complexity, 
very good for dense graphs that have many edges.  

think k internal nodes, 

```py
class Solution:
    def calcEquation(self, equations, values, queries):
        graph = defaultdict(dict)
        # INITIALIZE THE VALUES FOR EDGES AND ITSELF
        for (a, b), val in zip(equations, values):
            graph[a][b] = val
            graph[b][a] = 1.0/val
            graph[a][a] = 1.0
            graph[b][b] = 1.0
        # (i,j) => (i,k) + (k,j), k is internal node
        for k, i, j in permutations(graph, 3):
            if k in graph[i] and j in graph[k]:
                graph[i][j] = graph[i][k]*graph[k][j]
        return [graph[i][j] if j in graph[i] else -1.0 for i,j in queries]
```


```py
class Solution:
    def calcEquation(self, equations, values, queries):
        graph = defaultdict(dict)
        # INITIALIZE THE VALUES FOR EDGES AND ITSELF
        for (a, b), val in zip(equations, values):
            graph[a][b] = val
            graph[b][a] = 1.0/val
            graph[a][a] = 1.0
            graph[b][b] = 1.0
        n = len(graph)
        # (i,j) => (i,k) + (k,j), k is internal node
        for k in graph.keys():
            for i in graph.keys():
                if k not in graph[i]: continue
                for j in graph.keys():
                    if j not in graph[k]: continue
                    graph[i][j] = graph[i][k]*graph[k][j]
        return [graph[i][j] if j in graph[i] else -1.0 for i,j in queries]
```

### Solution 4: Union Find with Weighted Edges

```py

```

## 431. Encode N-ary Tree to Binary Tree

### Solution 1: BFS type algorithm 

The strategy is that for a given nary tree, for the first child we add it as a left node in the binary tree
Then from that binary tree we add each additional child node as a right node from each binary node. 


```py
class Codec:
    # Encodes an n-ary tree to a binary tree.
    def encode(self, root: 'Optional[Node]') -> Optional[TreeNode]:
        if not root: return None
        binary_root = TreeNode(root.val)
        queue = deque([(root, binary_root)])
        while queue:
            nary_node, binary_node = queue.popleft()
            current_node = binary_node # current binary node
            for child in nary_node.children:
                if not binary_node.left:
                    current_node.left = TreeNode(child.val)
                    current_node=current_node.left
                else:
                    current_node.right = TreeNode(child.val)
                    current_node=current_node.right
                queue.append((child, current_node))
        return binary_root
	
	# Decodes your binary tree to an n-ary tree.
    def decode(self, data: Optional[TreeNode]) -> 'Optional[Node]':
        if not data: return None
        queue = deque([(data, None)])
        while queue:
            binary_node, nary_node = queue.popleft()
            # nary node is going to be the parent for the current nary node
            current_node = Node(binary_node.val, []) # current nary node
            if not nary_node:
                root = current_node
            else:
                nary_node.children.append(current_node)
            if binary_node.left:
                queue.append((binary_node.left, current_node))
            if binary_node.right:
                queue.append((binary_node.right, nary_node))
        return root
```

### Solution 2: Alternative BFS

```py
class Codec:
    # Encodes an n-ary tree to a binary tree.
    def encode(self, root: 'Optional[Node]') -> Optional[TreeNode]:
        if not root: return None
        binary_root = TreeNode(root.val)
        queue = deque([(root, binary_root)])
        while queue:
            nary_node, binary_node = queue.popleft()
            current_node = binary_node # current binary node
            for child in nary_node.children:
                if not binary_node.left:
                    current_node.left = TreeNode(child.val)
                    current_node=current_node.left
                else:
                    current_node.right = TreeNode(child.val)
                    current_node=current_node.right
                queue.append((child, current_node))
        return binary_root
	
	# Decodes your binary tree to an n-ary tree.
    def decode(self, data: Optional[TreeNode]) -> 'Optional[Node]':
        if not data: return None
        root = Node(data.val, [])
        queue = deque([(data, root)])
        while queue:
            binary_node, nary_node = queue.popleft()
            # nary node is going to be the parent for the current nary node
            sibling = binary_node.left
            while sibling:
                current_node = Node(sibling.val, []) # current nary node
                nary_node.children.append(current_node)
                queue.append((sibling, current_node))
                sibling = sibling.right
        return root
```

### Solution 3: DFS with Recursion

```py
class Codec:
    # Encodes an n-ary tree to a binary tree.
    def encode(self, root: 'Optional[Node]') -> Optional[TreeNode]:
        if not root: return None
        binary_root = TreeNode(root.val)
        if not root.children: return binary_root
        binary_root.left = self.encode(root.children[0])
        current_node = binary_root.left
        for i in range(1,len(root.children)):
            current_node.right = self.encode(root.children[i])
            current_node = current_node.right
        return binary_root
	
	# Decodes your binary tree to an n-ary tree.
    def decode(self, data: Optional[TreeNode]) -> 'Optional[Node]':
        if not data: return None
        root = Node(data.val, [])
        sibling_node = data.left
        while sibling_node:
            root.children.append(self.decode(sibling_node))
            sibling_node = sibling_node.right
        return root
```

## 785. Is Graph Bipartite?

### Solution 1: dfs with 2 coloring algorithm 

```py
class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        n=len(graph)
        colors = {}
        def dfs(node):
            if node not in colors: 
                colors[node] = 0
            for nei in graph[node]:
                if nei in colors and node in colors and colors[nei]==colors[node]: return False
                if nei in colors and node in colors: continue
                colors[nei] = colors[node]^1
                if not dfs(nei): return False
            return True
        
        for i in range(n):
            if not dfs(i): return False
        return True
```

### Solution 2: BFS with 2 coloring algorithm 

```py
class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        n=len(graph)
        colors = {}
        def bfs(node):
            queue = deque([node])
            colors[node] = 0
            while queue:
                node = queue.popleft()
                for nei in graph[node]:
                    if node in colors and nei in colors:
                        if colors[node]==colors[nei]: return False
                        continue
                    colors[nei] = colors[node]^1
                    queue.append(nei)
            return True
        for i in range(n):
            if i in colors: continue
            if not bfs(i): return False
        return True
```

### Solution 3: Union Find

idea is that a node and all of it's neighbors should be in two disjoint sets.  If you ever catch
a node being in the same disjoint set as one of it's neighbor it is not a bipartite graph

```py
class UnionFind:
    pass
class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        n=len(graph)
        dsu = UnionFind(n)
        for i in range(n):
            for j in graph[i]:
                if dsu.find(i) == dsu.find(j): return False
                dsu.union(graph[i][0], j)
        return True
```

## 905. Sort Array By Parity

### Solution 1: sorting by parity

```py
class Solution:
    def sortArrayByParity(self, nums: List[int]) -> List[int]:
        return sorted(nums, key=lambda x: x%2)
```

### Solution 2: two pointers

```py
class Solution:
    def sortArrayByParity(self, nums: List[int]) -> List[int]:
        n=len(nums)
        i, j = 0, n-1
        while i < j:
            if nums[i]%2==0:
                i += 1
            else:
                nums[i], nums[j] = nums[j], nums[i]
                j -= 1
        return nums
```

## 581. Shortest Unsorted Continuous Subarray

### Solution 1: Prefix Max + Suffix Min + two pointers + 3 loops + extra space

suppose it is sorted
nums = [2,4,6,6,6,6,8,9,10,15]
prefixMax = [-f,02,04,06,06,08,09,10,15]
suffixMin = [02,04,06,06,08,09,10,15,+f]
since prefixMax[i]==suffixMin[i-1] we know we can keep moving and it is sorted
the reason is that prefix max says the largest element I've seen at say index = 0, 
and so we ask the question what is the smallest element I've seen at index = 0, if they are the same
that would indicate that it is bost the max in the prefix and the min in the suffix, thus it is
the largest element so far, while also being the samllest from the suffix, so that means it belongs
at the prefix, and as they keep equal we are good, cause that means it is sorted according to this prefix max and suffix min
it doesn't mean it is the largest too, just that it is the equal to the largest or is the largest
and it doesn't mean it is the smallest, just that it is equal to the smallest or is the smallest. 
then again we know we can do the same from the right side, cause if suffixmin equals prefix max that means we can move it to left, cause that means it is th elargest element so far. 

```py
class Solution:
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        n=len(nums)
        left, right = 1, n
        prefixMax, suffixMin = [-inf]*(n+1), [inf]*(n+1)
        for i in range(n):
            prefixMax[i+1] = max(prefixMax[i], nums[i])
        for i in range(n)[::-1]:
            suffixMin[i] = min(suffixMin[i+1], nums[i])
        while left <= right:
            if prefixMax[left]==suffixMin[left-1]:
                left += 1
            elif prefixMax[right]==suffixMin[right-1]:
                right -= 1
            else:
                break
        return right - left +1
```

### Solution 2: prefix Max + suffix Min + find last index + find first index + 2 O(n) loops + no extra space

```py
class Solution:
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        # find the last index that breaks the sort
        prefixMax = -inf
        right = 0
        for i in range(n):
            if nums[i] < prefixMax:
                right = i
            else:
                prefixMax = nums[i]
        # find the first index that breaks sort
        suffixMin = inf
        left = n-1
        for i in range(n)[::-1]:
            if nums[i] > suffixMin:
                left = i
            else:
                suffixMin = nums[i]
        return right-left+1 if right>0 else 0
```

```py
class Solution:
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        # find the last index that breaks the sort
        # find the first index that breaks sort
        prefixMax, suffixMin = -inf, inf
        left, right = n-1,0
        for i in range(n):
            if nums[i] < prefixMax:
                right = i
            if nums[n-i-1] > suffixMin:
                left = n-i-1
            prefixMax = max(prefixMax, nums[i])
            suffixMin = min(suffixMin, nums[n-i-1])
        return right-left+1 if right>0 else 0
```

### Solution 3: Using two deque for nums and sorted(nums), then just pop from left and from right as long as they are equal

```py
class Solution:
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        nq = deque(nums)
        sq = deque(sorted(nums))
        while nq and nq[0]==sq[0]:
            nq.popleft(), sq.popleft()
        while nq and nq[-1]==sq[-1]:
            nq.pop(), sq.pop()
        return len(nq)
```

## 1679. Max Number of K-Sum Pairs

### Solution 1: sort + two pointers

```py
class Solution:
    def maxOperations(self, nums: List[int], k: int) -> int:
        nums.sort()
        n=len(nums)
        i, j = 0, n-1
        cnt = 0
        while i < j:
            if nums[i]+nums[j] == k:
                cnt += 1
                i += 1
                j -= 1
            elif nums[i]+nums[j] > k:
                j -= 1
            else:
                i += 1
        return cnt
```

### Solution 2: count + hash table

```py
class Solution:
    def maxOperations(self, nums: List[int], k: int) -> int:
        count = Counter()
        cnt = 0
        for x in nums:
            y = k - x
            if count[y] > 0:
                count[y] -= 1
                cnt += 1
            else:
                count[x] += 1
        return cnt
```

## 225. Implement Stack using Queues

### Solution 1: 2 queues push O(n), pop O(1)

```py
class MyStack:

    def __init__(self):
        self.queue = deque()

    def push(self, x: int) -> None:
        tmp_queue = deque([x])
        while self.queue:
            tmp_queue.append(self.queue.popleft())
        self.queue = tmp_queue

    def pop(self) -> int:
        return self.queue.popleft()

    def top(self) -> int:
        return self.queue[0]

    def empty(self) -> bool:
        return not self.queue
```

### Solution 2: Single queue with push O(n) and pop O(1)

```py
class MyStack:

    def __init__(self):
        self.queue = deque()

    def push(self, x: int) -> None:
        n = len(self.queue)
        self.queue.append(x)
        for _ in range(n):
            self.queue.append(self.queue.popleft())

    def pop(self) -> int:
        return self.queue.popleft()

    def top(self) -> int:
        return self.queue[0]

    def empty(self) -> bool:
        return not self.queue
```

### Solution 3:  queue of queues

```py
class MyStack:

    def __init__(self):
        self.queue = deque()

    def push(self, x: int) -> None:
        q = deque([x])
        q.append(self.queue)
        self.queue = q

    def pop(self) -> int:
        elem = self.queue.popleft()
        self.queue = self.queue.popleft()
        return elem

    def top(self) -> int:
        return self.queue[0]

    def empty(self) -> bool:
        return not self.queue
```


## 1209. Remove All Adjacent Duplicates in String II

### Solution 1: stack + store count

```py
class Solution:
    def removeDuplicates(self, s: str, k: int) -> str:
        stack = []
        for ch in s:
            if not stack or stack[-1][0] != ch:
                stack.append([ch, 1])
            elif stack[-1][0] == ch:
                stack[-1][1] += 1
            if stack and stack[-1][1] == k:
                stack.pop()
        return "".join(char*cnt for char, cnt in stack)
```

## 232. Implement Queue using Stacks

### Solution 1: temporary stack in push O(n), O(1) for pop 

```py
class MyQueue:

    def __init__(self):
        self.stack = []

    def push(self, x: int) -> None:
        tmp_stack = []
        while self.stack:
            tmp_stack.append(self.stack.pop())
        tmp_stack.append(x)
        while tmp_stack:
            self.stack.append(tmp_stack.pop())
        
    def pop(self) -> int:
        return self.stack.pop()

    def peek(self) -> int:
        return self.stack[-1]

    def empty(self) -> bool:
        return not bool(self.stack)
```

### Solution 2: two stacks, when one is empty place into there, O(1) push, and amortized O(1) pop

```py
class MyQueue:
    def __init__(self):
        self.stack1, self.stack2 = [], []
    def push(self, x: int) -> None:
        self.stack1.append(x)  
    def pop(self) -> int:
        self.move()
        return self.stack2.pop()
    def peek(self) -> int:
        self.move()
        return self.stack2[-1]
    def move(self) -> None:
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
    def empty(self) -> bool:
        return not bool(self.stack1) and not bool(self.stack2)
```

## 456. 132 Pattern

### Solution 1: Sorted Dictionary + binary search through sorted dictionary to find elemen that is greater than prefix min and less than current element. 

```py
from sortedcontainers import SortedDict
class Solution:
    def find132pattern(self, nums: List[int]) -> bool:
        n = len(nums)
        sdict = SortedDict()
        for i in range(1,n):
            cnt = sdict.setdefault(nums[i], 0) + 1
            sdict[nums[i]] = cnt
        prefixMin = nums[0]
        for j in range(1,n-1):
            sdict[nums[j]] -= 1
            if sdict[nums[j]] == 0:
                sdict.pop(nums[j])
            k = sdict.bisect_right(prefixMin)
            keys = sdict.keys()
            if k < len(keys) and keys[k] < nums[j]: return True
            prefixMin = min(prefixMin, nums[j])
        return False
```

### Solution 2: prefix min for nums[i] candidates + min stack for nums[k] candidates + backwards iteration for nums[j] candidates to find if nums[i] < nums[k] < nums[j]

```py
class Solution:
    def find132pattern(self, nums: List[int]) -> bool:
        n = len(nums)
        prefixMin = list(accumulate(nums, min, initial=inf))
        minStack = [] # top element is minimum of stack, sorted in descending order
        k = n
        for i in range(n)[::-1]:
            k = bisect_right(nums, prefixMin[i], k, n)
            if k < n and prefixMin[i] < nums[k] < nums[i]: return True
            k -= 1
            nums[k] = nums[i]
        return False
```

### Solution 3: prefix min for nums[i] + storing values in nums array with a specific left and right pointer in nums array that is the segment to be binary searched to find elements that are greater than nums[i] for nums[k], then just need to check nums[k] < nums[j]

```py
class Solution:
    def find132pattern(self, nums: List[int]) -> bool:
        n = len(nums)
        prefixMin = list(accumulate(nums, min, initial=inf))
        minStack = [] # top element is minimum of stack, sorted in descending order
        k = n
        for i in range(n)[::-1]:
            k = bisect_right(nums, prefixMin[i], k, n)
            if k < n and prefixMin[i] < nums[k] < nums[i]: return True
            k -= 1
            nums[k] = nums[i]
        return False
```

## 484. Find Permutation

### Solution 1: stack based to reverse when it is D, and then always add to I and add from stack.  

```py
class Solution:
    def findPermutation(self, s: str) -> List[int]:
        result = []
        stack = []
        for i, ch in enumerate(s, start=1):
            if ch=='I':
                stack.append(i)
                while stack:
                    result.append(stack.pop())
            else:
                stack.append(i)
        stack.append(len(s)+1)
        while stack:
            result.append(stack.pop())
        return result
```

### Solution 2: greedily fill in the result array with elements in decreasing order everytime see an I, and add until you hit the len of result array, 

```py
class Solution:
    def findPermutation(self, s: str) -> List[int]:
        result = []
        for i, ch in enumerate(s, start=1):
            if ch=='I':
                result.extend(range(i, len(result),-1))
        result.extend(range(len(s)+1,len(result),-1))
        return result
```

## 2264. Largest 3-Same-Digit Number in String

### Solution 1: Check previous values

```py
class Solution:
    def largestGoodInteger(self, num: str) -> str:
        return max(num[i-2:i+1] if num[i-2]==num[i-1]==num[i] else "" for i in range(2,len(num)))
```

## 2265. Count Nodes Equal to Average of Subtree

### Solution 1: recursion and postorder traversal of binary tree

```py
class Solution:
    def averageOfSubtree(self, root: Optional[TreeNode]) -> int:
        self.cnt = 0
        def dfs(node):
            if not node: return 0, 0
            lsum, lcnt = dfs(node.left)
            rsum, rcnt = dfs(node.right)
            sum_ = lsum + rsum + node.val
            cnt_ = lcnt + rcnt + 1
            if sum_//cnt_ == node.val:
                self.cnt += 1
            return sum_, cnt_
        dfs(root)
        return self.cnt
```

## 2266. Count Number of Texts

### Solution 1:

```py

```

## 

### Solution 1: dynamic programming with state being the row,col,balance of the parentheses

```py
class Solution:
    def hasValidPath(self, grid: List[List[str]]) -> bool:
        R, C = len(grid), len(grid[0])
        for r, c in product(range(R), range(C)):
            grid[r][c] = 1 if grid[r][c] == '(' else -1
        stack = []
        if grid[0][0] == ')': return False
        stack.append((0,0,1))
        visited = set()
        in_bounds = lambda r, c: 0<=r<R and 0<=c<C
        while stack:
            row, col, bal = stack.pop()
            for nr, nc in [(row+1,col),(row,col+1)]:
                if not in_bounds(nr,nc): continue
                nbal = bal + grid[nr][nc]
                if nbal < 0 or nbal > (R+C)//2: continue
                state = (nr,nc,nbal)
                if state in visited: continue
                if nr==R-1 and nc==C-1 and nbal == 0: 
                    return True
                visited.add(state)
                stack.append(state)
        return False
```

## 341. Flatten Nested List Iterator

### Solution 1: Preprocess to flatten the list with a preorder traversal, and treating the nestedlist as a tree, where the leaf nodes are integers and internal nodes are nestedLists, gives a normal list of integers, and that is easy to creat iterator with a pointer.  Uses Recursion for the preorder traversal

```py
class NestedIterator:
    def __init__(self, nestedList: [NestedInteger]):
        self.nestedList = nestedList
        self.flatList = []
        self.flattenList(nestedList)
        self.pointer = 0
        
    def flattenList(self, node):
        for nei_node in node:
            if nei_node.isInteger():
                self.flatList.append(nei_node.getInteger())
            else:
                self.flattenList(nei_node.getList())
    
    def next(self) -> int:
        elem = self.flatList[self.pointer]
        self.pointer += 1
        return elem
    
    def hasNext(self) -> bool:
        return self.pointer < len(self.flatList)
```

### Solution 2: Same as solution 1, but just add a yield and make the flat list a flat list generator, then included peeked so that we can check if it hasNext element any number of times. 

```py
class NestedIterator:
    def __init__(self, nestedList: [NestedInteger]):
        self.flatList_generator = self.flattenList(nestedList)
        self.peeked = None
        
    def flattenList(self, node):
        for nei_node in node:
            if nei_node.isInteger():
                yield nei_node.getInteger()
            else:
                yield from self.flattenList(nei_node.getList())
    
    def next(self) -> int:
        if not self.hasNext(): return None # so we can get the next element for peeked
        integer, self.peeked = self.peeked, None
        return integer
    
    def hasNext(self) -> bool:
        if self.peeked is not None: return True
        try:
            self.peeked = next(self.flatList_generator)
            return True
        except:
            return False
```

### Solution 3: stack of nestedIntegers

```py

```

### Solution 4: optimized stack with 2 stack, pointers and nested list

```py

```

## 251. Flatten 2D Vector

### Solution 1: flatten generator with iterative solution 

```py
class Vector2D:

    def __init__(self, vec: List[List[int]]):
        self.flattenGen = self.flatten_generator(vec)
        self.peeked = None
        
    def flatten_generator(self, vec):
        for lst in vec:
            for elem in lst:
                yield elem

    def next(self) -> int:
        if not self.hasNext(): return None
        integer, self.peeked = self.peeked, None
        return integer

    def hasNext(self) -> bool:
        if self.peeked is not None: return True
        try: 
            self.peeked = next(self.flattenGen)
            return True
        except:
            return False
```

### Solution 2: two pointers 

```py
class Vector2D:

    def __init__(self, vec: List[List[int]]):
        self.vec = vec
        self.outer = self.inner = 0
        
    def update_pointers(self):
        while self.outer < len(self.vec) and self.inner == len(self.vec[self.outer]):
            self.outer += 1
            self.inner = 0

    def next(self) -> int:
        if not self.hasNext(): return
        elem = self.vec[self.outer][self.inner]
        self.inner += 1
        return elem

    def hasNext(self) -> bool:
        self.update_pointers()
        return self.outer < len(self.vec)
```

## 216 Combination Sum III

### Solution 1: Uses itertools.combinations in python to generate combinations of an iterage of range(1,10) with length of k

```py
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        return [combo for combo in combinations(range(1,10),k) if sum(combo) == n]
```

### Solution 2: Iterates through all possible combinations of 1-9 digits with a bitmask and checks if the sum of digits equal to n

```py
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        combinations = []
        for bitmask in range(1, 1<<9):
            if bitmask.bit_count() != k: continue
            cur_comb = []
            for i in range(9):
                if (bitmask>>i)&1:
                    cur_comb.append(i+1)
            if sum(cur_comb) == n:
                combinations.append(cur_comb)
        return combinations
```

## 223. Rectangle Area

### Solution 1: Geometry and Math

```py
class Solution:
    def computeArea(self, ax1: int, ay1: int, ax2: int, ay2: int, bx1: int, by1: int, bx2: int, by2: int) -> int:
        x_overlap, y_overlap = max(min(ax2,bx2) - max(ax1,bx1), 0), max(min(ay2,by2) - max(ay1,by1),0)
        area_overlap = x_overlap*y_overlap
        get_area = lambda x1, x2, y1, y2: (x2-x1)*(y2-y1)
        return get_area(ax1,ax2,ay1,ay2) + get_area(bx1,bx2,by1,by2) - area_overlap
```

## 2268. Minimum Number of Keypresses

### Solution 1: Math + count 

```py
class Solution:
    def minimumKeypresses(self, s: str) -> int:
        return sum(cnt*((i+9)//9) for i, cnt in enumerate(sorted(Counter(s).values(), reverse=True)))
```

## 2254. Design Video Sharing Platform

### Solution 1: hash table + minheap

```py
Video = namedtuple('Video', ['video', 'likes', 'dislikes', 'views'])
class VideoSharingPlatform:
    
    def __init__(self):
        self.video_dict = {}
        self.minheap = []
        self.pointer = 0

    def upload(self, video: str) -> int:
        if self.minheap:
            videoId = heappop(self.minheap)
        else:
            videoId = self.pointer
            self.pointer += 1
        self.video_dict[videoId] = Video(video,0,0,0)
        return videoId
        
    def remove(self, videoId: int) -> None:
        if videoId not in self.video_dict: return
        self.video_dict.pop(videoId)
        heappush(self.minheap, videoId)

    def watch(self, videoId: int, startMinute: int, endMinute: int) -> str:
        if videoId not in self.video_dict: return "-1"
        video = self.video_dict[videoId]
        self.video_dict[videoId] = video._replace(views=video.views+1)
        return video.video[startMinute:endMinute+1]

    def like(self, videoId: int) -> None:
        if videoId in self.video_dict:
            video = self.video_dict[videoId]
            self.video_dict[videoId] = video._replace(likes=video.likes + 1)

    def dislike(self, videoId: int) -> None:
        if videoId in self.video_dict:
            video = self.video_dict[videoId]
            self.video_dict[videoId] = video._replace(dislikes=video.dislikes+1)

    def getLikesAndDislikes(self, videoId: int) -> List[int]:
        if videoId not in self.video_dict: return [-1]
        likes, dislikes = self.video_dict[videoId].likes, self.video_dict[videoId].dislikes
        return [likes, dislikes]

    def getViews(self, videoId: int) -> int:
        if videoId not in self.video_dict: return -1
        return self.video_dict[videoId].views
```

## 117. Populating Next Right Pointers in Each Node II

### Solution 1: Preorder dfs with recursion 

```py
class Solution:
    def connect(self, root: 'Node') -> 'Node':
        depth_node_dict = {}
        def preorder_dfs(depth, node):
            if not node: return
            if depth in depth_node_dict:
                depth_node_dict[depth].next = node
            depth_node_dict[depth] = node
            preorder_dfs(depth+1, node.left)
            preorder_dfs(depth+1, node.right)
        preorder_dfs(0, root)
        return root
```

### Solution 2: stack based solution

```py
class Solution:
    def connect(self, root: 'Node') -> 'Node':
        depth_node_dict = {}
        stack = [(0, root)]
        while stack:
            depth, node = stack.pop()
            if not node: continue
            print(depth, node.val)
            if depth in depth_node_dict:
                depth_node_dict[depth].next = node
            depth_node_dict[depth] = node
            stack.append((depth+1, node.right))
            stack.append((depth+1, node.left))
        return root
```

### Solution 3: BFS level order traversal to remove needing the dictionary of previous node

```py
class Solution:
    def connect(self, root: 'Node') -> 'Node':
        queue = deque([(root)])
        while queue:
            sz = len(queue)
            prev_node = None
            for _ in range(sz):
                node = queue.popleft()
                if not node: continue
                if prev_node:
                    prev_node.next = node
                prev_node = node
                queue.append(node.left)
                queue.append(node.right)
        return root
```

### Solution 4: no extra space + using current level as linked list with next pointers set, and set next pointers for next level

```py
class Solution:
    def processChild(self, child_node, prev_node, leftmost_node):
        if child_node:
            if prev_node:
                prev_node.next = child_node
            else:
                leftmost_node = child_node
            prev_node = child_node
        return prev_node, leftmost_node
    def connect(self, root: 'Node') -> 'Node':
        leftmost_node = root
        
        while leftmost_node:
            
            curr_node, leftmost_node, prev_node = leftmost_node, None, None
            while curr_node:
                prev_node, leftmost_node = self.processChild(curr_node.left, prev_node, leftmost_node)
                prev_node, leftmost_node = self.processChild(curr_node.right, prev_node, leftmost_node)
                curr_node=curr_node.next
            
        return root
```

## Network Delay Time

### Solution 1: shortest path from single source in directed graph + dijkstra algorithm + O((V+E)logV)

```py
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        graph = defaultdict(list)
        for u, v, w in times:
            graph[u].append((v,w))
        minheap = [(0, k)] # (time, node)
        dist = defaultdict(lambda: inf)
        dist[k] = 0
        while minheap:
            time, u = heappop(minheap)
            for v, w in graph[u]:
                ntime = time + w
                if ntime < dist[v]:
                    dist[v] = ntime
                    heappush(minheap, (ntime, v))
        return max(dist.values()) if len(dist)==n else -1
```

### Solution 2: Floyd Warshall + O(v^3) + good if dense networks (lots of edges)

```py
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, K: int) -> int:
        dist = [[inf]*n for _ in range(n)]
        for u, v, w in times:
            dist[u-1][v-1] = w
        for i in range(n):
            dist[i][i] = 0
        for k, i, j in product(range(n),repeat=3):
            dist[i][j] = min(dist[i][j], dist[i][k]+dist[k][j])
        return max(dist[K-1]) if max(dist[K-1]) < inf else -1
```

### Solution 3: Bellman ford + SSSP(Single Source Shortest Path) + O(VE) + negative edge weights

```py
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:  
        dist = defaultdict(lambda: inf)
        dist[k] = 0
        for _ in range(n):
            for u, v, w in times:
                dist[v] = min(dist[v], dist[u]+w)
        print(dist)
        return max(dist.values()) if len(dist)== n and max(dist.values()) < inf else -1
```

## Reverse Integer

### Solution 1: string 

```py
class Solution:
    def reverse(self, x: int) -> int:
        sign = [1,-1][x<0]
        rev_x = sign * int(str(abs(x))[::-1])
        return rev_x if rev_x>= -2**31 and rev_x<2**31 else 0
```

## 2269. Find the K-Beauty of a Number

### Solution 1: convert to strings + sliding window

```py
class Solution:
    def divisorSubstrings(self, num: int, k: int) -> int:
        nums = [int(str(num)[i-k:i]) for i in range(k,len(str(num))+1)]
        return sum(1 for n in nums if n!=0 and num%n==0)
```

## 2270. Number of Ways to Split Array

### Solution 1: prefix and suffix sum

```py
class Solution:
    def waysToSplitArray(self, nums: List[int]) -> int:
        psum = 0
        ssum = sum(nums)
        n = len(nums)
        cnt = 0
        for i in range(n-1):
            psum += nums[i]
            ssum -= nums[i]
            cnt += (psum>=ssum)
        return cnt
```

## 2271. Maximum White Tiles Covered by a Carpet

### Solution 1:

```py

```

## 2272. Substring With Largest Variance

### Solution 1: dynamic programming with maximum subarray by converting values to 1 and -1 + reduce search space by considering pair of characters, and convert to 1 and -1, 1 for the maximize one, and -1 for the minimize one. Then do the opposite. 

```py

class Solution:
    def largestVariance(self, s: str) -> int:
        n, var = len(s), 0
        chars = list(set(s))
        def maxSubarray(arr):
            mxSub = rsum = 0
            seen = False
            for x in arr:
                if x < 0: seen = True
                rsum += x
                if seen:
                    mxSub = max(mxSub, rsum)
                else:
                    mxSub = max(mxSub, rsum-1)
                if rsum < 0:
                    rsum = 0
                    seen = False
            return mxSub
        for i in range(len(chars)):
            for j in range(i+1,len(chars)):
                a, b = chars[i], chars[j]
                arr = []
                for ch in s:
                    if ch == a:
                        arr.append(1)
                    elif ch == b:
                        arr.append(-1)
                var = max(var, maxSubarray(arr), maxSubarray([-v for v in arr]))
        return var
```

## 1302. Deepest Leaves Sum

### Solution 1: Tree traversal + sum array

```py
class Solution:
    def deepestLeavesSum(self, root: Optional[TreeNode]) -> int:
        # sum of each level
        sum_arr = [0]
        def preorder(depth, node):
            if not node: return
            if depth == len(sum_arr):
                sum_arr.append(0)
            sum_arr[depth] += node.val
            preorder(depth+1, node.left)
            preorder(depth+1, node.right)
        preorder(0, root)
        return sum_arr[-1]
```

```py
class Solution:
    def deepestLeavesSum(self, root: Optional[TreeNode]) -> int:
        # sum of each level
        sum_arr = [0]
        def getDepth(node):
            if not node: return 0
            return max(getDepth(node.left), getDepth(node.right)) + 1
        self.depth = getDepth(root)
        self.sum = 0
        def deepSum(depth, node):
            if not node: return 0
            self.sum += (node.val if depth==self.depth else 0)
            deepSum(depth+1,node.left)
            deepSum(depth+1,node.right)
        deepSum(1, root)
        return self.sum
```

### Solution 2: Iterative BFS

```py
class Solution:
    def deepestLeavesSum(self, root: Optional[TreeNode]) -> int:
        last_sum = 0
        queue = deque([root])
        while queue:
            sz = len(queue)
            last_sum = 0
            for _ in range(sz):
                node = queue.popleft()
                last_sum += node.val
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return last_sum
```

## 694. Number of Distinct Islands

### Solution 1: hash table for distinct islands + DFS to create the local coordinates + frozenset to make set hashable

```py
class Solution:
    def numDistinctIslands(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        unique_islands_set = set()
        def dfs(row, col):
            if 0<=row<R and 0<=col<C and grid[row][col]==1:
                grid[row][col] = 0
                island_set.add((row-cur_row,col-cur_col))
                for nr, nc in [(row+1,col), (row-1,col), (row,col+1), (row,col-1)]:
                    dfs(nr,nc)
        for r, c in product(range(R), range(C)):
            island_set = set()
            cur_row, cur_col = r, c
            dfs(r,c)
            if island_set:
                unique_islands_set.add(frozenset(island_set))
        return len(unique_islands_set)
```

### Solution 2: hash table + bfs + local coordinates + frozenset

```py
class Solution:
    def numDistinctIslands(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        unique_islands_set = set()
        def bfs(row, col):
            queue = deque([(row,col)])
            grid[row][col] = 0
            in_bounds = lambda r, c: 0<=r<R and 0<=c<C
            while queue:
                r, c = queue.popleft()
                island_set.add((r-cur_row,c-cur_col))
                for nr, nc in [(r+1,c), (r-1,c), (r,c+1), (r,c-1)]:
                    if not in_bounds(nr,nc) or grid[nr][nc] == 0: continue
                    queue.append((nr,nc))
                    grid[nr][nc] = 0
        for r, c in product(range(R), range(C)):
            if grid[r][c] == 1:
                island_set = set()
                cur_row, cur_col = r, c
                bfs(r,c)
                unique_islands_set.add(frozenset(island_set))
        return len(unique_islands_set)
```

### Solution 2: dfs + hash table of tuple of path signature + need to store when backtracking in dfs

```py
class Solution:
    def numDistinctIslands(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        unique_islands_set = set()
        def dfs(r, c, direction='0'):
            if 0<=r<R and 0<=c<C and grid[r][c] == 1:
                grid[r][c] = 0
                path_sig.append(direction)
                for nr, nc, ndirection in [(r+1,c,'D'),(r-1,c,'U'),(r,c+1,'R'),(r,c-1,'L')]:
                    dfs(nr,nc,ndirection)
                path_sig.append('0')
        for r, c in product(range(R), range(C)):
            if grid[r][c] == 1:
                path_sig = []
                dfs(r,c)
                unique_islands_set.add(tuple(path_sig))
        return len(unique_islands_set)
```

## 2273. Find Resultant Array After Removing Anagrams

### Solution 1: stack

```py
class Solution:
    def removeAnagrams(self, words: List[str]) -> List[str]:
        stk = [words[0]]
        for i in range(1,len(words)):
            if Counter(words[i]) == Counter(words[i-1]): continue
            stk.append(words[i])
        return stk
```

## 2274. Maximum Consecutive Floors Without Special Floors

### Solution 1: Sort + iterate

```py
class Solution:
    def maxConsecutive(self, bottom: int, top: int, special: List[int]) -> int:
        special.extend([bottom-1, top + 1])
        special.sort()
        result = 0
        for x, y in zip(special, special[1:]):
            result = max(result, y - x - 1)
        return result
```

## 2275. Largest Combination With Bitwise AND Greater Than Zero

### Solution 1: count maximum bit set across all candidates + 24 bits long

```py
class Solution:
    def largestCombination(self, candidates: List[int]) -> int:
        counts = [0]*24
        for cand in candidates:
            for i in range(24):
                if (cand>>i)&1:
                    counts[i] += 1
        return max(counts)
```

## 2276. Count Integers in Intervals

### Solution 1: binary search + merge intervals

```py
class Node:
    def __init__(self, lo=0, hi=10 ** 9):
        self.lo = lo
        self.hi = hi
        self.mi = (lo + hi) // 2
        self.cnt = 0
        self.left = self.right = None
    
    def add(self, lo, hi):
        if lo > self.hi or hi < self.lo or self.cnt == self.hi - self.lo + 1:
            return
        if lo <= self.lo and hi >= self.hi:
            self.cnt = self.hi - self.lo + 1
        else:
            if self.left is None:
                self.left = Node(self.lo, self.mi)
            self.left.add(lo, hi)
            if self.right is None:
                self.right = Node(self.mi + 1, self.hi)
            self.right.add(lo, hi)
            self.cnt = self.left.cnt + self.right.cnt


class CountIntervals:

    def __init__(self):
        self.root = Node()

    def add(self, left: int, right: int) -> None:
        self.root.add(left, right)

    def count(self) -> int:
        return self.root.cnt
```

```cpp
class CountIntervals {
public:
    set<pair<LL, LL> > se;
    LL W = 0;

    CountIntervals() {
    }
    
    void add(int left_, int right_) {
	LL left = left_;
	LL right = right_ + 1;
	auto it = se.lower_bound(make_pair(left, -1LL));
	while (it != se.end() && it->second <= right) {
	    W -= it->first - it->second;
	    amin(left, it->second);
	    amax(right, it->first);
	    se.erase(it++);
	}
        
	W += right - left;
	se.emplace(right, left);
    }
    
    int count() {
        
	return W;
    }
};

```

## 1091. Shortest Path in Binary Matrix

### Solution 1: BFS + queue

```py
class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        if grid[0][0] == 1: return -1
        n = len(grid)
        queue = deque([(0,0,1)])
        grid[0][0] = 1
        in_bounds = lambda r,c: 0<=r<n and 0<=c<n
        while queue:
            r, c, dist = queue.popleft()
            if r==c==n-1:
                return dist
            for nr,nc in [(r+1,c),(r-1,c),(r,c+1),(r,c-1),(r+1,c+1),(r+1,c-1),(r-1,c+1),(r-1,c-1)]:
                if not in_bounds(nr,nc) or grid[nr][nc]==1: continue
                queue.append((nr,nc,dist+1))
                grid[nr][nc] = 1
        return -1
```

## 1192. Critical Connections in a Network

### Solution 1: Find Articulation Points with DFS in undirected graph

```py

```

## 329. Longest Increasing Path in a Matrix

### Solution 1: sort + memoization

```py
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        R, C = len(matrix), len(matrix[0])
        cells = sorted([(r,c) for r, c in product(range(R), range(C))], key=lambda x: matrix[x[0]][x[1]])
        memo = [[1]*C for _ in range(R)]
        in_bounds = lambda r, c: 0<=r<R and 0<=c<C
        longest_path = 0
        for r, c in cells:
            val = matrix[r][c]
            for nr, nc in [(r+1,c), (r-1,c), (r,c+1), (r,c-1)]:
                if not in_bounds(nr,nc) or matrix[nr][nc] >= val: continue
                memo[r][c] = max(memo[r][c], memo[nr][nc]+1)
            longest_path = max(longest_path, memo[r][c])
        return longest_path
```

### Solution 2: Topological sort and longest path in DAG

```py

```

## 647. Palindromic Substrings

### Solution 1: dynamic programming

```py
class Solution:
    def countSubstrings(self, s: str) -> int:
        cnt, n = 0, len(s)
        def expand(left, right):
            cnt = 0
            while left >= 0 and right < n and s[left]==s[right]:
                cnt += 1
                left -= 1
                right += 1
            return cnt
        for i in range(n):
            cnt += expand(i, i)
            cnt += expand(i, i+1)
        return cnt
```

## 277. Find the Celebrity

### Solution 1: Graph problem 

```py
class Solution:
    def findCelebrity(self, n: int) -> int:
        celebrity = [True]*n 
        for celeb in range(n):
            if not celebrity[celeb]: continue
            for guest in range(n):
                if guest==celeb: continue
                if knows(guest, celeb):
                    celebrity[guest] = False # guest can't be celeb
                else:
                    celebrity[celeb] = False
                    break
                if knows(celeb, guest):
                    celebrity[celeb] = False
                else:
                    celebrity[guest] = False
        try:
            return celebrity.index(True)
        except:
            return -1
```

## 474. Ones and Zeroes

### Solution 1: Dynamic Programming with take or not take + Recursion

```py
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        @cache
        def dfs(i, zeros, ones):
            if i == len(strs) or zeros==ones==0: return 0
            cntOnes = strs[i].count('1')
            cntZeros = len(strs[i]) - cntOnes
            if zeros >= cntZeros and ones >= cntOnes:
                return max(dfs(i+1,zeros,ones), dfs(i+1,zeros-cntZeros,ones-cntOnes)+1)
            return dfs(i+1,zeros,ones)
            
        return dfs(0,m,n)
```

### Solution 2: dynamic programming + knapsack

```py
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        dp = [[0]*(m+1) for _ in range(n+1)]
        for s in strs:
            cntOnes = s.count('1')
            cntZeros = len(s) - cntOnes
            for ones, zeros in product(range(n,cntOnes-1,-1), range(m,cntZeros-1,-1)):
                dp[ones][zeros] = max(dp[ones][zeros], 1+dp[ones-cntOnes][zeros-cntZeros])
        return dp[-1][-1]
```

## 32. Longest Valid Parentheses

### Solution 1: stack 

```py
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        stack = [-1]
        ans = 0
        for i, ch in enumerate(s):
            if ch == '(':
                stack.append(i)
            else:
                stack.pop()
                if not stack:
                    stack.append(i)
                else:
                    ans = max(ans, i - stack[-1])
        return ans
```

### Solution 2: stack + dynamicc programming

```py
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        stack = []
        n = len(s)
        memo = [0]*(n+1)
        for i, ch in enumerate(s):
            if ch == '(':
                stack.append(i)
            elif stack:
                j = stack.pop()
                memo[i+1] = i-j+1+memo[j]
        return max(memo)
```

## Russian Doll Envelopes

### Solution 1: dynamic programming with sort + binary search

```py
class Solution:
    def maxEnvelopes(self, envelopes):
        nums = sorted(envelopes, key = lambda x: (x[0], -x[1]))
        UPPER_BOUND = 100001
        dp = [UPPER_BOUND] * (len(nums) + 1)
        for w, h in nums: 
            i = bisect_left(dp, h)
            dp[i] = h
        return dp.index(UPPER_BOUND)
```

## 268. Missing Number

### Solution 1: bit manipulation with xor

```py
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        answer = len(nums)
        for i, x in enumerate(nums):
            answer = answer ^ i ^ x
        return answer
```

### Solution 2: Gauss formula 

```py
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        n = len(nums)
        expected_sum = n*(n+1)//2
        return expected_sum - sum(nums)
```

## 1059. All Paths from Source Lead to Destination

### Solution 1: BFS + Detect cycle in directed graph with dfs with backtracking algorithm, checking if node is in path. 

```py
class Solution:
    def leadsToDestination(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
        visited = [0]*n
        in_path = [0]*n
        def detect_cycle(node):
            visited[node] = 1
            in_path[node] = 1
            for nei in graph[node]:
                if in_path[nei]: return True
                if not visited[nei] and detect_cycle(nei): return True
            in_path[node] = 0
            return False
        if detect_cycle(source): return False
        visited = [0]*n
        visited[source] = 1
        queue = deque([source])
        is_terminal = lambda x: len(graph[x])==0
        while queue:
            node = queue.popleft()
            if is_terminal(node) and node != destination: return False
            for nei in graph[node]:
                if visited[nei]: continue
                visited[nei] = 1
                queue.append(nei)
        return True
```

## 595. Big Countries

### Solution 1: WHERE clause

```sql
SELECT name, population, area
FROM World
WHERE area >= 3000000
OR population >= 25000000;
```

## 1757. Recyclable and Low Fat Products

### Solution 1: Where clause

```sql
SELECT product_id
FROM Products
WHERE recyclable = 'Y'
AND low_fats = 'Y';
```

## 584. Find Customer Referee

### Solution 1: Where clause and IS NULL

```sql
SELECT name
FROM Customer
WHERE referee_id IS NULL
OR referee_id != 2;
```

## 183. Customers Who Never Order

### Solution 1: NOT IN clause

```sql
SELECT name AS Customers
FROM Customers
WHERE id NOT IN (SELECT customerId FROM Orders);
```

## 2286. Booking Concert Tickets in Groups

### Solution 1: fenwick tree for range sum queries and segment tree for range max query

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
class MaxSegmentTree:
    def __init__(self,arr):
        self.arr = arr
        n = len(arr)
        self.neutral = -inf
        self.size = 1
        while self.size<n:
            self.size*=2
        self.tree = [0 for _ in range(self.size*2)]
    def update_tree(self, idx, val):
        self.update(idx,val,0,0,self.size)
    def build_tree(self):
        for i, val in enumerate(self.arr):
            self.update_tree(i,val)
    def update(self,idx,val,x,lx,rx):
        if rx-lx==1:
            self.tree[x] = val
            return
        mid = rx+lx>>1
        if idx<mid:
            self.update(idx,val,2*x+1,lx,mid)
        else:
            self.update(idx,val,2*x+2,mid,rx)
        self.tree[x] = max(self.tree[2*x+1],self.tree[2*x+2])
    def query(self, l, r, x, lx, rx):
        if lx>=r or l>=rx:
            return self.neutral
        if lx>=l and rx<=r:
            return self.tree[x]
        m = lx+rx>>1
        sl = self.query(l,r,2*x+1,lx,m)
        sr = self.query(l,r,2*x+2,m,rx)
        return max(sl,sr)
    def query_tree(self, l, r):
        return self.query(l,r,0,0,self.size)
    def get_first_tree(self,l, r,val):
        return self.get_first(l,r,0,0,self.size,val)
    def get_first(self,l,r,x,lx,rx,val):
        if lx>=r or rx<=l: return -1
        if l<=lx and rx<=r:
            if self.tree[x] < val: return -1
            while rx != lx+1:
                mid = lx + rx >> 1
                if self.tree[2*x+1]>=val:
                    x = 2*x+1
                    rx = mid
                else:
                    x = 2*x+2
                    lx = mid

            return lx
        mid = lx+rx>>1
        left_segment = self.get_first(l,r,2*x+1,lx,mid,val)
        if left_segment != -1: return left_segment
        return self.get_first(l,r,2*x+2,mid,rx,val)
    def __repr__(self):
        return f"array: {self.tree}"

class BookMyShow:

    def __init__(self, n: int, m: int):
        self.seats = [m]*n # cnt of empty seats
        self.row_len = m
        self.fenwick = FenwickTree(n)
        for i in range(1,n+1):
            self.fenwick.update(i,m)
        self.maxSegTree = MaxSegmentTree(self.seats)
        self.maxSegTree.build_tree()
        self.cur_row = 0
    def gather(self, k: int, maxRow: int) -> List[int]:
        r = self.maxSegTree.get_first_tree(0,maxRow+1,k)
        if r < 0: return []
        empty_seats = self.seats[r]
        c = self.row_len - empty_seats
        self.seats[r] -= k # update the empty seats in the row
        self.fenwick.update(r+1,-k)
        self.maxSegTree.update_tree(r,self.seats[r])
        return [r,c]

    def scatter(self, k: int, maxRow: int) -> bool:
        if self.fenwick.query(maxRow+1) < k: return False
        for r in range(self.cur_row, maxRow+1):
            fill_seats = min(k, self.seats[r])
            self.seats[r] -= fill_seats
            k -= fill_seats
            self.fenwick.update(r+1,-fill_seats)
            self.maxSegTree.update_tree(r,self.seats[r])
            if self.seats[r] == 0:
                self.cur_row = r + 1
            if k == 0: break
        return True
```

## 2283. Check if Number Has Equal Digit Count and Digit Value

### Solution 1: counter

```py
class Solution:
    def digitCount(self, num: str) -> bool:
        counts = Counter(map(int, num))
        return not any(counts[digit]!=cnt for digit, cnt in enumerate(map(int,num)))
```

## 2284. Sender With Largest Word Count

### Solution 1: counter + sorting

```py
class Solution:
    def largestWordCount(self, messages: List[str], senders: List[str]) -> str:
        sender_counter = Counter()
        for message, sender in zip(messages, senders):
            sender_counter[sender] += message.count(' ') + 1
        pairs = [(cnt , sender) for sender, cnt in sender_counter.items()]
        pairs.sort(reverse=True)
        return pairs[0][1]
```

## 2285. Maximum Total Importance of Roads

### Solution 1: count indegrees of undirected graph

```py
class Solution:
    def maximumImportance(self, n: int, roads: List[List[int]]) -> int:
        graph = defaultdict(list)
        for u, v in roads:
            graph[u].append(v)
            graph[v].append(u)
        num_nei = sorted([len(edges) for edges in graph.values()], reverse=True)
        return sum(i*cnt for i, cnt in zip(range(n+1)[::-1], num_nei))
```

```py
class Solution:
    def maximumImportance(self, n: int, roads: List[List[int]]) -> int:
        indegrees = [0]*n
        for u, v in roads:
            indegrees[u]+=1
            indegrees[v]+=1
        indegrees.sort()
        return sum(node*cnt_indegrees for node, cnt_indegrees in enumerate(indegrees, start=1))
```

## 318. Maximum Product of Word Lengths

### Solution 1: Counter + bitmask + bit manipulation

```py
class Solution:
    def maxProduct(self, words: List[str]) -> int:
        bitmask_dict = Counter()
        for word in words:
            bitmask = 0
            for ch in word:
                right_shift_index = ord(ch)-ord('a')
                bitmask |= (1<<right_shift_index)
            bitmask_dict[bitmask] = max(bitmask_dict[bitmask], len(word))
        max_prod = 0
        for (b1,l1), (b2,l2) in product(bitmask_dict.items(), repeat=2):
            if b1 & b2 == 0:
                max_prod = max(max_prod, l1*l2)
        return max_prod
```

```py

```

## 1136. Parallel Courses

### Solution 1: Topological sort + Use Topological sort to detect cycle + BFS Implementation + Called Kahn's Algorithm

```py
class Solution:
    def minimumSemesters(self, n: int, relations: List[List[int]]) -> int:
        visited = [0]*(n+1)
        graph = defaultdict(list)
        indegrees = [0]*(n+1)
        for u, v in relations:
            graph[u].append(v)
            indegrees[v] += 1
        num_semesters = studied_count = 0
        queue = deque()
        for node in range(1,n+1):
            if indegrees[node] == 0:
                queue.append(node)
                studied_count += 1
        while queue:
            num_semesters += 1
            sz = len(queue)
            for _ in range(sz):
                node = queue.popleft()
                for nei in graph[node]:
                    indegrees[nei] -= 1
                    if indegrees[nei] == 0 and not visited[nei]:
                        queue.append(nei)
                        studied_count += 1
        return num_semesters if studied_count == n else -1
```

## 2289. Steps to Make Array Non-decreasing

### Solution 1: stack + dynamic programming

![edge case 1](images/make_array_nondecreasing1.PNG)
![edge case 2](images/make_array_nondecreasing2.PNG)

```py
class Solution:
    def totalSteps(self, nums: List[int]) -> int:
        stack = []
        total_steps = 0
        for num in nums:
            steps = 0
            while stack and num >= stack[-1][0]:
                steps = max(steps, stack[-1][1])
                stack.pop()
            steps = steps + 1 if stack else 0
            total_steps = max(total_steps, steps)
            stack.append((num,steps))
        return total_steps
```

## 29. Divide Two Integers

### Solution 1: Repeated Exponential Search 

```py
class Solution:
    def divide(self, dividend: int, divisor: int) -> int:
        LOWER, UPPER = -2**31, 2**31-1  
        sign = 1 if (divisor>0)^(dividend>0) else -1
        # SPECIAL CASE
        if dividend == LOWER and divisor == -1: return UPPER
        quotient = 0
        dividend, divisor = -abs(dividend), -abs(divisor)
        while dividend <= divisor:
            value = divisor
            num_times = -1
            while value >= LOWER//2 and value+value >= dividend:
                value+=value
                num_times += num_times
            quotient += num_times
            dividend -= value
        return sign*quotient
```

### Solution 2: Single Exponential Search + memoization + linear scan in reverse through potential values

```py
class Solution:
    def divide(self, dividend: int, divisor: int) -> int:
        LOWER, UPPER = -2**31, 2**31-1  
        sign = 1 if (divisor>0)^(dividend>0) else -1
        # SPECIAL CASE
        if dividend == LOWER and divisor == -1: return UPPER
        quotient = 0
        dividend, divisor = -abs(dividend), -abs(divisor)
        powers_arr, values_arr = [], []
        values_arr, powers_arr = [divisor], [-1]
        while values_arr[-1] >= LOWER//2 and values_arr[-1]+values_arr[-1] >= dividend:
            values_arr.append(values_arr[-1]+values_arr[-1])
            powers_arr.append(powers_arr[-1]+powers_arr[-1])
        for val, pow_ in zip(reversed(values_arr), reversed(powers_arr)):
            if val >= dividend:
                quotient += pow_
                dividend -= val
        return sign*quotient
```

## 2287. Rearrange Characters to Make Target String

### Solution 1: Counter + min 

```py
class Solution:
    def rearrangeCharacters(self, s: str, target: str) -> int:
        counts, tcounts = map(Counter, (s, target))
        return min(counts[c]//tcounts[c] for c in tcounts)
```

## 2288. Apply Discount to Prices

### Solution 1: isnumeric() + string + floats

```py
class Solution:
    def discountPrices(self, sentence: str, discount: int) -> str:
        delta = (100-discount)/100
        words = sentence.split()
        for i, word in enumerate(words):
            if word[0] == '$' and word[1:].isnumeric():
                updated_price = int(word[1:])*delta
                words[i] = f'${updated_price:.2f}'
        return ' '.join(words)
```

## 627. Swap Salary

### Solution 1: UPDATE Query with CASE statement

```sql
UPDATE Salary
SET 
    sex =  CASE 
        WHEN sex = 'f' THEN 'm'
        ELSE 'f'
    END;
```

## 196. Delete Duplicate Emails

### Solution 1: DELETE FROM WITH IMPLICT JOIN QUERY USING WHERE

```sql
DELETE p1 FROM Person p1, Person p2
WHERE p1.email = p2.email 
AND p1.id > p2.id;
```

## 1873. Calculate Special Bonus

### Solution 1: SELECT + CASE + SUBSTRING

```sql
SELECT employee_id, 
    CASE 
        WHEN employee_id%2 = 0 OR SUBSTRING(name, 1, 1) = 'M' THEN 0
        ELSE salary
    END bonus
FROM Employees;
```

### Solution 2: SELECT + CASE + LIKE for prefix

```sql
SELECT employee_id, 
    CASE 
        WHEN employee_id%2 != 0 AND name NOT LIKE 'M%' THEN salary
        ELSE 0
    END bonus
FROM Employees;
```

## 2290. Minimum Obstacle Removal to Reach Corner

### Solution 1: Dijkstra Algorithm + minheap datastructure + shortest path

```py
class Solution:
    def minimumObstacles(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        minheap = [(0,0,0)] # (num_obstacles_removed, row, col)
        # goal is to minimize on the num_obstacles_removed
        dist = [[inf]*C for _ in range(R)]
        in_bounds = lambda r, c: 0<=r<R and 0<=c<C
        def neighbors(row, col):
            for nr, nc in [(row-1,col),(row+1,col),(row,col-1),(row,col+1)]:
                if not in_bounds(nr,nc): continue
                yield nr, nc
        dist[0][0] = 0
        while minheap:
            cost, row, col = heappop(minheap)
            if row == R-1 and col == C-1:
                return cost
            for nr, nc in neighbors(row,col):
                ncost = cost + grid[nr][nc]
                if ncost < dist[nr][nc]:
                    dist[nr][nc] = ncost
                    heappush(minheap, (ncost,nr,nc))
        return -1
```

### Solution 2: 0/1 BFS + Modified BFS

```py
class Solution:
    def minimumObstacles(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        queue = deque([(0,0,0)]) # (num_obstacles_removed, row, col)
        # goal is to minimize on the num_obstacles_removed
        dist = [[inf]*C for _ in range(R)]
        in_bounds = lambda r, c: 0<=r<R and 0<=c<C
        def neighbors(row, col):
            for nr, nc in [(row-1,col),(row+1,col),(row,col-1),(row,col+1)]:
                if not in_bounds(nr,nc) or dist[nr][nc] < inf: continue
                yield nr, nc
        dist[0][0] = 0
        while queue:
            cost, row, col = queue.popleft()
            if row == R-1 and col == C-1:
                return cost
            for nr, nc in neighbors(row,col):
                if grid[nr][nc] == 1:
                    dist[nr][nc] = cost + 1
                    queue.append((cost+1,nr,nc))
                else:
                    dist[nr][nc] = cost
                    queue.appendleft((cost, nr,nc))
        return -1
```

## 1667. Fix Names in a Table

### Solution 1: CONCAT + UPPER/LOWER + SUBSTRING

```sql
SELECT user_id, CONCAT(UPPER(SUBSTRING(name, 1, 1)),LOWER(SUBSTRING(name, 2))) AS name
FROM Users
ORDER BY user_id
```

## 1484. Group Sold Products By The Date

### Solution 1: GROUPBY + ORDERBY + GROUP_CONCAT create a list from a column

```sql
SELECT sell_date, COUNT(DISTINCT(product)) AS num_sold, GROUP_CONCAT(DISTINCT product ORDER BY product ASC) AS products
FROM Activities
GROUP BY sell_date
ORDER BY sell_date 
```

## 1527. Patients With a Condition

### Solution 1: WHERE WITH LIKE for prefix search + search for prefix in string

```sql
SELECT *
FROM Patients
WHERE conditions LIKE '% DIAB1%'
OR conditions LIKE 'DIAB1%'
```

## 

### Solution 1: 

```sql

```

## 

### Solution 1: 

```sql

```

## 

### Solution 1: 

```sql

```

## 

### Solution 1: 

```sql

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