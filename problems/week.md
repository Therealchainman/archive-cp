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

## 

### Solution 1:

```py

```

## 

### Solution 1:

```py

```