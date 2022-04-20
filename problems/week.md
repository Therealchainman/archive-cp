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


