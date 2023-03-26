# Leetcode Weekly Contest 283

## 2194. Cells in a Range on an Excel Sheet

### Solution: Iterate through strings

```py
class Solution:
    def cellsInRange(self, s: str) -> List[str]:
        start, end = s.split(':')
        res = []
        for col in range(ord(start[0]), ord(end[0])+1):
            for row in range(ord(start[1]), ord(end[1])+1):
                res.append(chr(col)+chr(row))
        return res
```

## 2195. Append K Integers With Minimal Sum

### Solution: Compute arithmetic series + diff

```py
class Solution:
    def minimalKSum(self, nums: List[int], k: int) -> int:
        def arith_sum(n, a):
            return n*(2*a+n-1)//2
        nums.append(0)
        nums.append(10**10)
        nums = sorted(list(set(nums)))
        res = 0
        for i in range(1,len(nums)):
            diff = nums[i]-nums[i-1]-1
            delta = min(k,diff)
            k-= delta
            res += arith_sum(delta, nums[i-1]+1)
            if k==0: break
        res += arith_sum(k, nums[-1]+1)
        return res
```

### Solution: binary search 

```py
class Solution:
    def minimalKSum(self, nums: List[int], k: int) -> int:
        nums = set(nums)
        lo, hi = 1, 10**10
        while lo<hi:
            mid = (lo+hi)>>1
            if mid - sum(i<=mid for i in nums) >= k:
                hi = mid
            else:
                lo = mid+1
        return lo*(lo+1)//2 - sum(i for i in nums if i<=lo)
```

## 2196. Create Binary Tree From Descriptions

### Solution: Hashmap + construct tree online

```py
class Solution:
    def createBinaryTree(self, descriptions: List[List[int]]) -> Optional[TreeNode]:
        nodes = {}
        children = set()
        for parent, child, is_left in descriptions:
            pnode = nodes.setdefault(parent, TreeNode(parent))
            cnode = nodes.setdefault(child, TreeNode(child))
            children.add(child)
            if is_left:
                pnode.left = cnode
            else:
                pnode.right = cnode
        for key, node in nodes.items():
            if key not in children:
                return node
        print("Error the description does not contain a valid tree with a root node")
        return -1
```

## 2197. Replace Non-Coprime Numbers in Array

### Solution: stack + gcd + lcm 

This must be a weak spot for me, cause I just thought use
doubly linked list when you need to delete elements, but I forget
you can use a stack. 

```py
from math import gcd, lcm
class Solution:
    def replaceNonCoprimes(self, nums: List[int]) -> List[int]:
        stack = []
        for num in nums:
            while stack and gcd(stack[-1], num) > 1:
                num = lcm(stack[-1],num)
                stack.pop()
            stack.append(num)
        return stack
```