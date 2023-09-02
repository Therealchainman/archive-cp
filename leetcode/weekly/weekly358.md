# Leetcode Weekly Contest 358

## 2815. Max Pair Sum in an Array

### Solution 1:  defaultdict + sort 

the hashed value for each list is the maximum digit in that num, so they are all grouped together, and then just take the two largest from each group.

```py
class Solution:
    def maxSum(self, nums: List[int]) -> int:
        n = len(nums)
        d = defaultdict(list)
        for num in nums:
            dig = max(map(int, str(num)))
            d[dig].append(num)
        res = -1
        for vals in d.values():
            if len(vals) == 1: continue
            vals.sort(reverse=True)
            res = max(res, vals[0] + vals[1])
        return res
```

## 2816. Double a Number Represented as a Linked List

### Solution 1:  linked lists + reverse

```py
class Solution:
    def doubleIt(self, head: Optional[ListNode]) -> Optional[ListNode]:
        digs = []
        cur = head
        while cur:
            digs.append(cur.val)
            cur = cur.next
        carry = 0
        digs.reverse()
        res = []
        for dig in digs:
            cur = dig * 2 + carry
            res.append(cur % 10)
            carry = cur // 10
        if carry:
            res.append(carry)
        head = ListNode(res.pop())
        cur = head
        while res:
            cur.next = ListNode(res.pop())
            cur = cur.next
        return head
```

## 2817. Minimum Absolute Difference Between Elements With Constraint

### Solution 1:  sortedlist + binary search

```py
from sortedcontainers import SortedList

class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        n = len(nums)
        sl = SortedList()
        res = math.inf
        for i in range(x, n):
            sl.add(nums[i-x])
            j = sl.bisect_right(nums[i])
            if j < len(sl):
                res = min(res, abs(nums[i] - sl[j]))
            if j > 0:
                res = min(res, abs(nums[i] - sl[j - 1]))
        return res
```

## 2818. Apply Operations to Maximize Score

### Solution 1:  monotonic stack + prime factorization + prime count + sort + offline queries + pow

prime_count function counts the number of prime factors for any given num. 



```py
def prime_count(num):
    cnt = 0
    i = 2
    while i * i <= num:
        cnt += num % i == 0
        while num % i == 0:
            num //= i
        i += 1
    cnt += num > 1
    return cnt

class Solution:
    def maximumScore(self, nums: List[int], k: int) -> int:
        mod = int(1e9) + 7
        n = len(nums)
        pscores = list(map(lambda x: prime_count(x), nums))
        left, right = [0] * n, [0] * n
        # forward pass to compute the greater right elements
        # find how much forward from an element it can go and be the largest
        stk = []
        for i, p in enumerate(pscores + [math.inf]):
            while stk and pscores[stk[-1]] < p:
                j = stk.pop()
                right[j] = i - 1
            stk.append(i)
        # backward pass to compute the lesser left elements
        # find howmuch back from an element it can go and be the largest
        stk = []
        for i, p in zip(range(n - 1, -2, -1), reversed([math.inf] + pscores)):
            if stk:
                index = stk[-1]
            while stk and pscores[stk[-1]] <= p:
                j = stk.pop()
                left[j] = index
            stk.append(i)
        queries = sorted([(num, i) for i, num in enumerate(nums)], reverse = True)
        res = 1
        for num, i in queries:
            left_, right_ = i - left[i] + 1, right[i] - i + 1
            t = min(left_ * right_, k)
            res = (res * pow(num, t, mod)) % mod
            k -= t
            if k == 0: break
        return res
```

