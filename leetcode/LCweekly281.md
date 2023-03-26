# Leetcode Weekly Contest 281

## 2180. Count Integers With Even Digit Sum

### Solution: Convert to string and sum the digits and check if it is even

```py
class Solution:
    def countEven(self, num: int) -> int:
        return sum(1 for x in range(2,num+1) if sum(map(int,str(x)))%2==0)
```

## 2181. Merge Nodes in Between Zeros

### Solution: Two pointer, slow and fast

```py
class Solution:
    def mergeNodes(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow, fast = head, head.next
        sum_ = 0
        while fast:
            sum_ += fast.val
            if fast.val == 0:
                slow.next = ListNode(sum_)
                slow = slow.next
                sum_ = 0
            fast = fast.next
        return head.next
```

## 2182. Construct String With Repeat Limit

### Solution: max Heap datastructure with constraints applied 

```py
from heapq import heappop, heappush
class Solution:
    def repeatLimitedString(self, s: str, repeatLimit: int) -> str:
        Character = namedtuple('Character', ['char', 'count'])
        Character.__lt__ = lambda self, other: self.char > other.char
        freq = Counter(s)
        heap = [Character(ch, cnt) for ch, cnt in freq.items()]
        heapify(heap)
        res = []
        while len(heap) > 0:
            ch = heappop(heap)
            upper_bound = min(repeatLimit, ch.count)
            for _ in range(upper_bound):
                res.append(ch.char)
            if upper_bound == ch.count: continue
            if len(heap) == 0: break
            replacech = heappop(heap)
            res.append(replacech.char)
            if replacech.count > 1:
                heappush(heap, Character(replacech.char, replacech.count - 1))
            heappush(heap, Character(ch.char, ch.count - upper_bound))
        return "".join(res)
```

### Solution: Simply construct string from descending order and look at next whenever reach repeatLimit

```py

```

## 2183. Count Array Pairs Divisible by K

### Solution: Factorization + hashmap

Find the prime factorization for K, and every integer in nums. 
Use that to find the difference between both prime factorizations.
The difference will be the needed prime factors. We can create
a power set from that or every possible set of those prime factors, or 
all subsets, and get the product of each one and increase the counter by that

So then all we need is to use the counter and find how many times
we've seen this integer previously.

This solution is valid but it TLE, it needs to be optimized to pass on 
Leetcode

```py
from math import sqrt
from numpy import prod
class Solution:
    def prime_factors(self, num):
        res = []
        i = 2
        while num > 1: 
            while num%i==0:
                num//=i
                res.append(i)
            i += 1
        return res
    
    # COMPUTES THE FACTORS WE NEED TO MAKE CURRENT NUMBER NUMS2 DIVISIBLE BY K
    def difference(self, nums1, nums2):
        res = []
        i, j = 0, 0
        while i < len(nums1) and j < len(nums2):
            if nums1[i] < nums2[j]:
                res.append(nums1[i])
                i += 1
            elif nums1[i] > nums2[j]:
                j += 1
            else:
                i += 1
                j += 1
        while i < len(nums1):
            res.append(nums1[i])
            i += 1
        return res
    
    def countPairs(self, nums: List[int], k: int) -> int:
        pk = self.prime_factors(k)
        counter = Counter()
        cnt = 0

        for num in nums:
            nk = self.prime_factors(num)
            needed_factors = self.difference(pk,nk)

            # COUNTER OF TIMES THIS PROD HAS BEEN SEEN
            cnt += counter[prod(needed_factors)]
            cset = []
            # POWER SET
            def subset(start):
                counter[prod(cset)] += 1
                if start == len(nk): return
                for i in range(start, len(nk)):
                    if i > start and nk[i] == nk[i-1]: continue
                    cset.append(nk[i])
                    subset(i+1)
                    cset.pop()
            
            # UPDATING COUNTER
            subset(0)
        return cnt
```

### Solution: GCD + Hashmap

Since we could have at most 128 factors for k possibly, we can say counter will have at most 128 elements
so it is N*N, where N = number of factors of k.  But that is really low, so it works

if
a*b%k==0 then
gcd(a,k)*gcd(b,k)%k==0 

GCD is O(log(min(a,b))) if it uses euclidean algorithm

so we have O(N*log(100000 + N*K),  where N = len(nums), K = all possible gcd

```py
from math import gcd
class Solution:
    def countPairs(self, nums: List[int], k: int) -> int:
        counter = Counter(gcd(num, k) for num in nums)
        cnt = 0
        for a in counter:
            for b in counter:
                if a <= b and a*b%k == 0:
                    cnt += counter[a] * counter[b] if a < b else counter[a] * (counter[a] - 1) // 2
        return cnt
```



```py
from math import gcd
class Solution:
    def countPairs(self, nums: List[int], k: int) -> int:
        counter = Counter()
        res = 0
        for num in nums:
            curgcd = gcd(num, k)
            for g, cnt in counter.items():
                if g*curgcd%k==0:
                    res += cnt
            counter[curgcd] += 1
        return res
```