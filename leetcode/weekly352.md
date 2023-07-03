# Leetcode Weekly Contest 352

## 2760. Longest Even Odd Subarray With Threshold

### Solution 1:  sliding window

```py
class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        left = res = 0
        while left < n:
            while left < n and nums[left] & 1:
                left += 1
            if left < n and nums[left] > threshold:
                left += 1
                continue
            right = left
            while right + 1 < n and nums[right] % 2 != nums[right + 1] % 2 and nums[right + 1] <= threshold:
                right += 1
            if left < n:
                res = max(res, right - left + 1)
            left = right + 1
        return res
```

## 2761. Prime Pairs With Target Sum

### Solution 1: 

```py
def prime_sieve(lim):
    sieve,primes = [[] for _ in range(lim)], []
    integer = 2
    if integer < lim:
        primes.append(integer)
        for possibly_divisible_integer in range(integer,lim,integer):
            current_integer = possibly_divisible_integer
            while not current_integer%integer:
                sieve[possibly_divisible_integer].append(integer)
                current_integer //= integer
    for integer in range(3,lim, 2):
        if not len(sieve[integer]):
            primes.append(integer)
            for possibly_divisible_integer in range(integer,lim,integer):
                current_integer = possibly_divisible_integer
                while not current_integer%integer:
                    sieve[possibly_divisible_integer].append(integer)
                    current_integer //= integer
    return primes

def is_prime(x: int) -> bool:
    if x < 2: return False
    if x == 2: return True
    if x % 2 == 0: return False
    for i in range(3, int(math.sqrt(x)) + 1, 2):
        if x % i == 0: return False
    return True

class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        primes = prime_sieve(n // 2 + 1)
        res = []
        for x in primes:
            y = n - x
            if x > y: break
            if not is_prime(y): continue
            res.append([x, y])
        return res
```

```py
class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        res = []
        memo = {}
        def is_prime(x: int) -> bool:
            if x in memo: return memo[x]
            if x < 2: return False
            if x == 2: return True
            if x % 2 == 0: return False
            for i in range(3, int(math.sqrt(x)) + 1, 2):
                if x % i == 0: 
                    memo[x] = False
                    return memo[x]
            memo[x] = True
            return memo[x]
        if n >= 4:
            if is_prime(n - 2): res.append([2, n - 2])
        for x in range(3, n // 2 + 1, 2):
            y = n - x
            if not is_prime(x) or not is_prime(y): continue
            res.append([x, y])
        return res
```

```cpp
class Solution {
public:
    vector<vector<int>> findPrimePairs(int n) {
        vector<std::vector<int>> res;
        auto is_prime = [&](int x) {
            
            if (x < 2)
                return false;
            
            if (x == 2)
                return true;
            
            if (x % 2 == 0)
                return false;
            
            for (int i = 3; i <= std::sqrt(x); i += 2) {
                if (x % i == 0) {
                    return false;
                }
            }
            return true;
        };
        
        if (n >= 4) {
            if (is_prime(n - 2))
                res.push_back({2, n - 2});
        }
        
        for (int x = 3; x <= n / 2; x += 2) {
            int y = n - x;
            if (!is_prime(x) || !is_prime(y))
                continue;
            
            res.push_back({x, y});
        }
        
        return res;
    }
};
```

## 2762. Continuous Subarrays

### Solution 1: sliding window + monotonic deque

```py
class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        res = left = 0
        min_stack, max_stack = deque(), deque()
        for right in range(n):
            while min_stack and nums[right] <= nums[min_stack[-1]]:
                min_stack.pop()
            while max_stack and nums[right] >= nums[max_stack[-1]]:
                max_stack.pop()
            min_stack.append(right)
            max_stack.append(right)
            while abs(nums[max_stack[0]] - nums[min_stack[0]]) > 2:
                left += 1
                while min_stack and min_stack[0] < left:
                    min_stack.popleft()
                while max_stack and max_stack[0] < left:
                    max_stack.popleft()
            delta = right - left + 1
            res += delta
        return res
```

## 2763. Sum of Imbalance Numbers of All Subarrays

### Solution 1: 

```py
class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        res = 0
        n = len(nums)
        for i in range(n):
            seen = [0] * (n + 2)
            seen[nums[i]] = 1
            min_val = max_val = nums[i]
            imbalance = 0
            for j in range(i + 1, n):
                prv, val, nxt = nums[j] - 1, nums[j], nums[j] + 1
                if nums[j] > max_val and not seen[prv]: imbalance += 1
                if nums[j] < min_val and not seen[nxt]: imbalance += 1
                if min_val < val < max_val and seen[prv] and seen[nxt] and not seen[val]: imbalance -= 1
                if min_val < val < max_val and not seen[prv] and not seen[nxt] and not seen[val]: imbalance += 1
                min_val = min(min_val, nums[j])
                max_val = max(max_val, nums[j])
                seen[val] = 1
                res += imbalance
        return res
```