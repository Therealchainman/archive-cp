# Leetcode Weekly Contest 351

## 2748. Number of Beautiful Pairs

### Solution 1:  gcd + nested loops

```py
class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        n = len(nums)
        get = lambda i, loc: int(str(nums[i])[loc])
        return sum(1 for i in range(n) for j in range(i + 1, n) if math.gcd(get(i, 0), get(j, -1)) == 1)
```

## 2749. Minimum Operations to Make the Integer Zero

### Solution 1:  bit manipulation + math

determine if can form by sum of powers of two with i terms. 

So the lower bound is you need at least the number of 1s in the binary representation of v.  Right cause if you have 1010, you need at least 2 power 2, 2^3 + 2^1, 
The upper bound is you need at least v power of twos, cause you can just do 2^0+2^0 which is same as 1 + 1 + 1, so you can do that in v terms.

But what is tricky is why can you do anything in between, and the reason is that you can always split something into two, that is 2^3 = 2^2+2^2, and in this way you can get to 2^0+2^0+...+2^0, by increasing number of power of twos that sum to 10 all the way until you reach upper bound, by incrementing by one each time, by breaking a 2^x into 2^(x - 1) + 2^(x - 1)

```py
class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        for i in range(61):
            v = num1 - i * num2
            if v.bit_count() <= i <= v: return i
        return -1
```

## 2750. Ways to Split Array Into Good Subarrays

### Solution 1:  combinatorics + counting

given some array
0011001001, you can partition any sandwiched 0 segment, which means it lies between 1s. 
And the possible ways to split it is the number of 0s + 1, 
cause for instance
0011001001, you can split that first 0 segment these ways
0011|001001
00110|01001
001100|1001

multiply all the number of partition for each zero segment

```py
class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        n = len(nums)
        if sum(nums) == 0: return 0
        for left in range(n):
            if nums[left] == 1: break
        for right in reversed(range(n)):
            if nums[right] == 1: break
        nums = nums[left: right + 1]
        res = 1
        mod = 10 ** 9 + 7
        for key, grp in groupby(nums):
            if key == 0:
                res *= len(list(grp)) + 1
                res %= mod
        return res
```

## 2751. Robot Collisions

### Solution 1:  stack + greedy

```py
class Solution:
    def survivedRobotsHealths(self, positions: List[int], healths: List[int], directions: str) -> List[int]:
        n = len(positions)
        robots = sorted(range(n), key = lambda i: positions[i])
        stack = []
        for i in robots:
            if directions[i] == 'L':
                while stack and directions[stack[-1]] == 'R' and healths[i] > 0:
                    idx = stack.pop()
                    if healths[idx] < healths[i]:
                        healths[i] -= 1
                        healths[idx] = 0
                    elif healths[idx] == healths[i]:
                        healths[i] = healths[idx] = 0
                    elif healths[idx] > healths[i]:
                        healths[i] = 0
                        healths[idx] -= 1
                    if healths[idx] > 0:
                        stack.append(idx)
            if healths[i] > 0:
                stack.append(i)
        return filter(lambda x: x > 0, healths)
```