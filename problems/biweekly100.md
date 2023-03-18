# Leetcode Biweekly Contest 100

## 2591. Distribute Money to Maximum Children

### Solution 1:  math + division theory

d = nq + r, where money = 7n + r in this case. 

give all children 1 dollar, if not enough return -1 else use this and try to give 7 more to get it to 8 dollars for everyone. 

sometimes there is extra giving, was able to give to more children than need to, then you can put that into the remainder.  And basically if the remainder is greater than 0 that means you cannot give to more than children - 1, cause the last child will need to push all that extra remainder onto. 

But there is one more special case which is when the remainder = 3, then this only applies when technically you have n = children - 1, cause then you can't give 3 to the last child but need to split it between two so that means you will need to subtract 1 more from children.

```py
class Solution:
    def distMoney(self, money: int, children: int) -> int:
        money -= children
        if money < 0: return -1
        n, r = divmod(money, 7)
        extra_giving = max(0, n - children)
        r += 7*extra_giving
        return min(children - (r > 0) - (n < children and r == 3), n)
```

## 2592. Maximize Greatness of an Array

### Solution 1:  sort + two pointers

sort array and use left to point to cur smallest element that has not have found another element in the array that is greater than it.  Once that happens increment left and left pointer will be the number of elements that were able to have a permutated element from same array that is greater than it. 

```py
class Solution:
    def maximizeGreatness(self, nums: List[int]) -> int:
        nums.sort()
        left = 0
        for right in range(len(nums)):
            if nums[right] > nums[left]:
                left += 1
        return left
```

## 2593. Find Score of an Array After Marking All Elements

### Solution 1:  sort + hash table

just have to track the marked elements, and do it in sorted order from smallest to largest. 

```py
class Solution:
    def findScore(self, nums: List[int]) -> int:
        score = 0
        marked = [0]*(len(nums) + 1)
        for i, num in sorted(enumerate(nums), key = lambda pair: pair[1]):
            if marked[i]: continue
            marked[i] = marked[i - 1] = marked[i + 1] = 1
            score += num
        return score
```

## 2594. Minimum Time to Repair Cars

### Solution 1:  greedy binary search 

This became way faster after adding the Counter cause now it puts ranks into buckets, so decreases the size of iteration sometimes, obviously worse case is when every single mechanic has a different rank.

```py
class Solution:
    def repairCars(self, ranks: List[int], cars: int) -> int:
        f = lambda time, rank: math.isqrt(time//rank)
        counts = Counter(ranks)
        left, right = 0, min(counts)*cars*cars
        def possible(time):
            repaired = 0
            for rank, cnt in counts.items():
                repaired += cnt*f(time, rank)
                if repaired >= cars: return True
            return False
        while left < right:
            mid = (left + right) >> 1
            if possible(mid):
                right = mid
            else:
                left = mid + 1
        return left
```

### Solution 2:  binary search with bisect left + custom comparator 

This will check for each time in range of 0 to min(counts)*cars*cars cand check based on the key which will iterate through all the counts of ranks and find the total number of cars repaired and compared it to the cars variable, 
so if cars <= repaired_cars it will return True

```py
class Solution:
    def repairCars(self, ranks: List[int], cars: int) -> int:
        f = lambda time, rank: math.isqrt(time//rank)
        counts = Counter(ranks)
        return bisect.bisect_left(range(0, min(counts)*cars*cars), cars, key = lambda time: sum(cnt*f(time, rank) for rank, cnt in counts.items()))
```

This one is a little more clear, we are going to say when repaired cars is greater than equal to cars then return True
I want the first instance of True, 

cause we will have something like this, monotonic array of repaired cars as time increases

[1,1,2,2,3,3,4,5,6,7,8,9]
FFFFTTTTT

first T is when the first instance that repaired cars is greater than or equal to cars, so can repair all cars within time with these mechanics

```py
class Solution:
    def repairCars(self, ranks: List[int], cars: int) -> int:
        f = lambda time, rank: math.isqrt(time//rank)
        counts = Counter(ranks)
        return bisect.bisect_left(range(0, min(counts)*cars*cars), True, key = lambda time: sum(cnt*f(time, rank) for rank, cnt in counts.items()) >= cars)
```