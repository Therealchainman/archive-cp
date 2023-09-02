# Leetcode Biweekly Contest 73


## 2190. Most Frequent Number Following Key In an Array

### Solution: Counter + Hashmap

```py
class Solution:
    def mostFrequent(self, nums: List[int], key: int) -> int:
        c = Counter()
        for i in range(1, len(nums)):
            if nums[i-1]==key:
                c[nums[i]] += 1
        most = c.most_common()
        return most[0][0]
```

## 2191. Sort the Jumbled Numbers

### Solution: custom sort

```py
class Solution:
    def sortJumbled(self, mapping: List[int], nums: List[int]) -> List[int]:
        def map_sort(num):
            now = 0
            for x in str(num):
                x = ord(x)-ord('0')
                now = now*10+mapping[x]
            return now
                
        nums.sort(key=map_sort)
        return nums
```


```py
class Solution:
    def sortJumbled(self, mapping: List[int], nums: List[int]) -> List[int]:
        def map_sort(num):
            now = 0
            for x in str(num):
                x = ord(x)-ord('0')
                now = now*10+mapping[x]
            return now
        nums = zip(nums, [map_sort(num) for num in nums])
        return [x for x,y in sorted(nums, key=lambda x: x[1])]
```

## 2192. All Ancestors of a Node in a Directed Acyclic Graph

### Solution: DFS + iterate through each ancestor in order and add it to answer for each node you can reach in dfs

```py
class Solution:
    def getAncestors(self, n: int, edges: List[List[int]]) -> List[List[int]]:
        answer = [[] for _ in range(n)]
        graph = defaultdict(list)
        for fro, to in edges:
            graph[fro].append(to)
        def dfs(ancestor, node):
            for nei in graph[node]:
                if answer[nei] and answer[nei][-1] == ancestor: continue
                answer[nei].append(ancestor)
                dfs(ancestor, nei)
            
        for i in range(n): dfs(i,i)
        return answer
```

## 2193. Minimum Number of Moves to Make Palindrome

### Solution: 2 pointer algorithm + greedily choose the best swap with left or right as the pivot character

"scpcyxprxxsjyjrww"


```py
class Solution:
    def minMovesToMakePalindrome(self, s: str) -> int:
        s = list(s)
        cnt = 0
        n = len(s)
        for L in range(n//2):
            R = n-L-1
            if s[L]!=s[R]:
                right = R
                while s[L] != s[right]:
                    right-=1
                left = L
                while s[left] != s[R]:
                    left += 1
                if R-right < left-L: # it is better to swap from element at right position to the end R position
                    cnt += R-right
                    for k in range(right,R):
                        s[k] = s[k+1]
                else:
                    cnt += left-L
                    for k in range(left, L, -1):
                        s[k] = s[k-1]
                    
        return cnt
```