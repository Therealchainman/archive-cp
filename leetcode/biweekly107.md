# Leetcode Biweekly Contest 107

## 2744. Find Maximum Number of String Pairs

### Solution 1:  brute force

```py
class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        n = len(words)
        res = 0
        for i in range(n):
            for j in range(i + 1, n):
                if words[i] == words[j][::-1]:
                    res += 1
                    break
        return res
```

## 2745. Construct the Longest New String

### Solution 1:  math

derived these by realizing legal pairs were (xy, yx, zx, yz, zz)
1. if x > y, can take all the z strings, take one extra x
zz...zzxyxy...xyx
2. if y > x, can take all the z strings, take on extra y
yxyx....yxyzz...zz
3. if x == y, take everything x + y + z

```py
class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        a = min(x, y)
        extra = min(1, max(x, y) - a)
        res = 2 * a + extra + z
        return 2 * res
```

## 2746. Decremental String Concatenation

### Solution 1:  dynamic programming + minimize the length of the concatenated string

two recurrence relations, and the dp states for 0...i words are already solved, and the only thing that matters is the
(first_character, last_character) that makes up that concatenated string, and store the length, cause want to minimize on that. 

So updates will be something like dp[first][last] = min(dp[first][last], ....)

read code explains logic well

```py
class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        n = len(words)
        dp = {(words[0][0], words[0][-1]): len(words[0])}
        for i in range(1, n):
            ndp = defaultdict(lambda: math.inf)
            fc, lc = words[i][0], words[i][-1]
            for (s, e), l in dp.items():
                ndp[(s, lc)] = min(ndp[(s, lc)], l + len(words[i]) - (1 if e == fc else 0))
                ndp[(fc, e)] = min(ndp[(fc, e)], l + len(words[i]) - (1 if s == lc else 0))
            dp = ndp
        return min(dp.values())
```

## 2747. Count Zero Request Servers

### Solution 1:  offline queries + sort + two pointers + sliding window of size x + frequency counter

sort the queries and the logs based on time. 
Then take two pointers left, right
And move the right pointer up to the current queries[i]
And move the left pointer up to less than queries[i] - x 

track frequency of each server, and update the cnt appropriately
cnt represents the number of server that have received a server request in the current window
So the answer will be total number of servers - cnt will give server with 0 requests in the current query window.

```py
class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        nlogs, m = len(logs), len(queries)
        cnt = 0
        freq = Counter()
        queries = sorted([(v, i) for i, v in enumerate(queries)])
        ans = [0] * m
        left = right = 0
        logs.sort(key = lambda x: x[1])
        for v, i in queries:
            while right < nlogs and logs[right][1] <= v:
                server = logs[right][0]
                freq[server] += 1
                if freq[server] == 1:
                    cnt += 1
                right += 1
            while left < nlogs and logs[left][1] < v - x:
                server = logs[left][0]
                freq[server] -= 1
                if freq[server] == 0:
                    cnt -= 1
                left += 1
            ans[i] = n - cnt
        return ans
```