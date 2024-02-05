## 1043. Partition Array for Maximum Sum

### Solution 1:  dynammic programming, O(n^2)

dp[i] = maximum sum of partitioning arr[:i + 1] into segments of length at most k when setting the values equal to the max value in each segment. 

For each position i it computes the maximum sum that can be achieved by partitioning the array up to and including the ith element.

Then it increases the size of the current partition that includes i, by moving the j pointer back until it reaches the max size of k.  And it tracks the maximum element in that partition, as that will be the value of all elements in the partition.  And then it computes the value by taking the maximum sum of the partition up to j, and adding the value of the partition to the sum.  And then it updates the dp[i + 1] with the maximum value of the partition.

```py
class Solution:
    def maxSumAfterPartitioning(self, arr: List[int], k: int) -> int:
        n = len(arr)
        dp = [-math.inf] * (n + 1)
        dp[0] = 0
        for i in range(n):
            segmax = -math.inf
            for j in range(i, max(-1, i - k), -1):
                segmax = max(segmax, arr[j])
                dp[i + 1] = max(dp[i + 1], dp[j] + (i - j + 1) * segmax)
        return dp[-1]
```

##

### Solution 1:

```py

```

##

### Solution 1:

```py

```