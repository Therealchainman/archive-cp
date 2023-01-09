# 0/1 Knapsack Problem

## Applied to solve the count of subsets with sum equal to k

For this problem the weight of each item is equal to it's value and the k is like the capacity of the knapsack.  
So you are not maximizing the value or anything but just counting the number of subsets

```py
def countSubsetSums(nums: List[int], k: int) -> int:
    n = len(nums)
    # the subproblem is the count of subsets for on the ith item at the jth sum
    dp = [[0]*(k+1) for _ in range(n+1)] 
    for i in range(n+1):
        dp[i][0] = 1 # i items, 0 sum/capacity 
        # only 1 unique solution which is the empty subset for this subproblem
    # i represents an item, j represents the current sum
    for i, j in product(range(1, n+1), range(1, k+1)):
        if nums[i-1] > j: # cannot place an item that exceeds the j, so it can only be excluded
            dp[i][j] = dp[i-1][j]
        else: # the subset count up to this sum is going to be if you combine the exclusion of this item added to the inclusion of this item
            dp[i][j] = dp[i-1][j] + dp[i-1][j-nums[i-1]]
    # count of subsets for when considered all items and capacity is k
    return dp[n][k]
```

## Applied to solve the maximization problem with values and weights for each item