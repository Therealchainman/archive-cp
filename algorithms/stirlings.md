# Stirlings Numbers

## Stirling's Numbers of the Second Kind

This solves the problem of counting the number of ways to partition a set of n elements into k non-empty subsets.  The Stirling number of the second kind is denoted as S(n, k).

time complexity and space complexity is O(n^2). This is a dynamic programming solution.

The recurrence relation for Stirling numbers of the second kind, S(n,k)=k⋅S(n−1,k)+S(n−1,k−1). This recurrence helps count the number of ways to partition n objects into k non-empty subsets. To understand why this recurrence works, consider the following:

Base case: S(n,1)=1 for all n, since there's only one way to partition n objects into a single subset: put all the objects together.
Also, S(n,k)=0 if k>n, since it's impossible to partition n objects into more than n non-empty subsets.

Recurrence breakdown:
When partitioning n objects into k subsets, consider the nth object separately. There are two possibilities:
The nth object forms its own subset. The number of ways to partition the remaining n−1 objects into k−1 subsets is S(n−1,k−1).
The nth object is added to one of the existing subsets. There are k subsets to choose from, and once a subset is chosen, the remaining n−1 objects must be partitioned into k subsets. This is counted by k⋅S(n−1,k).
Thus, the total number of ways to partition n objects into k subsets is the sum of these two cases:

S(n,k)=k⋅S(n−1,k)+S(n−1,k−1)
This recurrence captures all possibilities of how the nth object can be distributed across k subsets, ensuring that all partitions are counted correctly.

```cpp
vector<vector<int>> s;
void stirlings(int n, int m) {
    s.assign(n + 1, vector<long long>(n + 1, 0));
    s[0][0] = 1;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= i; j++) {
            s[i][j] = (s[i - 1][j - 1] + j * s[i - 1][j]) % m;
        }
    }
}
```