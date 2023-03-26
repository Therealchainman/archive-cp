# Leetcode Weekly Contest 276


## 2138. Divide a String Into Groups of Size k

### Solution: Array

```c++
vector<string> divideString(string s, int k, char fill) {
    int n = s.size();
    vector<string> arr;
    for (int i = 0;i<n;i+=k) {
        arr.push_back(s.substr(i,k));
    }
    for (int i = arr.back().size();i<k;i++) {
        arr.back()+=fill;
    }
    return arr;
}
```


## 2139. Minimum Moves to Reach Target Score

### Solution: backward thinking + greedy (choose divide by 2 first)

```c++
int minMoves(int target, int maxDoubles) {
    int moves = 0;
    while (target>1 && maxDoubles--) {
        moves += 1 + target%2;
        target/=2;
    }
    return moves + target-1;
}
```


## 2140. Solving Questions With Brainpower

### Solution: dynamic programming + memoization

```c++
long long recurse(int i, vector<vector<int>>& Q, vector<long long>& dp) {
    int n = Q.size();
    if (i>=n) {return 0;}
    if (dp[i]!=-1) {
        return dp[i];
    }
    return dp[i] = max(Q[i][0]+recurse(i+Q[i][1]+1,Q,dp), recurse(i+1,Q,dp));
}
long long mostPoints(vector<vector<int>>& questions) {
    vector<long long> points((int)questions.size(), -1);
    return recurse(0,questions,points);
}
```

## 2141. Maximum Running Time of N Computers

### Solution: binary search

```c++

```