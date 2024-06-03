# Knapsack Algorithms

## Here are a bunch of 0/1 knapsack problems

0/1 just means you can either take the item or not,  but it removes the possibility of taking fractions of an item or something. This is more given the set, and you are selecting a subset that optimizes on some constraint. 

## reversible counting knapsack algorithm

### knapsack algorithm for counting number of ways to form subset sums

```py
for i in range(k, x - 1, -1):
    dp[i] += dp[i - x]
```

### knapsack algorithm for removing element from subset sums

```py
for i in range(x, k + 1):
    dp[i] -= dp[i - x]
```

## Bounded Knapsack Problem

This is the iteration style of knapsack dp

```cpp
bool canPartition(vector<int>& nums) {
    int n = nums.size(), sum = accumulate(nums.begin(), nums.end(),0);
    if (sum%2==1) {return false;}
    sum/=2;
    vector<vector<bool>> dp(n+1, vector<bool>(sum+1,false));
    for (int i = 0;i<=n;i++) {
        dp[i][0] = true;
    }
    for (int s = 1;s<=sum;s++) {
        for (int i = 0;i<n;i++) {
            dp[i+1][s] = dp[i][s];
            if (nums[i]<=s) {
                dp[i+1][s] = dp[i+1][s] || dp[i][s-nums[i]];
            }
        }
    }
    return dp[n][sum];
}
```

## Unbounded Knapsack Problem

Example of knapsack problem where you want to maximize the value for taking items up to total weight.  Unbounded means you can take an item an infinite number of times, or in another way you can take an item with no bounds. 

This problem also had a fact that the total weight of all the items was bounded by something like 200,000, that means you can only have at most sqrt(MAXN) distinct weights for items in this regard.  So you can solve it in O(N*sqrt(N)) basically. 

```cpp
const int MAXN = 2e5 + 5;
int N, M;
int items[MAXN];
vector<int> values, weights, dp;

void solve() {
    cin >> N >> M;
    memset(items, 0, sizeof(items));
    for (int i = 0; i < N; i++) {
        int v;
        string s;
        cin >> s >> v;
        items[s.size()] = max(items[s.size()], v);
    }
    for (int i = 1; i < MAXN; i++) {
        if (!items[i]) continue;
        weights.push_back(i);
        values.push_back(items[i]);
    }
    int V = values.size();
    dp.assign(M + 1, 0);
    for (int cap = 0; cap <= M; cap++) {
        for (int i = 0; i < V; i++) {
            if (cap < weights[i]) break;
            dp[cap] = max(dp[cap], dp[cap - weights[i]] + values[i]);
        }
    }
    cout << dp.end()[-1] << endl;
}
```


## Min Cost Knapsack Problem

This is also bounded knapsack problem.

Minimize the cost of items taken with a given total weight

example implementation, the prefix sum is just because you aren't allowed to spend over some amount of cost at that specific iteration, notice it increments by a constant x.
This was just an added constraint to the problem this solved, but otherwise it is a traditional dp approach to the min cost knapsack problem

weight is happy vector here, and cost is costs vector

```cpp
const int INF = 1e18;

void solve() {
    int m, x;
    cin >> m >> x;
    vector<int> costs(m);
    vector<int> happy(m);
    int H = 0; 
    for (int i = 0; i < m; ++i) {
        cin >> costs[i] >> happy[i];
        H += happy[i];
    }
    vector<int> dp(H + 1, INF), ndp(H + 1, INF);
    dp[0] = 0; // (happiness) -> cheapest cost)
    int psum = 0;
    for (int i = 0; i < m; ++i) {
        ndp.assign(H + 1, INF);
        for (int j = 0; j < H; j++) {
            if (dp[j] + costs[i] <= psum) {
                ndp[j + happy[i]] = min(ndp[j + happy[i]], dp[j] + costs[i]);
            }
            ndp[j] = min(ndp[j], dp[j]);
        }
        swap(dp, ndp);
        psum += x;
    }
    int ans = 0;
    for (int i = 0; i <= H; i++) {
        if (dp[i] < INF) {
            ans = max(ans, i);
        }
    }
    cout << ans << endl;
}
```

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