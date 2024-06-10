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

## 0/1 Knapsack Problem

This is the 0/1 knapsack iterative dp approach

```cpp
int N, W;
vector<int> values, weights, dp, ndp;

void solve() {
    cin >> N >> W;
    values.resize(N);
    weights.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> weights[i] >> values[i];
    }
    dp.assign(W + 1, 0);
    for (int i = 0; i < N; i++) {
        ndp.assign(W + 1, 0);
        for (int cap = 0; cap <= W; cap++) {
            if (cap >= weights[i]) {
                ndp[cap] = max(ndp[cap], dp[cap - weights[i]] + values[i]);
            }
            ndp[cap] = max(ndp[cap], dp[cap]);
        }
        swap(dp, ndp);
    }
    cout << dp[W] << endl;
}
```

## Bounded Knapsack Problem

This one is slightly different than above, you are given some fixed amount of each item, but I think you can treat it like 0/1 knapsack if you treat each item as an individual item. 

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


## 0/1 Min Cost Knapsack Problem

Minimize the cost of items taken with a given total weight where you can take an item only once.

```cpp
const int INF = 1e18;
int N, W;
vector<int> values, weights, dp, ndp;

void solve() {
    cin >> N >> W;
    int V = 0;
    values.resize(N);
    weights.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> weights[i] >> values[i];
        V += values[i];
    }
    dp.assign(V + 1, INF);
    dp[0] = 0;
    for (int i = 0; i < N; i++) {
        ndp.assign(V + 1, INF);
        for (int v = 0; v <= V; v++) {
            ndp[v] = min(ndp[v], dp[v]);
            if (values[i] <= v) {
                ndp[v] = min(ndp[v], dp[v - values[i]] + weights[i]);
            }
        }
        swap(dp, ndp);
    }
    int ans = 0;
    for (int v = 0; v <= V; v++) {
        if (dp[v] <= W) ans = v;
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