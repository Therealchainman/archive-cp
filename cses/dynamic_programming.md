# Dynamic Programming

## At the top of each script

```cpp
#include <bits/stdc++.h>
using namespace std;

inline int read()
{
	int x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}

inline long long readll() {
	long long x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}
```

## Coin Combinations I

### Solution 1:  iterative dp + order doesn't matter + Unordered Coin Change + O(nx) time

This can be solved by having two for loops in a particular order.

Iterate through the sum of the coins first and then through the coins, and add the coins for that sum. This leads to adding up very quicly for example if you have

```py
coins = [2, 3, 5], x = 9
dp = [[], [], [], [], [], [], [], [], [], []]
for when coin_sum = 2 it becomes
dp = [[], [], [2], [], [], [], [], [], [], []]
coin_sum = 3
dp = [[], [], [2], [3], [], [], [], [], [], []]
coin_sum = 4
dp = [[], [], [2], [3], [2, 2], [], [], [], [], []]
coin_sum = 5
dp = [[], [], [2], [3], [2, 2], [[3, 2], [2, 3], [5]], [], [], [], []]
coin_sum = 6
dp = [[], [], [2], [3], [2, 2], [[3, 2], [2, 3], [5]], [[2, 2, 2], [3, 3]], [[3, 2, 2], [2, 3, 2], [5, 2], [2, 2, 3], [2, 5]], [], []]
```

```cpp
int main() {
    int n = read(), x = read();
    int mod = 1e9 + 7;
    vector<int> dp(x + 1, 0);
    dp[0] = 1;
    vector<int> coins;
    for (int i = 0; i < n; i++) {
        int c = read();
        coins.push_back(c);
    }
    for (int coin_sum = 1; coin_sum <= x; coin_sum++) {
        for (int i = 0; i < n; i++) {
            if (coins[i] > coin_sum) continue;
            dp[coin_sum] = (dp[coin_sum] + dp[coin_sum - coins[i]]) % mod;
        }
    }
    cout << dp[x] << endl;
    return 0;
}
```

## Coin Combinations II

### Solution 1: iterative dp + order matters + O(nx) time

For this problem you want to iterate through the coins first and then the coin_sum. This is because you want to add the coins in a particular order. For example if you have

```py
coins = [2, 3, 5], x = 9
dp = [[], [], [], [], [], [], [], [], [], []]
coin = 2
dp = [[], [], [2], [], [[2, 2]], [], [[2, 2, 2]], [], [[2, 2, 2, 2]], []]
coin = 3
dp = [[], [], [2], [[3]], [[2, 2]], [[2, 3]], [[2, 2, 2], [3, 3]], [[2, 2, 3]], [[2, 2, 2, 2], [2, 3, 3]], [[2, 2, 2, 3], [3, 3, 3]]]
coin = 5
dp = [[], [], [2], [[3]], [[2, 2]], [[2, 3], [5]], [[2, 2, 2], [3, 3]], [[2, 2, 3], [2, 5]], [[2, 2, 2, 2], [2, 3, 3], [3, 5]], [[2, 2, 2, 3], [3, 3, 3], [2, 2, 5]]]
```

```cpp
int main() {
    int n = read(), x = read();
    int mod = 1e9 + 7;
    vector<int> dp(x + 1, 0);
    dp[0] = 1;
    vector<int> coins;
    for (int i = 0; i < n; i++) {
        int c = read();
        coins.push_back(c);
    }
    for (int i = 0; i < n; i++) {
        for (int coin_sum = coins[i]; coin_sum <= x; coin_sum++) {
            if (coins[i] > coin_sum) continue;
            dp[coin_sum] = (dp[coin_sum] + dp[coin_sum - coins[i]]) % mod;
        }
    }
    cout << dp[x] << endl;
    return 0;
}
```

##

### Solution 1: 

```py

```

## Counting Towers

### Solution 1: 

```py
def main():
    n = int(input())
    mod = int(1e9) + 7
    psum = 1
    dp = 1
    for i in range(1, n + 1):
        psum += pow(2, 2 * i - 2, mod)
        psum %= mod
        print('psum', psum)
        dp = psum
        psum += dp
        psum %= mod
        # print(i, dp)
    return dp


if __name__ == '__main__':
    # print(main())
    T = int(input())
    for _ in range(T):
        print(main())
```

## Projects

### Solution 1:  sort + iterative dynammic programming + coordinates compression

```py
def main():
    n = int(input())
    events = []
    days = set()
    for i in range(n):
        a, b, p = map(int, input().split())
        events.append((a, -p, 0))
        events.append((b, p, a))
        days.update([a, b])
    compressed = {x: i + 1 for i, x in enumerate(sorted(days))}
    events.sort()
    dp = [0] * (len(compressed) + 1)
    for day, p, start in events:
        i = compressed[day]
        if p < 0:
            dp[i] = max(dp[i], dp[i - 1])
        else:
            dp[i] = max(dp[i], dp[i - 1], dp[compressed[start] - 1] + p)
    return dp[-1]

if __name__ == '__main__':
    print(main())
```

```cpp
int main() {
    int n = read();
    vector<tuple<int, int, int>> events;
    set<int> days;
    for (int i = 0; i < n; i++) {
        int a = read(), b = read(), p = read();
        events.push_back({a, -p, 0});
        events.push_back({b, p, a});
        days.insert(a);
        days.insert(b);
    }
    map<int, int> compressed;
    int i = 1;
    for (auto day : days) {
        compressed[day] = i++;
    }
    sort(events.begin(), events.end());
    vector<long long> dp(i + 1);
    for (auto [day, p, start] : events) {
        i = compressed[day];
        if (p < 0) {
            dp[i] = max(dp[i], dp[i - 1]);
        } else {
            dp[i] = max(dp[i], dp[i - 1]);
            dp[i] = max(dp[i], dp[compressed[start] - 1] + p);
        }
    }
    cout << dp[i] << endl;
}
```

## Removal Game

### Solution 1:  dynammic programming + interval

dp(i, j) = maximum score player can score compared to score of other player for the interval [i, j)

```cpp
int main() {
    int n = read();
    vector<long long> numbers(n);
    for (int i = 0; i < n; i++) {
        numbers[i] = readll();
    }
    vector<vector<long long>> dp(n + 1, vector<long long>(n + 1, LONG_LONG_MIN));
    for (int i = 0; i <= n; i++) {
        dp[i][i] = 0;
    }
    for (int len = 1; len <= n; len++) {
        for (int i = 0; i + len <= n; i++) {
            int j = i + len;
            dp[i][j] = max(dp[i][j], numbers[i] - dp[i + 1][j]);
            dp[i][j] = max(dp[i][j], numbers[j - 1] - dp[i][j - 1]);
        }
    }
    long long res = (dp[0][n] + accumulate(numbers.begin(), numbers.end(), 0LL)) / 2;
    cout << res << endl;
}
```

## Two Sets II

### Solution 1:  0/1 knapsack dp problem

dp[i][x] = count of ways for the subset of elements in 0...i with sum of x
dp[i][x] = dp[i-1][x] + dp[i-1][x-i]
Convert to 0/1 knapsack where you can either take the element or not take it.  It can be converted to this by realize that you just need to find the number of ways that the sum is equal to n*(n+1)/4, 

cause the summation of the natural number is n*(n+1)/2, but you just need a set to reach half the sum, then the other elements must be in other set and the sum of each set is equal.  So just need to look for half, can quickly check if it is odd, then there is 0 solutions. 

Then just iterate through all the possibilities with dynammic programming

```cpp
long long mod = int(1e9) + 7;

int main() {
    int n = read();
    int target = n * (n + 1) / 2;
    if (target & 1) {
        cout << 0 << endl;
        return 0;
    }
    target /= 2;
    vector<vector<long long>> dp(n + 1, vector<long long>(target + 1, 0));
    dp[0][0] = 1;
    for (int i = 1; i < n; i++) {
        for (int j = 0; j <= target; j++) {
            dp[i][j] = dp[i - 1][j];
            if (j - i >= 0) dp[i][j] = (dp[i][j] + dp[i - 1][j - i]) % mod;
        }
    }
    cout << dp[n - 1][target] << endl;
}
```

## Elevator Rides

### Solution 1:  bitmask dp

dp[mask] = minimum pair of number of rides and then weight on last ride.  
So the best combination of these two values for taking a subset of weights is the best solution to subproblem, where subproblem is that of taking this subset of weights. 

```py
import math

def main():
    n, x = map(int, input().split())
    weights = list(map(int, input().split()))
    dp = [(math.inf, math.inf)] * (1 << n)
    dp[0] = (1, 0) # number rides, weight on last ride
    for mask in range(1, 1 << n):
        for i in range(n):
            if (mask >> i) & 1:
                prev_mask = mask ^ (1 << i)
                if dp[prev_mask][1] + weights[i] <= x:
                    dp[mask] = min(dp[mask], (dp[prev_mask][0], dp[prev_mask][1] + weights[i]))
                else:
                    dp[mask] = min(dp[mask], (dp[prev_mask][0] + 1, weights[i]))
    print(dp[-1][0])

if __name__ == '__main__':
    main()
```

## Grid Paths

### Solution 1:  dynamic programming + grid + counting number of paths + O(n^2)

```py
from itertools import product

def main():
    n = int(input())
    mod = int(1e9) + 7
    grid = [input() for _ in range(n)]
    wall = '*'
    dp = [[0] * n for _ in range(n)]
    dp[0][0] = 0 if grid[0][0] == wall else 1
    for r, c in product(range(n), repeat = 2):
        if grid[r][c] == wall: continue
        if r > 0:
            dp[r][c] = (dp[r][c] + dp[r - 1][c]) % mod
        if c > 0:
            dp[r][c] = (dp[r][c] + dp[r][c - 1]) % mod
    print(dp[-1][-1])

if __name__ == '__main__':
    main()
```

## Book Shop

### Solution 1: 0/1 knapsack, dynamic programming

```cpp
int N, W;
vector<int> weights, values;

void solve() {
    cin >> N >> W;
    weights.resize(N);
    values.resize(N);
    for (int i = 0; i < N; ++i) {
        cin >> weights[i];
    }
    for (int i = 0; i < N; ++i) {
        cin >> values[i];
    }
    vector<int> dp(W + 1, 0);
    for (int i = 0; i < N; ++i) {
        for (int j = W; j >= weights[i]; --j) {
            dp[j] = max(dp[j], dp[j - weights[i]] + values[i]);
        }
    }
    int ans = *max_element(dp.begin(), dp.end());
    cout << ans << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Book Shop II

### Solution 1: bounded knapsack dp, binary grouping or binary splitting optimization

1. You convert the multiple copies into single copies with some multiple of weight and value, such you can recover any possible number of copies you can take from it.
1. Reduces it to the 0/1 knapsack problem

```cpp
int N, W;
vector<int> ow, ov, counts, weights, values;

void solve() {
    cin >> N >> W;
    ow.resize(N);
    ov.resize(N);
    counts.resize(N);
    for (int i = 0; i < N; ++i) {
        cin >> ow[i];
    }
    for (int i = 0; i < N; ++i) {
        cin >> ov[i];
    }
    for (int i = 0; i < N; ++i) {
        cin >> counts[i];
    }
    for (int i = 0; i < N; ++i) {
        int c = 1;
        while (counts[i] > c) {
            counts[i] -= c;
            weights.emplace_back(c * ow[i]);
            values.emplace_back(c * ov[i]);
            c <<= 1;
        }
        // leftover
        if (counts[i]) {
            weights.emplace_back(counts[i] * ow[i]);
            values.emplace_back(counts[i] * ov[i]);
        }
    }
    int M = weights.size();
    vector<int> dp(W + 1, 0);
    for (int i = 0; i < M; ++i) {
        for (int j = W; j >= weights[i]; --j) {
            dp[j] = max(dp[j], dp[j - weights[i]] + values[i]);
        }
    }
    int ans = *max_element(dp.begin(), dp.end());
    cout << ans << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## 

### Solution 1: 

```cpp

```

## Counting Numbers

### Solution 1: digit dp, recursive memoization

```cpp
int64 L, R;
int64 dp[19][10][2][2];

int64 dfs(const string &num, int i, int d, int z, int t) {
    if (i == num.size()) return 1;
    if (dp[i][d][z][t] != -1) return dp[i][d][z][t];
    int c = num[i] - '0';
    int64 ans = 0;
    for (int v = 0; v < 10; ++v) {
        if (!z && v == d) continue; // doesn't matter if just padding left with zero still
        if (t && v > c) break;
        ans += dfs(num, i + 1, v, int(v == 0) & z, int(v == c) & t);
    }
    return dp[i][d][z][t] = ans;
}

int64 f(const string &num) {
    memset(dp, -1, sizeof(dp));
    return dfs(num, 0, 0, 1, 1);
}

void solve() {
    cin >> L >> R;
    int64 right = f(to_string(R));
    int64 left = L > 0 ? f(to_string(L - 1)) : 0;
    int64 ans = right - left;
    cout << ans << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Increasing Subsequence

### Solution 1: dynamic programming, greedy, binary search, patience sorting O(n log n)

dp[i] = smallest ending value of an increasing subsequence of length i+1

If you see a value which is larger than any prior then you can extend the longest increasing subsequence found so far by 1.

If you see a value which is greater than current dp[i] you update it to dp[i] = x where x is the new value, because this may allow for longer increasing subsequences to be formed later.

```cpp
int N;
vector<int> A;

void solve() {
    cin >> N;
    A.resize(N);
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
    }
    vector<int> values;
    for (int x : A) {
        int i = lower_bound(values.begin(), values.end(), x) - values.begin();
        if (i < values.size()) {
            values[i] = x;
        } else {
            values.emplace_back(x);
        }
    }
    int ans = values.size();
    cout << ans << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Increasing Subsequence II

### Solution 1: dynamic programming, counting, fenwick tree (binary indexed tree), coordinate compression

dp[v] = number of increasing subsequences that ends with A[i]
where v is the index in the compressed coordinates for A[i]

transition states: dp[v] = sum(dp[u]) + 1 for all u < v

Need a fenwick tree to quickly get the sum of dp[u] for all u < v, and also to support the point updates to the dp states.

```cpp
const int MOD = 1e9 + 7;
int N;
vector<int> A, values;

template <typename T>
struct FenwickTree {
    vector<T> nodes;
    T neutral;

    FenwickTree() : neutral(T(0)) {}

    void init(int n, T neutral_val = T(0)) {
        neutral = neutral_val;
        nodes.assign(n + 1, neutral);
    }

    void update(int idx, T val) {
        while (idx < (int)nodes.size()) {
            nodes[idx] = (nodes[idx] + val) % MOD;    
            idx += (idx & -idx);
        }
    }

    T query(int idx) {
        T result = neutral;
        while (idx > 0) {
            result = (result + nodes[idx]) % MOD;
            idx -= (idx & -idx);
        }
        return result;
    }

    T query(int left, int right) {
        int ans = right >= left ? query(right) - query(left - 1) : T(0);
        if (ans < 0) ans += MOD;
        return ans;
    }
};

void solve() {
    cin >> N;
    A.resize(N); values.resize(N);
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
        values[i] = A[i];
    }
    values.emplace_back(0);
    sort(values.begin(), values.end());
    values.erase(unique(values.begin(), values.end()), values.end());
    int M = values.size();
    FenwickTree<int> ft;
    ft.init(M);
    for (int x : A) {
        int i = lower_bound(values.begin(), values.end(), x) - values.begin();
        int res = ft.query(i) + 1; // add one for set with just {x}
        ft.update(i + 1, res);
    }
    int ans = ft.query(M);
    cout << ans << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```