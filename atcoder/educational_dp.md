# Educational DP

##

### Solution 1: 

```cpp

```

## F - LCS

### Solution 1: dynamic programming, longest common subsequence

dp problem with O(NM) time and space complexity

This one you need backtracking to reconstruct the LCS string itself.

```cpp
string S, T;

void solve() {
    cin >> S >> T;
    int N = S.size(), M = T.size();
    vector<vector<int>> dp(N + 1, vector<int>(M + 1, 0));
    vector<vector<pair<int, int>>> C(N + 1, vector<pair<int, int>>(M + 1, {-1, -1}));
    for (int i = 1; i <= N; ++i) {
        for (int j = 1; j <= M; ++j) {
            dp[i][j] = dp[i - 1][j];
            C[i][j] = C[i - 1][j];
            if (dp[i][j - 1] > dp[i][j]) {
                dp[i][j] = dp[i][j - 1];
                C[i][j] = C[i][j - 1];
            }
            if (S[i - 1] == T[j - 1] && dp[i - 1][ j - 1] >= dp[i][j]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
                C[i][j] = {i - 1, j - 1};
            }
        }
    }
    string ans = "";
    int i = N, j = M;
    while (true) {
        tie(i, j) = C[i][j];
        if (i == -1 && j == -1) break;
        ans += S[i];
    }
    reverse(ans.begin(), ans.end());
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

## G - Longest Path

### Solution 1: directed acyclic graph dynamic programming, longest path, dp push, topological sort

```cpp
int N, M;
vector<vector<int>> adj;

void solve() {
    cin >> N >> M;
    adj.assign(N, vector<int>());
    vector<int> indeg(N, 0);
    for (int i = 0; i < M; ++i) {
        int u, v;
        cin >> u >> v;
        u--, v--;
        adj[u].emplace_back(v);
        indeg[v]++;
    }
    vector<int> dp(N, 0); 
    queue<int> q;
    for (int i = 0; i < N; ++i) {
        if (!indeg[i]) q.emplace(i);
    }
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (int v : adj[u]) {
            dp[v] = max(dp[v], dp[u] + 1);
            if (--indeg[v] == 0) {
                q.emplace(v);
            }
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

## H - Grid 1

### Solution 1: grid dynamic programming, counting paths

```cpp
const int MOD = 1e9 + 7;
int R, C;
vector<string> grid;

void solve() {
    cin >> R >> C;
    grid.resize(R);
    for (int i = 0; i < R; ++i) {
        cin >> grid[i];
    }
    vector<vector<int>> dp(R, vector<int>(C, 0));
    dp[0][0] = 1;
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < C; ++c) {
            if (r == 0 && c == 0 || grid[r][c] == '#') continue;
            if (r > 0) dp[r][c] = (dp[r][c] + dp[r - 1][c]) % MOD;
            if (c > 0) dp[r][c] = (dp[r][c] + dp[r][c - 1]) % MOD;
        }
    }
    cout << dp[R - 1][C - 1] << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## I - Coins

### Solution 1: dynamic programming, probability

Use the state dp[i][j] = probability to get j tails after i coins

Fun fact this is not related to Bernoulli trial because for that you are doing independent trials with same probability.

Here you flipping coins with different probabilities.

Why does dp work here, cause you can say you know the probability to reach this particular state after flipping i coins and you have flipped j heads so far,  So now you consider what is the transition if you flip the next coin, it can be head or tail and you can update the dp accordingly.

```cpp
int N;
vector<long double> P;

void solve() {
    cin >> N;
    P.resize(N);
    for (int i = 0; i < N; ++i) {
        cin >> P[i];
    }
    vector<vector<long double>> dp(N + 1, vector<long double>(N + 1, 0));
    dp[0][0] = 1;
    for (int i = 1; i <= N; ++i) {
        for (int j = 0; j <= N; ++j) {
            // success
            if (j > 0) dp[i][j] += dp[i - 1][j - 1] * P[i - 1];
            // failure
            dp[i][j] += dp[i - 1][j] * (1.0 - P[i - 1]);
        }
    }
    long double ans = 0;
    for (int i = N / 2 + 1; i <= N; ++i) {
        ans += dp[N][i];
    }
    cout << fixed << setprecision(15) << ans << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## J - Sushi

### Solution 1: 

If you think carefully, you don’t actually need to know “which plate has how many.” You only need to know “how many plates have 1 piece, how many have 2 pieces, and how many have 3 pieces.” The expected value is determined solely by those counts.

recurrence relation

Just need to know the expected number of steps from a state (i, j, k) where i is number of plates with 1 piece, j is number of plates with 2 pieces, k is number of plates with 3 pieces. 

So for a transition just add 1 operation 

```cpp

```

## K - Stones

### Solution 1: 

impartial game dynamic programming with boolean true, false for who can win from that state

recurrence relation

dp[i] = winning or losing position when i stones have been taken so far.
Everything starts as a losing position

You find dp[i] is a winning position if you can find one losing position you can transition to. 

```cpp
int N, K;
vector<int> A;

void solve() {
    cin >> N >> K;
    A.resize(N);
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
    }
    vector<bool> dp(K + 1, false);
    for (int i = K; i >= 0; --i) {
        for (int x : A) {
            if (i - x < 0) break;
            dp[i - x] = dp[i - x] | !dp[i];
        }
    }
    string ans = dp[0] ? "First" : "Second";
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

## L - Deque

### Solution 1: 

minimax dynamic programming

Where the states dp[i][j][k] represents the maximum or minimum depending on if k is 0 or 1.  Let's say when k = 0, that dp[i][j][k] represents the maximum score, then the i and j represents A[i,...j] subarray remains, where you can take element from i or j end.

base case A[0,...,N - 1], 

if i > j, no elements remain so score is 0.

```cpp

```

## M - candies

### Solution 1: dynamic programming, counting, sliding window sum optimization

dp counting problem

The dp state is dp(i, j) = the number of ways to distribute candies to A[0,..,i] with j candies used so far. 

the transition state would be dp(i, j) = $\sum_{k=j-a_i}^j dp(i - 1, k)$

Which can solve by using sliding window sum to compute the sum for the window of size $a_i + 1$ efficiently.

```cpp
const int MOD = 1e9 + 7;
int N, K;
vector<int> A;

void solve() {
    cin >> N >> K;
    A.resize(N);
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
    }
    vector<int> dp(K + 1, 0), ndp(K + 1, 0);
    dp[0] = 1;
    for (int i = 0; i < N; ++i) {
        int wsum = 0;
        ndp.assign(K + 1, 0);
        for (int j = 0; j <= K; ++j) {
            wsum = (wsum + dp[j]) % MOD;
            ndp[j] = wsum;
            if (j >= A[i]) wsum -= dp[j - A[i]];
            if (wsum < 0) wsum += MOD;
        }
        swap(dp, ndp);
    }
    cout << dp[K] << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## N - Slimes

### Solution 1: 

```cpp

```

## O - Matching

### Solution 1: bitmask dp, combinatorial counting, bipartite graphing perfect matching

Easy to derive the O(N^2 * 2^N) dp solution, where you just store bitmask of already selected women. And you iterate over the men so the state is dp[i][mask] = number of ways,  so you consider matching the ith man for each of the previous possible masks to match with women, and add one more. 

Counting perfect matchings in a bipartite graph

You can speed up by making observation that for each mask, you can determine it has matched with the number of men by taking popcount(mask), so you can reduce the dp to one dimension dp[mask] only.

Now you just iterate over the increasing masks, and for each mask you know the ith men it is for, so then you just have to iterate over the candidate women that the ith man is compatible with and not already matched in the mask.

```cpp
const int MOD = 1e9 + 7;
int N;
vector<vector<int>> grid;

void solve() {
    cin >> N;
    grid.assign(N, vector<int>(N, 0));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            cin >> grid[i][j];
        }
    }
    vector<int> dp(1 << N, 0);
    dp[0] = 1;
    for (int mask = 0; mask + 1 < (1 << N); ++mask) {
        int men = __builtin_popcount(mask);
        for (int i = 0; i < N; ++i) {
            if (!grid[men][i]) continue; // incompatible
            if ((mask >> i) & 1) continue;
            int nmask = mask | (1 << i);
            dp[nmask] = (dp[nmask] + dp[mask]) % MOD;
        }
    }
    cout << dp.back() << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## P - Independent Set

### Solution 1: postorder dfs, tree dp, combinatorics

Given a tree you want to paint each node white or black, following the constraint that no two adjacent vertex can be painted black.

Draw out the case for when you have one node with 2 children and 3 children, you end up with this thing where you are taking the product which for 3 nodes is based on this (w1 + b1) * (w2 + b2) * (w3 + b3) for each child node 1, 2, 3.  And so you see it is the product of (w + b) for each child node.

And for the black case, you can only have white children so it is product of w for each child node.

You can also use or and and logic for combinations, for each child node, you can either take white or black for the parent node, so you can use or logic to combine the two cases.

For combining child nodes, you can use and logic since you need to satisfy both child nodes conditions, and you can take child a and child b independently. 

```cpp
const int MOD = 1e9 + 7;
int N;
vector<vector<int>> adj;
vector<int> dp[2];

void dfs(int u, int p = -1) {
    int64 white = 1, total = 1;
    for (int v : adj[u]) {
        if (v == p) continue;
        dfs(v, u);
        white = (white * dp[0][v]) % MOD;
        total = (total * (dp[0][v] + dp[1][v])) % MOD;
    }
    dp[0][u] = total; // paint white
    dp[1][u] = white; // paint black
}

void solve() {
    cin >> N;
    adj.assign(N, vector<int>());
    for (int i = 0; i < N - 1; ++i) {
        int u, v;
        cin >> u >> v;
        u--, v--;
        adj[u].emplace_back(v);
        adj[v].emplace_back(u);
    }
    for (int i = 0; i < 2; ++i) {
        dp[i].assign(N, 0);
    }
    dfs(0);
    int ans = (dp[0][0] + dp[1][0]) % MOD;
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

## Q - Flowers

### Solution 1: 

```cpp

```

## R - Walk

### Solution 1: binary jumping, directed graph, sparse table, matrix exponentiation

binary jumping algorithm, consider every triplet (u, v, w) where you have a walk from u, to v of length 2^i and walk from v to w of length 2^i, then you can combine them to get a walk from u to w of length 2^(i + 1).  And the number of walks is just the product of the two walks counts.

But you can do this binary jumping with just matrix exponentiation of the adjacency matrix. 

Take the adjacency matrix and multiple by itself and you will see you will calculate r * c1 * r1 * c, and c1 = r1, so it is acting as those triplets (u, v, w) above. where v = c1 = r1 and u = r and w = c.

Drawing it out for the first case of taking walk of size 1 helps make it make sense. 

```cpp

```

## T - Permutation

### Solution 1: 



```cpp

```

## U - Grouping

### Solution 1: bitmask dp

recursive dp, where you calculate the cost for a subset, and then consider splitting by summing and making into smaller subsets of that set.

```cpp

```

## V - Subtree

### Solution 1: 

reroot dp 

prefix and suffix product to calculate the parent product without including the child subtree


```cpp

```

## W - Intervals

### Solution 1: 

```cpp

```

## X - Tower

### Solution 1: 

```cpp

```

## Y - Grid 2

![images](images/grid_2_1.png)
![images](images/grid_2_2.png)
![images](images/grid_2_3.png)

### Solution 1: 

```cpp

```

## Z - Frog 3

### Solution 1: 

```cpp

```

## S - Digit Sum 

### Solution 1:  digit dp, remainder math

When you take the modulus is important, if you added cnt first and then % mod, it TLE.

```py
mod = int(1e9) + 7

# dp state (index, remainder modulo D of digit sum, tight, zero)
def main():
    K = input()
    D = int(input())
    dp = Counter()
    dp[(0, True, True)] = 1
    for d in map(int, K):
        ndp = Counter()
        for (rem, tight, zero), cnt in dp.items():
            for dig in range(10 if not tight else d + 1):
                nrem, ntight, nzero = (rem + dig) % D, tight and dig == d, zero and dig == 0
                ndp[(nrem, ntight, nzero)] = (ndp[(nrem, ntight, nzero)] + cnt) % mod
        dp = ndp
    ans = sum(cnt for (rem, _, zero), cnt in dp.items() if rem == 0 and not zero) % mod
    print(ans)

if __name__ == '__main__':
    main()
```

### Solution 2:  Same as above but with arrays

It is not that much, faster so the approach above is simpler to code anyway so probably recommend it. 

```py
from itertools import product
mod = int(1e9) + 7

# dp state (index, remainder modulo D of digit sum, tight, zero)
def main():
    K = input()
    D = int(input())
    dp = [[[0] * 2 for _ in range(2)] for _ in range(D)]
    dp[0][1][1] = 1
    for d in map(int, K):
        ndp = [[[0] * 2 for _ in range(2)] for _ in range(D)]
        for rem, tight, zero in product(range(D), range(2), range(2)):
            for dig in range(10 if not tight else d + 1):
                nrem, ntight, nzero = (rem + dig) % D, tight and dig == d, zero and dig == 0
                ndp[nrem][ntight][nzero] = (ndp[nrem][ntight][nzero] + dp[rem][tight][zero]) % mod
        dp = ndp
    ans = sum(dp[0][t][0] for t in range(2)) % mod
    print(ans)

if __name__ == '__main__':
    main()
```

## knapsack 1

### Solution 1:  0/1 knapsack dp

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

signed main() {
    solve();
    return 0;
}
```

## knapsack 2

### Solution 1:  0/1 min cost knapsack dp

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

signed main() {
    solve();
    return 0;
}
```
