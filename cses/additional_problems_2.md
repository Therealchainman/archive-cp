# Additional Problems II

## Bouncing Balls Steps

### Solution 1:  modular arithmetic, independent variables, lowest common multiple

1. One key is to identify that the position horizontal or vertical are independent of each other. 
1. And the number of corners you hit is based on the lowest common multiple.

```cpp
int64 N, M, K;

int calc(int n) {
    int v = K / n;
    if (v & 1) {
        return n - (K % n);
    }
    return K % n;
}

void solve() {
    cin >> N >> M >> K;
    int64 a = K / (N - 1); // side 
    int64 b = K / (M - 1); // side
    int64 corners = K / lcm<int64>(N - 1, M - 1); // corners
    int64 cnt = a + b - corners;
    int r = calc(N - 1), c = calc(M - 1);
    cout << r + 1 << " " << c + 1 << " " << cnt << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    int T;
    cin >> T;
    while (T--) {
        solve();
    }
    return 0;
}
```

## Bouncing Ball Cycle

### Solution 1:

```cpp

```

## Knight Moves Queries

### Solution 1:

```cpp

```

## K Subset Sums I

### Solution 1:

```cpp

```

## K Subset Sums II

### Solution 1:

```cpp

```

## Increasing Array II

### Solution 1:

```cpp

```

## Food Division

### Solution 1:

```cpp

```

## Swap Round Sorting

### Solution 1:

```cpp

```

## Binary Subsequences

### Solution 1:

```cpp

```

## School Excursion

### Solution 1:

```cpp

```

## Coin Grid

### Solution 1:

```cpp

```

## Grid Coloring II

### Solution 1:

```cpp

```

## Programmers and Artists

### Solution 1:

```cpp

```

## Removing Digits II

### Solution 1:

```cpp

```

## Coin Arrangement

### Solution 1:

```cpp

```

## Replace with Difference

### Solution 1:

```cpp

```

## Grid Puzzle I

### Solution 1:

```cpp

```

## Grid Puzzle II

### Solution 1:

```cpp

```

## Bit Substrings

### Solution 1:

```cpp

```

## Reversal Sorting

### Solution 1:

```cpp

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

## GCD Subsets

### Solution 1:

```cpp

```

## Minimum Cost Pairs

### Solution 1:

```cpp

```

## Same Sum Subsets

### Solution 1:

```cpp

```

## Mex Grid Queries

### Solution 1:

```cpp

```

## Maximum Building II

### Solution 1: monotonic stack, second-order difference array, prefix sums, suffix sums

Every rectangle has a left edge.
For that left edge, its maximum possible width is the minimum r-value among its rows.
The monotonic stack assigns that minimum to exactly one row.
That row contributes a whole range of heights using difference arrays.
Then a suffix sum over widths turns “maximum width” into “all usable widths.”

Choose a left column j.
Look downward at r[i][j].
Every vertical interval has a minimum r-value.
That minimum tells the maximum possible width.
The monotonic stack assigns each interval to exactly one minimum row.
The difference array adds all possible heights from that owner in O(1).
The suffix sum converts maximum width into every usable width.

monotonic stack = who owns the minimum?
difference array = how do I add many heights at once?
suffix sum = max width W also gives all widths <= W

Yes, the upper/lower bounds can overlap between different rows. That is okay. The key is:

The bounds are not saying “these rows form one rectangle.”
They are saying “these are the top/bottom choices for intervals where this row is the unique owner of the minimum width.”

Second-order difference array

A second-order difference array stores:

where the slope starts and stops changing

For the pattern:

value:  1  2  3  3  3  2  1  0

Look at how values change:

height: 1  2  3  4  5  6  7  8
value:  1  2  3  3  3  2  1  0
slope:  1  1  1  0  0 -1 -1 -1

The slope starts as +1.

Because ans is storing a second-order difference array.

second difference
→ prefix once gives slope
→ prefix twice gives actual count

h:          1  2  3  4  5  6  7  8  9
diff2:      1  0  0 -1  0 -1  0  0  1

h:          1  2  3  4  5  6  7  8  9
slope:      1  1  1  0  0 -1 -1 -1  0

h:          1  2  3  4  5  6  7  8  9
count:      1  2  3  3  3  2  1  0  0

Then:

prefix once  -> recover the slope
prefix twice -> recover the values

It is the discrete version of:

acceleration -> velocity -> position

Or in array terms:

change in slope -> slope -> value

At that point:

ans[h][w]

means:

number of rectangles of height h whose maximum possible width is exactly w

But the problem asks:

number of rectangles of height h and width exactly w

Those are different.

```cpp
int N, M;

void solve() {
    cin >> N >> M;
    vector<vector<int>> maxWidth(N + 1, vector<int>(M + 1, 0)); // maxWidth[i][j] = maximum width of a building with top edge at row i and left edge at column j
    vector<vector<int>> U(N + 1, vector<int>(M + 1, 0)); // U[i][j] = row of the next tree above row i in column j
    vector<vector<int>> D(N + 1, vector<int>(M + 1, 0)); // D[i][j] = row of the next tree below row i in column j
    vector<queue<int>> cols(M);
    for (int i = 0; i < N; ++i) {
        string row;
        cin >> row; 
        for (int j = M - 1; j >= 0; --j) {
            if (row[j] == '*') {
                maxWidth[i][j] = 0;
            } else {
                maxWidth[i][j] = maxWidth[i][j + 1] + 1; // if current cell is empty, then the maximum width is 1 + maximum width of the cell to the right
            }
        }
    }
    stack<int> stk;
    for (int j = 0; j < M; ++j) {
        while (!stk.empty()) stk.pop();
        for (int i = 0; i < N; ++i) {
            while (!stk.empty() && maxWidth[i][j] < maxWidth[stk.top()][j]) stk.pop();
            U[i][j] = stk.empty() ? 0 : stk.top() + 1;
            stk.emplace(i);
        }
        while (!stk.empty()) stk.pop();
        for (int i = N - 1; i >= 0; --i) {
            while (!stk.empty() && maxWidth[i][j] <= maxWidth[stk.top()][j]) stk.pop();
            D[i][j] = stk.empty() ? N - 1 : stk.top() - 1;
            stk.emplace(i);
        }
    }
    vector<vector<int>> ans(N + 3, vector<int>(M + 3, 0)); // ans[i][j] = number of buildings with top edge at row i and left edge at column j
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            int w = maxWidth[i][j];
            int upcnt = i - U[i][j] + 1; // number of rows above row i that can be the top edge of a building with left edge at column j and width w
            int downcnt = D[i][j] - i + 1; // number of rows below row i that can be the bottom edge of a building with left edge at column j and
            ans[1][w]++;
            ans[upcnt + 1][w]--;
            ans[downcnt + 1][w]--;
            ans[upcnt + downcnt + 1][w]++;
        }
    }
    for (int j = 1; j <= M; ++j) {
        for (int i = 1; i <= N; ++i) {
            ans[i][j] += ans[i - 1][j];
        }
        for (int i = 1; i <= N; ++i) {
            ans[i][j] += ans[i - 1][j];
        }
    }
    for (int i = 1; i <= N; ++i) {
        for (int j = M; j > 1; --j) {
            ans[i][j - 1] += ans[i][j];
        }
    }
    for (int i = 1; i <= N; ++i) {
        for (int j = 1; j <= M; ++j) {
            cout << ans[i][j] << " ";
        }
        cout << endl;
    }
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Stick Divisions

### Solution 1:

```cpp

```

## Stick Difference

### Solution 1:

```cpp

```

## Coding Company

### Solution 1:

```cpp

```

## Two Stacks Sorting

### Solution 1:

```cpp

```