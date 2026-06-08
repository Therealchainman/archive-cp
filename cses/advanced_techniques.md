# Advanced Techniques

## 

### Solution 1:  

```cpp

```

## 

### Solution 1:  

```cpp

```

## Houses and Schools

### Solution 1:  Dynamic programming with divide and conquer optimization

What is the main idea for solving this problem? 

We need to pick the locations of schools to minimize the distance each child has to travel. 

For each pos, calculate the nearest school, and the cost of moving those children is the number of children times the distance to the nearest school.

I think there a couple ways to solve this, but I'm really thinking divide and conquer? 

We just need to restate the dp like the following:
dp[g][m] = min(dp[g-1][j] + cost(j+1, m)) for 0 <= j < m
where dp[g][m] is the minimum cost of placing g schools among the first m houses, and cost(j+1, m) is the cost of placing a school at position m and having it serve the children from positions j+1 to m.

Then we can solve this with divide and conquer optimization, which reduces the time complexity from O(g * m^2) to O(g * m log m).
-+
Because of the properties of the cost function, we can apply divide and conquer optimization to compute the dp efficiently.

That is the best j for dp[g][i] is less than or equal to the best j for dp[g][i+1], which allows us to compute the dp row in O(m log m) time instead of O(m^2).

The tricky part for this problem is that I need to calculate for each interval cost(l, r) the best placement of school and the cost of that placement.  This would have to be precomputed. I can calculate by the fact that for each interval for a school to serve all the children in that interval, the best place to put the school is at the median of the children's positions. So I can precompute cost(l, r) by finding the median position of the children in that interval and calculating the total distance from that median to all the children in that interval. However it is the weighed median. 

```cpp
const int64 INF = numeric_limits<int64>::max();
int N, K;
vector<int> A;
vector<vector<int64>> cost;
vector<int64> dp, ndp;

void dfs(int l, int r, int optL, int optR) {
    if (l > r) return;
    int mid = l + (r - l) / 2;
    int bestJ = -1;
    int64 bestCost = INF;
    for (int j = optL; j <= min(mid, optR); ++j) {
        if (dp[j] == INF) continue;
        int64 curCost = dp[j] + cost[j][mid - 1];
        if (curCost < bestCost) {
            bestCost = curCost;
            bestJ = j;
        }
    }
    ndp[mid] = bestCost;
    dfs(l, mid - 1, optL, bestJ);
    dfs(mid + 1, r, bestJ, optR);
}

void solve() {
    cin >> N >> K;
    A.resize(N);
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
    }
    vector<int64> prefAx(N + 1, 0), prefA(N + 1, 0);
    for (int i = 0; i < N; ++i) {
        prefAx[i + 1] = prefAx[i] + 1LL * A[i] * i;
        prefA[i + 1] = prefA[i] + A[i];
    }
    cost.assign(N, vector<int64>(N, 0));
    for (int l = 0; l < N; ++l) {
        int m = l;
        int64 total = 0, pref = 0;
        for (int r = l; r < N; ++r) {
            total += A[r];
            while (m <= r && 2LL * (pref + A[m]) < total) {
                pref += A[m++];
            }
            // left side, where m > i
            // a[i] * (m - i) = a[i] * m - a[i] * i
            int64 leftSum = (prefA[m] - prefA[l]) * m - (prefAx[m] - prefAx[l]);
            // right side, where m <= i
            // a[i] * (i - m) = a[i] * i - a[i] * m
            int64 rightSum = (prefAx[r + 1] - prefAx[m]) - (prefA[r + 1] - prefA[m]) * m;
            cost[l][r] = leftSum + rightSum;
        }
    }
    dp.assign(N + 1, INF);
    dp[0] = 0;
    while (K--) {
        ndp.assign(N + 1, INF);
        dfs(1, N, 0, N - 1);
        swap(dp, ndp);
    }
    cout << dp[N] << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

### Solution 2: Alien's trick, langragian relaxation, WQS binary search, penalty method

This is an interesting approach that can solve the problem in O(N^2logC) time complexity, so this is even better in a sense, cause if K is large, then you can solve like this, although why would K be larger than N, so really only reason to solve it like this is for educational purposes. 

When Alien's trick is absolutely required is in rare situations. 

This is the implementation, but it actually TLE for this problem, but it is still a very interesting approach to solve this problem.

```cpp
const int64 INF = numeric_limits<int64>::max();
int N, K;
vector<int> A, cnt;
vector<vector<int64>> cost;
vector<int64> dp;


pair<int64, int> solve_penalty(int64 lambda) {
    dp.assign(N + 1, INF);
    cnt.assign(N + 1, 0);
    dp[0] = 0;
    for (int i = 1; i <= N; ++i) {
        for (int j = 0; j < i; ++j) {
            if (dp[j] == INF) continue;
            int64 candCost = dp[j] + cost[j][i - 1] + lambda;
            int candCnt = cnt[j] + 1;
            if (candCost < dp[i]) {
                dp[i] = candCost;
                cnt[i] = candCnt;
            } else if (candCost == dp[i]) {
                cnt[i] = min(cnt[i], candCnt);
            }
        }
    }
    return {dp[N], cnt[N]};
}

void solve() {
    cin >> N >> K;
    A.resize(N);
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
    }
    vector<int64> prefAx(N + 1, 0), prefA(N + 1, 0);
    for (int i = 0; i < N; ++i) {
        prefAx[i + 1] = prefAx[i] + 1LL * A[i] * i;
        prefA[i + 1] = prefA[i] + A[i];
    }
    cost.assign(N, vector<int64>(N, 0));
    for (int l = 0; l < N; ++l) {
        int m = l;
        int64 total = 0, pref = 0;
        for (int r = l; r < N; ++r) {
            total += A[r];
            while (m <= r && 2LL * (pref + A[m]) < total) {
                pref += A[m++];
            }
            // left side, where m > i
            // a[i] * (m - i) = a[i] * m - a[i] * i
            int64 leftSum = (prefA[m] - prefA[l]) * m - (prefAx[m] - prefAx[l]);
            // right side, where m <= i
            // a[i] * (i - m) = a[i] * i - a[i] * m
            int64 rightSum = (prefAx[r + 1] - prefAx[m]) - (prefA[r + 1] - prefA[m]) * m;
            cost[l][r] = leftSum + rightSum;
        }
    }
    int64 lo = 0, hi = cost[0][N - 1] + 1;
    while (lo < hi) {
        int64 mid = lo + (hi - lo) / 2;
        auto [val, cnt] = solve_penalty(mid);
        if (cnt > K) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    auto [penalizedCost, used] = solve_penalty(lo);
    int64 ans = penalizedCost - lo * K;
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

## 

### Solution 1:  

```cpp

```

## 

### Solution 1:  

```cpp

```

## One Bit Positions

### Solution 1:  DFT, FFT, convolution, polynomial multiplication

```cpp
typedef complex<double> cd;
const int SIZE = 1<<19;
const double PI = acos(-1);

int n, m;
vector<cd> A(SIZE), B(SIZE);

void fft(vector<cd> &a, bool invert) {
    int n = a.size();

    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;

        if (i < j)
            swap(a[i], a[j]);
    }

    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * PI / len * (invert ? -1 : 1);
        cd wlen(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len) {
            cd w(1);
            for (int j = 0; j < len / 2; j++) {
                cd u = a[i+j], v = a[i+j+len/2] * w;
                a[i+j] = u + v;
                a[i+j+len/2] = u - v;
                w *= wlen;
            }
        }
    }

    if (invert) {
        for (cd & x : a)
            x /= n;
    }
}

signed main() {
    string s;
    cin >> s;
    n = s.size();
    for (int i = 0; i < n; i++) {
        A[i] = s[i] - '0';
        B[n - i - 1] = s[i] - '0';
    }
    fft(A, false);
    fft(B, false);
    for (int i = 0; i < SIZE; i++) {
        A[i] *= B[i];
    }
    fft(A, true);
    for (int i = n; i < 2 * n - 1; i++) {
        cout << llround(A[i].real()) << " ";
    }
    cout << endl;
}
```

## Signal Processing

### Solution 1:  DFT, FFT, convolution, polynomial multiplication

```cpp
typedef complex<double> cd;
const int SIZE = 1<<19;
const double PI = acos(-1);

int n, m;
vector<cd> A(SIZE), B(SIZE);

void fft(vector<cd> &a, bool invert) {
    int n = a.size();

    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;

        if (i < j)
            swap(a[i], a[j]);
    }

    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * PI / len * (invert ? -1 : 1);
        cd wlen(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len) {
            cd w(1);
            for (int j = 0; j < len / 2; j++) {
                cd u = a[i+j], v = a[i+j+len/2] * w;
                a[i+j] = u + v;
                a[i+j+len/2] = u - v;
                w *= wlen;
            }
        }
    }

    if (invert) {
        for (cd & x : a)
            x /= n;
    }
}

signed main() {
    cin >> n >> m;
    for (int i = 0; i < n; i++) {
        cin >> A[i];
    }
    for (int i = 0; i < m; i++) {
        cin >> B[m - i - 1];
    }
    fft(A, false);
    fft(B, false);
    for (int i = 0; i < SIZE; i++) {
        A[i] *= B[i];
    }
    fft(A, true);
    for (int i = 0; i < n + m - 1; i++) {
        cout << llround(A[i].real()) << " ";
    }
    cout << endl;
}
```

## 

### Solution 1:  

```py

```

## 

### Solution 1:  

```py

```

## 

### Solution 1:  

```py

```

## 

### Solution 1:  

```py

```