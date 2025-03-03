# Bay Area Programming Contest

# Bay Area Programming Contest 2025

## Grids on Grids

### Solution 1: frequency counting, map, combinatorics

```cpp
int N, M;
map<int, int64> freq;

int map2Dto1D(int r, int c) {
    return r * M + c;
}

int64 chooseTwo(int64 n) {
    return n * (n - 1) / 2;
}

void solve() {
    cin >> N >> M;
    freq.clear();
    for (int i = 0; i < N; i++) {
        int mask = 0;
        for (int r = 0; r < M; r++) {
            for (int c = 0; c < M; c++) {
                char ch;
                cin >> ch;
                int shift = map2Dto1D(r, c);
                if (ch == 'X') {
                    mask |= (1 << shift);
                }
            }
        }
        freq[mask]++;
    }
    int64 ans = 0;
    for (const auto &[m, c] : freq) {
        ans += chooseTwo(c);
        if (m == 0) continue;
        ans += c * freq[0];
    }
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

## A times B

### Solution 1: grid, prefix sums, prefix square sums, sum of natural numbers

```cpp
const int MAXN = 1e6 + 5;
int64 a, b, x, y, t;

int64 summation(int n) {
    if (n < 0) return 0;
    return 1LL * n * (n + 1) / 2;
}

int64 squares[MAXN], pref[MAXN];

void solve() {
    cin >> a >> b >> x >> y;
    int64 ans = a * b;
    // part 1
    if (a < b) {
        int64 d1 = summation(min(b, x)) - summation(a);
        ans += b * d1;
        a = min(b, x);
    } else {
        int64 d1 = summation(min(a, y)) - summation(b);
        ans += a * d1;
        b = min(a, y);
    }
    // part 2
    int l = max(a, b), r = min(x, y);
    if (l < r) {
        int64 d1 = squares[r] - squares[l];
        int64 d2 = pref[r] - pref[l];
        ans += d1 + d2;
        a = r;
        b = r;
    }
    // part 3
    if (x < y) {
        int64 d1 = summation(y) - summation(b);
        ans += a * d1;
    } else {
        int64 d1 = summation(x) - summation(a);
        ans += b * d1;
    }
    cout << ans << endl;
}
signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    squares[0] = 0;
    pref[0] = 0;
    for (int64 i = 1; i < MAXN; i++) {
        squares[i] = squares[i - 1] + i * i;
        pref[i] = pref[i - 1] + i * (i - 1);
    }
    int T;
    cin >> T;
    while (T--) {
        solve();
    }
    return 0;
}
```

## Even Even Odd Odd

### Solution 1: parity counting, difference

```cpp
int N;
vector<int> A, B, delta;

void solve() {
    cin >> N;
    A.assign(N, 0);
    B.assign(N, 0);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    for (int i = 0; i < N; i++) {
        cin >> B[i];
    }
    delta.assign(N, 0);
    int nEven = 0, nOdd = 0, cntEven = 0, cntOdd = 0;
    for (int i = 0; i < N; i++) {
        delta[i] = B[i] - A[i];
        if (A[i] % 2 == 0) cntEven++;
        else cntOdd++;
        if (!delta[i]) continue;
        if (delta[i] % 2 == 0) nEven++;
        else nOdd++;
    }
    if (nOdd + nEven == 0) {
        cout << "YES" << endl;
        return;
    }
    if (cntEven % 2 == 0 && cntOdd % 2 == 0) { // (even, even) => (even, odd)
        nEven = 0;
    } else if (cntEven % 2 == 0 && cntOdd % 2 == 1) { // (even, even) => (odd, odd)
        nOdd = max(0, nOdd - 1);
        nEven = 0;
    } else if (cntEven % 2 == 1 && cntOdd % 2 == 1) { // (odd, even) => (odd, odd)
        nOdd--;
        nEven = 0;
    }
    if (nEven + nOdd == 0) {
        cout << "YES" << endl;
    } else {
        cout << "NO" << endl;
    }
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

## Heaps of Queries

### Solution 1: 

```cpp

```

## In the News

### Solution 1: line sweep, intervals, counting active intervals, max heap, sorting

```cpp
int N, M;
vector<int> req;
vector<vector<int>> presenters;

void solve() {
    cin >> N >> M;
    req.assign(N, 0);
    presenters.assign(N, vector<int>());
    for (int i = 0; i < N; i++) {
        cin >> req[i];
        for (int j = 0; j < req[i]; j++) {
            int x;
            cin >> x;
            presenters[i].emplace_back(x);
        }
        sort(presenters[i].begin(), presenters[i].end());
    }
    vector<vector<int>> endpoints(N, vector<int>());
    for (int i = 0; i < M; i++) {
        int l, r;
        cin >> l >> r;
        l--, r--;
        endpoints[l].emplace_back(r);
    }
    for (const vector<int> &arr : presenters) {
        for (int i = 0; i < arr.size(); i++) {
            if (arr[i] <= i) {
                cout << "NO" << endl;
                return;
            }
        }
    }
    int activeCount = 0, inactiveCount = 0;
    vector<int> endCounts(N + 1, 0), counts(N + 1, 0);
    priority_queue<int> maxheap;
    for (int i = 0; i < N; i++) {
        for (int r : endpoints[i]) {
            endCounts[r]++;
            activeCount++;
            maxheap.emplace(r);
        }
        int delta = activeCount - inactiveCount;
        if (delta < req[i]) {
            cout << "NO" << endl;
            return;
        }
        for (int j = 0; j < req[i]; j++) {
            int r = maxheap.top();
            maxheap.pop();
            counts[r]++;
            inactiveCount++;
        }
        activeCount -= endCounts[i];
        inactiveCount -= counts[i];
    }
    cout << "YES" << endl;
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

## Count Pairs

### Solution 1: dynamic programming, bit manipulation, greedy, sorting

1. I get WA, but I think I'm really close with this solution. 

```cpp
const int BITS = 30;
int N;
vector<int> A;
int dp[BITS];

bool isSet(int mask, int i) {
    return (mask >> i) & 1;
}

void output(int* arr) {
    for (int i = 0; i < BITS; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
}

void solve() {
    cin >> N;
    A.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    sort(A.rbegin(), A.rend());
    memset(dp, 0, sizeof(dp));
    for (int x : A) {
        int b = log2(x);
        bool found = false;
        for (int i = BITS - 2; i >= 0; i--) {
            dp[i] = max(dp[i], dp[i + 1]);
        }
        for (int i = b; i >= 0; i--) {
            if (!isSet(x, i)) found = true;
            if (found) dp[i + 1] = max(dp[i + 1], dp[b + 1] + 1);
        }
        dp[0] = max(dp[0], dp[b + 1] + 1);
        for (int i = BITS - 2; i >= 0; i--) {
            dp[i] = max(dp[i], dp[i + 1]);
        }
        // output(dp);
    }
    int ans = *max_element(dp, dp + BITS);
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

## Killer Cows

### Solution 1: 

```cpp

```