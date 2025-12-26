# Sparse Tables

## Step 1: Square Root Decomposition

Square root decomposition is the simplest “preprocess to answer fast” pattern

### A. Range Minimum Query

#### Solution 1: range minimum query, square root decomposition

Calculate range minimum query using square root decomposition

```cpp

```

### B. Catapult That Ball

#### Solution 1: 

Calculate the range maximum query using square root decomposition

```cpp

```

## Step 2: Sparse Tables

### A. Range Minimum Query (Sparse Table)

#### Solution 1: range minimum query sparse table

Very straight forward just apply the sparse table structure for calculating range minimum queries

```cpp
int N, Q;
vector<int> A;

template <class T, class Op>
struct SparseTable {
    vector<int> lg;          // floor(log2(i))
    vector<vector<T>> st;    // st[k][i] = combine over [i, i + 2^k)

    SparseTable() = default;
    explicit SparseTable(const vector<T>& a) { build(a); }

    void build(const vector<T>& a) {
        int n = a.size();
        lg.assign(n + 1, 0);
        for (int i = 2; i <= n; ++i) lg[i] = lg[i / 2] + 1;

        int K = (n == 0) ? 0 : (lg[n] + 1);
        st.assign(K, vector<T>(n));
        if (n == 0) return;

        st[0] = a;
        for (int k = 1; k < K; ++k) {
            int len = 1 << k;
            int half = len >> 1;
            for (int i = 0; i + len <= n; ++i) {
                st[k][i] = Op::combine(st[k - 1][i], st[k - 1][i + half]);
            }
        }
    }

    // Query on half-closed interval [l, r]
    // Precondition: 0 <= l <= r <= n
    T query(int l, int r) const {
        int len = r - l + 1;
        int k = lg[len];
        // Two blocks cover [l,r], potentially overlapping: OK only if idempotent.
        return Op::combine(st[k][l], st[k][r - (1 << k) + 1]);
    }
};

struct MinOp {
    static int combine(int a, int b) { return std::min(a, b); }
};

void solve() {
    cin >> N;
    A.assign(N, 0);
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
    }
    SparseTable<int, MinOp> st(A);
    cin >> Q;
    while (Q--) {
        int l, r;
        cin >> l >> r;
        int ans = st.query(l, r);;
        cout << ans << endl; 
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

### B. Friends and Subsequences

#### Solution 1: binary search, range minimum and maximum query sparse table, monotonicity of max/min as you increase r

Use a min and max sparse table to answer the range minimum and maximum queries respectively.

Take each left endpoint l, and you want to run a query to find the r1 and r2, where basically [r1, r2] it is the case that max(A[l...r]) - min(A[l...r]) == 0, this works because max is weakly increasing and min is weakly decreasing as r increases, and so the max - min is also weakly increasing as r increases.  So you can binary search for the lower bound r1 where max - min >= 0, and upper bound r2 where max - min <= 0.

Then the number of valid subsequences starting at l is r2 - r1.

```cpp
int N;
vector<int> A, B;

template <class T, class Op>
struct SparseTable {
    vector<int> lg;          // floor(log2(i))
    vector<vector<T>> st;    // st[k][i] = combine over [i, i + 2^k)

    SparseTable() = default;
    explicit SparseTable(const vector<T>& a) { build(a); }

    void build(const vector<T>& a) {
        int n = a.size();
        lg.assign(n + 1, 0);
        for (int i = 2; i <= n; ++i) lg[i] = lg[i / 2] + 1;

        int K = (n == 0) ? 0 : (lg[n] + 1);
        st.assign(K, vector<T>(n));
        if (n == 0) return;

        st[0] = a;
        for (int k = 1; k < K; ++k) {
            int len = 1 << k;
            int half = len >> 1;
            for (int i = 0; i + len <= n; ++i) {
                st[k][i] = Op::combine(st[k - 1][i], st[k - 1][i + half]);
            }
        }
    }

    // Query on half-open interval [l, r]
    // Precondition: 0 <= l < r <= n
    T query(int l, int r) const {
        int len = r - l + 1;
        int k = lg[len];
        // Two blocks cover [l,r], potentially overlapping: OK only if idempotent.
        return Op::combine(st[k][l], st[k][r - (1 << k) + 1]);
    }
};

struct MinOP {
    static int combine(int x, int y) { return min(x, y); }
};

struct MaxOP {
    static int combine(int x, int y) { return max(x, y); }
};

SparseTable<int, MaxOP> stmax;
SparseTable<int, MinOP> stmin;

int delta(int l, int r) {
    return stmax.query(l, r) - stmin.query(l, r);
}

int search1(int l) {
    int lo = l, hi = N;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (delta(l, mid) < 0) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

int search2(int l) {
    int lo = l, hi = N;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (delta(l, mid) <= 0) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

void solve() {
    cin >> N;
    A.resize(N);
    B.resize(N);

    for (int i = 0; i < N; ++i) {
        cin >> A[i];
    }
    for (int i = 0; i < N; ++i) {
        cin >> B[i];
    }
    stmax.build(A);
    stmin.build(B);
    int64 ans = 0;
    for (int l = 0; l < N; ++l) {
        int r1 = search1(l), r2 = search2(l);
        ans += r2 - r1;
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


### C. Strip

#### Solution 1: 

Use two deques to maintain the maximum and minimum for a sliding window, and precompute an array L, where L[i] = the largest j such that the subarray A[j...i] has max - min <= s. 



Use dynamic programming with a range, that is you will want to answer the minum value over a range from index l to index r for the dp state.  So you want to take the minimum over a range and update current dp state with that plus 1.  So this calculates the answer with dynamic programming.  I would use a segment tree for optimizing the minmum range queries, given there will be point updates.  So a static RMQ structure like sparse table will not work here.

other words the transition state is dp[i] = min(dp[L[i] - 1] ... dp[i - l]) + 1

base case dp[0] = 0

```cpp

```

### D. High Cry

#### Solution 1: 

You can find the previous greater than element and next greater than or equal element using montonic queues or stacks.

You also can calculate the last index where the b bit is set using a simple loop from right to left.

And instead of counting the intervals that satisfy the constraint, you can count the intervals that do not satisfy the constraint and subtract from total intervals.

Then for each element you have the following interval that does not satisfy the constraint where the bitwise or is equal to the maximum element in that interval.  Given an element at index i, with value v, the interval is [max(prevGreater[i], prevBitSet[i]) + 1, min(nextGreaterEqual[i] - 1, nextBitSet[i] - 1)] what is the length of that interval.  Count the number of possible subarrays that can be formed with that length that include the element at index i.  This can be calculated as (i - left + 1) * (right - i + 1) where left and right are the bounds of the interval.

```cpp

```

### E. Drazil and Park

#### Solution 1: 

two different trees

convert the circular problem into a linear problem by duplicating the array, double the circle trick

l = next(b), r = next(a), all allowed trees lie in this clockwise path from l to r, if it is the case that r < l then r += N, will work with the doubled array trick, and now that is the interval you want to solve for.

You need to calculate the max($2h_i + p_i + 2h_j - p_j$) for all i < j in the range [l, r], where h and p are given arrays.

If you split that into two separate parts, and say A[i] = 2h_i + p_i and B[i] = 2h_i - p_i, then the expression becomes max(A[i] + B[j]) for all i < j in [l, r].

You can do this with a sparse table if you maintain your state as the following, that is for each node you represent the maxA and maxB values, as well as the answer for that segment which is maxA + maxB.

Then when you merge two nodes, you can calculate the new answer as the maximum of the left answer, right answer, and left maxA + right maxB. (because A must come from i and B must come from j, and i < j, so A must come from the left segment and B from the right segment when merging two segments)




```cpp

```

### F. Range GCD Query

#### Solution 1: sparse table for range GCD query

Precompute the GCDs for all ranges of lengths that are powers of two, and answer each query using two overlapping ranges.

Basically you can use the same idea as RMQ sparse table, but instead of min use gcd function.

```cpp
int N, Q;
vector<int> A;

template <class T, class Op>
struct SparseTable {
    vector<int> lg;          // floor(log2(i))
    vector<vector<T>> st;    // st[k][i] = combine over [i, i + 2^k)

    SparseTable() = default;
    explicit SparseTable(const vector<T>& a) { build(a); }

    void build(const vector<T>& a) {
        int n = a.size();
        lg.assign(n + 1, 0);
        for (int i = 2; i <= n; ++i) lg[i] = lg[i / 2] + 1;

        int K = (n == 0) ? 0 : (lg[n] + 1);
        st.assign(K, vector<T>(n));
        if (n == 0) return;

        st[0] = a;
        for (int k = 1; k < K; ++k) {
            int len = 1 << k;
            int half = len >> 1;
            for (int i = 0; i + len <= n; ++i) {
                st[k][i] = Op::combine(st[k - 1][i], st[k - 1][i + half]);
            }
        }
    }

    // Query on half-closed interval [l, r]
    // Precondition: 0 <= l <= r <= n
    T query(int l, int r) const {
        int len = r - l + 1;
        int k = lg[len];
        // Two blocks cover [l,r], potentially overlapping: OK only if idempotent.
        return Op::combine(st[k][l], st[k][r - (1 << k) + 1]);
    }
};

struct GcdOp {
    static int combine(int a, int b) { return std::gcd(a, b); }
};

void solve() {
    cin >> N;
    A.assign(N, 0);
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
    }
    SparseTable<int, GcdOp> st(A);
    cin >> Q;
    while (Q--) {
        int l, r;
        cin >> l >> r;
        int ans = st.query(l, r);
        cout << ans << endl;
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

## Step 3: Disjoint Sparse Tables

### A. Product on the Segment by Modulo

#### Solution 1: 

Very straightforward disjoint sparse table implementation for range product queries modulo p.

```cpp

```

### B. Non-decreasing Subsequences

#### Solution 1: 

dp[v] is the number of subsequences that end with value v, since only 1,...,K

dp[x] = 1 + dp[x] + sum(dp[1] ... dp[x]) for each element x in the array.




```cpp

```