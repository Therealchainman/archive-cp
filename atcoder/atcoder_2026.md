# AtCoder 2026

# Atcoder Beginner Contest 457

## C. Long Sequence

### Solution 1: simulation, index math, modular arithmetic

Each array A[i] is repeated C[i] times, forming one very long sequence. Instead of actually constructing that sequence, the solution keeps track of the target index K.

```cpp
int N;
int64 K;
vector<vector<int>> A;
vector<int> C;

void solve() {
    cin >> N >> K;
    K--;
    A.assign(N, vector<int>());
    for (int i = 0; i < N; ++i) {
        int L;
        cin >> L;
        A[i].resize(L);
        for (int j = 0; j < L; ++j) {
            cin >> A[i][j];
        }
    }
    C.resize(N);
    for (int i = 0; i < N; ++i) {
        cin >> C[i];
    }
    for (int i = 0; i < N; ++i) {
        int L = A[i].size();
        int64 dist = 1LL * C[i] * L;
        if (dist <= K) {
            K -= dist;
            continue;
        }
        K = K - 1LL * (K / L) * L;
        cout << A[i][K] << endl;
        return;
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

## D. Raise Minimum

### Solution 1: binary search, greedy, simulation

The goal is to maximize the minimum final value after using at most K operations.

For a guessed target value, the function possible(target) checks whether every A[i] can be raised to at least target.

For each index i, one operation can increase A[i] by i + 1, so the number of operations needed is:

ceil(target - A[i], i + 1)

Only positive deficits matter:

delta = max(0, target - A[i])

The total cost is summed across all elements. If the cost exceeds K, the target is impossible.

The predicate is monotonic:

If we can make every element at least x, then we can also make every element at least any smaller value.
If we cannot make every element at least x, then we cannot make every element at least any larger value.

Because of that, binary search finds the maximum feasible minimum value.

```cpp
const int64 INF = numeric_limits<int64>::max() / 2;
int N;
int64 K;
vector<int64> A;

int64 ceil(int64 x, int64 y) {
    return (x + y - 1) / y;
}

bool possible(int64 target) {
    int64 cost = 0;
    for (int i = 0; i < N; ++i) {
        int64 delta = max<int64>(0, target - A[i]);
        int64 ops = ceil(delta, i + 1);
        cost += ops;
        if (cost > K) return false;
    }
    return true;
}

void solve() {
    cin >> N >> K;
    A.resize(N);
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
    }
    int64 lo = 0, hi = INF;
    while (lo < hi) {
        int64 mid = lo + (hi - lo + 1) / 2;
        if (possible(mid)) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    cout << lo << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## E. Crossing Table Cloth

### Solution 1: sparse tables, range minimum queries, binary search on sorted lists, interval logic

The problem involves answering many queries about whether some crossing or connection condition exists between two positions l and r.

The solution preprocesses the given intervals in three ways:

L[l] = all right endpoints r for intervals starting at l
R[r] = all left endpoints l for intervals ending at r
A[l] = minimum right endpoint among intervals starting at l

Then:

L[i] is sorted so the code can quickly find intervals starting at i.
R[i] is sorted so the code can quickly find intervals ending at i.
A sparse table is built over A to answer range minimum queries.

The sparse table supports fast queries of the form:

min A[x] for x in [l + 1, r]

This tells whether there is an interval starting inside the query range whose right endpoint reaches far enough.

For each query [l, r], the code checks a few cases:

Case 1: Direct interval from l to r
L[l] contains r

If there is a direct interval and either there is another useful interval from l, or some interval starting inside (l, r] has right endpoint at most r, then the answer is "Yes".

Case 2: Two-interval bridge

The code finds:

L[l][i] = largest endpoint from l that is <= r
R[r][j] = smallest start point ending at r that is > l

Then it checks whether these two intervals overlap or touch:

R[r][j] <= L[l][i] + 1

If yes, the connection/crossing condition can be satisfied.

Otherwise

No valid structure was found, so the answer is "No".

The main algorithmic tools are:

Sorted vectors for fast endpoint lookup using upper_bound and lower_bound.
Sparse table for O(1) range minimum queries.
Careful interval case analysis for each query.

```cpp
const int INF = numeric_limits<int>::max();
int N, M, Q;
vector<vector<int>> L, R;

// Requirements for Op::combine:
// - associative: combine(a, combine(b,c)) == combine(combine(a,b), c)
// - idempotent: combine(x, x) == x
// Sparse table queries rely on idempotence to safely overlap intervals.

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
        if (l > r) return INF;
        int len = r - l + 1;
        int k = lg[len];
        // Two blocks cover [l,r], potentially overlapping: OK only if idempotent.
        return Op::combine(st[k][l], st[k][r - (1 << k) + 1]);
    }
};

// ----- Plug-in ops (examples) -----

struct MinOp {
    static int combine(int a, int b) { return std::min(a, b); }
};

void solve() {
    cin >> N >> M;
    L.assign(N, vector<int>());
    R.assign(N, vector<int>());
    vector<int> A(N, INF);
    for (int i = 0; i < M; ++i) {
        int l, r;
        cin >> l >> r;
        --l; --r; // 0-based indexing
        L[l].emplace_back(r);
        R[r].emplace_back(l);
        A[l] = min(A[l], r);
    }
    for (int i = 0; i < N; ++i) {
        sort(L[i].begin(), L[i].end());
        sort(R[i].begin(), R[i].end());
    }
    SparseTable<int, MinOp> st(A);
    cin >> Q;
    while (Q--) {
        int l, r;
        cin >> l >> r;
        --l; --r; // 0-based indexing
        int nl = L[l].size(), nr = R[r].size();
        int i = upper_bound(L[l].begin(), L[l].end(), r) - L[l].begin() - 1;
        int j = lower_bound(R[r].begin(), R[r].end(), l) - R[r].begin();
        if (j < nr && R[r][j] == l) j++;
        int min_r = st.query(l + 1, r);
        if (i >= 0 && L[l][i] == r && (i > 0 || min_r <= r)) {
            cout << "Yes" << endl;
            continue;
        }
        if (i >= 0 && j < nr && R[r][j] <= L[l][i] + 1) {
            cout << "Yes" << endl;
            continue;
        }
        cout << "No" << endl;
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

## F. Second Gap

### Solution 1: counting dp, reduce states, global lazy multiplier, modular inverse

The second largest position b is only needed to know the current top-two distance.

But that distance is already fixed by the input D[i+1], because the suffix P[i+1..N] was already required to be valid.

So storing b would be redundant.

The DP state:

dp[i][a]

means:

number of valid suffixes P[i..N]
where the largest value is at position a

The phrase valid suffixes is doing important work. It means the second largest is somewhere that already satisfies the required distance for that suffix. We do not need to know exactly where.

dp[i][a] = number of ways to build suffix P[i..N]
           where the largest value in the suffix is at position a

1. Track only the suffix maximum position.
2. The second maximum position can be ignored.
3. The distance D[i] forces old_max_pos = i + D[i].
4. The unchanged-top-two case is only valid when D[i] == D[i+1].
5. Bulk DP multiplication can be optimized with a global lazy coefficient.
6. When adding new values after updating the lazy coefficient, divide by the new factor so the stored representation remains correct.

```cpp
const int MOD = 998244353;
int N;
vector<int> A;

int64 inv(int i, int64 m) {
    return i <= 1 ? i : m - (m / i) * inv(m % i, m) % m;
}

void solve() {
    cin >> N;
    A.resize(N - 1);
    for (int i = 0; i < N - 1; ++i) {
        cin >> A[i];
    }
    map<int, int64> dp;
    int64 coef = 1;
    dp[N - 1] = 1;
    dp[N - 2] = 1;
    for (int i = N - 3; i >= 0; --i) {
        int j = i + A[i];
        int64 val = 0;
        if (dp.find(j) != dp.end()) {
            val = dp[j];
        }
        if (A[i] == A[i + 1]) {
            int64 c = N - i - 2;
            coef = coef * c % MOD;
            int64 add = val * inv(c, MOD) % MOD;
            dp[j] += add;
            dp[j] %= MOD;
            dp[i] += add;
            dp[i] %= MOD;
        } else {
            dp.clear();
            dp[j] = val;
            dp[i] = val;
        }
    }
    int64 ans = 0;
    for (const auto &[pos, ways] : dp) {
        ans += ways;
        ans %= MOD;
    }
    ans = ans * coef % MOD;
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

## G. Catch All Apples

### Solution 1: minimum path cover in a DAG, maximum matching, coordinate transformation, longest increasing subsequence, 2D partial order, Dilworth's theorem

Which robot should take this apple so that the future remains possible?

The cleanest name is:

Minimum path cover in a DAG

More specifically:

Minimum vertex-disjoint path cover in a directed acyclic graph

minimum path cover = N - maximum matching


So the shape is:

Sort events by time.
Build reachability edges between events.
Run bipartite matching.
Answer is:
N - maxMatching

The matching represents “this apple can be immediately followed by that apple using the same robot.”

But there is a better approach with faster time complexity cause maximum bipartite matching is like O(VE) and that can be bad for dense graphs.

# Coordinate Transformation for the Robot Problem

Each apple event is represented as `(T, X)`, where:

- `T` is the time the apple appears.
- `X` is the coordinate where the apple appears.

A robot can move at speed at most `1`.

So a robot can collect apple `i`, then apple `j`, only if:

$$
|X_i - X_j| \le T_j - T_i
$$

assuming:

$$
T_i \le T_j
$$

## Step 1: Split the absolute value

The condition:

$$
|X_i - X_j| \le T_j - T_i
$$

is equivalent to these two inequalities:

$$
X_i - X_j \le T_j - T_i
$$

and

$$
X_j - X_i \le T_j - T_i
$$

## Step 2: Rearrange the first inequality

Start with:

$$
X_i - X_j \le T_j - T_i
$$

Rearrange:

$$
T_i + X_i \le T_j + X_j
$$

Define:

$$
U = T + X
$$

Then:

$$
U_i \le U_j
$$

## Step 3: Rearrange the second inequality

Start with:

$$
X_j - X_i \le T_j - T_i
$$

Rearrange:

$$
T_i - X_i \le T_j - X_j
$$

Define:

$$
V = T - X
$$

Then:

$$
V_i \le V_j
$$

## Final transformed condition

For each event `(T, X)`, define:

$$
U = T + X
$$

$$
V = T - X
$$

Then:

$$
|X_i - X_j| \le T_j - T_i
$$

is equivalent to:

$$
U_i \le U_j
$$

and

$$
V_i \le V_j
$$

So:

$$
i \text{ can come before } j
\iff
U_i \le U_j \text{ and } V_i \le V_j
$$

## Why this helps

The original reachability condition has an absolute value:

$$
|X_i - X_j| \le T_j - T_i
$$

After the transformation, it becomes a simple 2D ordering condition:

$$
(U_i, V_i) \le (U_j, V_j)
$$

where both coordinates must be nondecreasing.

So a robot path becomes a chain:

$$
U_1 \le U_2 \le U_3 \le \cdots
$$

and

$$
V_1 \le V_2 \le V_3 \le \cdots
$$

This turns the robot scheduling problem into a partial order problem.

Why sorting helps

If you sort all apples by u, then u is already nondecreasing.

Now the only thing left to check is v.

So after sorting by u, a robot path is just a subsequence where:

v is nondecreasing

And an antichain is a subsequence where:

v is decreasing

That is why the problem collapses from “2D poset” to something LIS-like.

Find the antichain of maximum size, which is the longest decreasing subsequence in v after sorting by u.

Then the answer is:

```cpp
int N;
vector<pair<int, int>> A;

void solve() {
    cin >> N;
    A.resize(N);
    for (int i = 0; i < N; ++i) {
        int t, x;
        cin >> t >> x;
        A[i] = {t + x, t - x};
    }
    sort(A.begin(), A.end());
    vector<int> lis;
    for (auto [u, v] : A) {
        int x = -v; // turn into longest non-decreasing subsequence
        int i = lower_bound(lis.begin(), lis.end(), x) - lis.begin();
        if (i < lis.size()) {
            lis[i] = x;
        } else {
            lis.emplace_back(x);
        }
    }
    int ans = lis.size();
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

# Atcoder Beginner Contest 458

## D. Chalkboard Median

### Solution 1: multisets, balanced binary search tree, rolling median

```cpp
int X, Q;

multiset<int> low, high;

void add(int x) {
    if (low.empty() || x <= *low.rbegin()) {
        low.emplace(x);
    } else {
        high.emplace(x);
    }

    // low cannot be too large
    if (low.size() > high.size() + 1) {
        auto it = prev(low.end());
        high.emplace(*it);
        low.erase(it);
    }
    // high cannot be larger than low
    if (high.size() > low.size()) {
        auto it = high.begin();
        low.emplace(*it);
        high.erase(it);
    }
}

int median() {
    return *low.rbegin();
}

void solve() {
    cin >> X >> Q;
    add(X);
    while (Q--) {
        int x, y;
        cin >> x >> y;
        add(x);
        add(y);
        cout << median() << endl;
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

## E. Count 123

### Solution 1: combinatorial counting, stars and bars, gap method

2s create gaps.
Pick the gaps that contain 1s.
Split the 1s into those gaps.
Put all 3s into the remaining gaps.

counts the number of valid sequences where the 1s occupy exactly k of the gaps created by the 2s.

For a fixed $$k$$, let

$$
k = \text{the number of gaps that contain at least one } 1.
$$

First, place all $$X_2$$ twos:

$$
\_\,2\,\_\,2\,\_\,\cdots\,2\,\_
$$

This creates

$$
X_2 + 1
$$

gaps.

For each fixed $$k$$, we count valid sequences in three steps.

### 1. Choose the gaps that contain ones

$$
\binom{X_2+1}{k}
$$

There are $$X_2+1$$ total gaps, and we choose $$k$$ of them to contain the $$1$$'s.

### 2. Split the ones into those selected gaps

$$
\binom{X_1-1}{k-1}
$$

Each selected gap must contain at least one $$1$$. This counts the number of ways to split $$X_1$$ identical ones into $$k$$ nonempty groups.

### 3. Distribute the threes into the remaining gaps

After choosing $$k$$ gaps for the $$1$$'s, there are

$$
X_2+1-k
$$

remaining gaps for the $$3$$'s.

The $$3$$'s may be distributed among these remaining gaps, and some gaps may be empty. By stars and bars, this gives

$$
\binom{X_3 + (X_2+1-k) - 1}{(X_2+1-k)-1}
=
\binom{X_3+X_2-k}{X_2-k}.
$$

Therefore, for a fixed $$k$$, the number of valid sequences is

$$
\binom{X_2+1}{k}
\binom{X_1-1}{k-1}
\binom{X_3+X_2-k}{X_2-k}.
$$

Finally, we sum over all possible values of $$k$$:

$$
\boxed{
\sum_{k=1}^{\min(X_1,X_2)}
\binom{X_2+1}{k}
\binom{X_1-1}{k-1}
\binom{X_3+X_2-k}{X_2-k}
}
$$

In words: choose which gaps contain $$1$$'s, split the $$1$$'s into those gaps, then distribute all $$3$$'s among the remaining gaps.

```cpp
const int MOD = 998244353;
int X1, X2, X3;

int64 inv(int i, int64 m) {
  return i <= 1 ? i : m - (m / i) * inv(m % i, m) % m;
}

vector<int64> fact, inv_fact;

void factorials(int n, int64 m) {
    fact.assign(n + 1, 1);
    inv_fact.assign(n + 1, 0);
    for (int i = 2; i <= n; i++) {
        fact[i] = (fact[i - 1] * i) % m;
    }
    inv_fact.end()[-1] = inv(fact.end()[-1], m);
    for (int i = n - 1; i >= 0; i--) {
        inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % m;
    }
}

int64 choose(int n, int r, int64 m) {
    if (n < r) return 0;
    return (fact[n] * inv_fact[r] % m) * inv_fact[n - r] % m;
}

void solve() {
    cin >> X1 >> X2 >> X3;
    int N = X1 + X2 + X3;
    factorials(N, MOD);
    int64 ans = 0;
    for (int k = 1; k <= min(X1, X2); ++k) {
        int64 ways = 1;
        ways = ways * choose(X2 + 1, k, MOD) % MOD;
        ways = ways * choose(X1 - 1, k - 1, MOD) % MOD;
        ways = ways * choose(X3 + X2 - k, X2 - k, MOD) % MOD;
        ans = (ans + ways) % MOD;
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

## F. Critical Misread

### Solution 1: Aho-Corasick automaton, matrix exponentiation, counting paths in a graph

Count strings of length N that do not contain any banned word.

1. Build an Aho-Corasick automaton from the forbidden strings.
2. Mark every state as “bad” if reaching that state means some forbidden pattern has appeared.
3. Build a transition matrix between non-bad states.
4. Compute matrix^N.
5. Start from root state 0.
6. Sum all reachable safe states after N characters.

how many strings of the current length end in each Aho-Corasick automaton state, without ever having matched a forbidden string.

If I am currently in state, and I append character c, what state do I end in?

That state represents the longest suffix of the new string that is also a prefix of one of the forbidden strings.

In an Aho-Corasick trie, each inserted character can create at most one new node.
Since you can have at most 100 characters, the total number of states is 100, so the transition matrix is at most 100x100, and it works for matrix multiplication as well. 

This is just intermediate solution without matrix exponentiation, but it is the same idea.

```cpp
const int MOD = 998244353, ALPHABET_SIZE = 26;
int N, K;

// why don't I need output link?
struct Vertex {
    bool is_leaf = false;
    bool bad = false;
    int suffix_link = 0;
    int depth = 0;
    int transition[ALPHABET_SIZE];
    void init() {
        fill(begin(transition), end(transition), 0);
    }
};
vector<Vertex> trie;
void add_string(const string& s) {
    int cur = 0, depth = 0;
    for (char ch : s) {
        int c = ch - 'a';
        depth++;
        if (trie[cur].transition[c] == 0) {
            trie[cur].transition[c] = trie.size();
            Vertex v;
            v.init();
            v.depth = depth;
            trie.push_back(v);
        }
        cur = trie[cur].transition[c];
    }
    trie[cur].is_leaf = true;
}
void push_links() {
    int queue[trie.size()];
    queue[0] = 0;
    for (int state = 0, next_state = 0; state < trie.size(); state++) {
        int v = queue[state];
        int u = trie[v].suffix_link;
        if (v == 0) {
            trie[v].bad = trie[v].is_leaf;
        } else {
            trie[v].bad = trie[v].is_leaf || trie[u].bad;
        }
        for (int c = 0; c < ALPHABET_SIZE; c++) {
            int nxt = trie[v].transition[c];
            if (nxt != 0) {
                trie[nxt].suffix_link = v ? trie[u].transition[c] : 0;
                queue[++next_state] = nxt;
            } else {
                trie[v].transition[c] = trie[u].transition[c];
            }
        }
    }
}

void solve() {
    cin >> N >> K;
    trie.resize(1);
    trie[0].init();
    for (int i = 0; i < K; ++i) {
        string s;
        cin >> s;
        add_string(s);
    }
    push_links();
    int states = trie.size();
    vector<int64> dp(states, 0);
    dp[0] = 1;
    for (int len = 0; len < N; ++len) {
        vector<int64> ndp(states, 0);
        for (int state = 0; state < states; ++state) {
            if (dp[state] == 0) continue;
            if (trie[state].bad) continue;
            for (int c = 0; c < ALPHABET_SIZE; ++c) {
                int nxt = trie[state].transition[c];
                if (trie[nxt].bad) continue;
                ndp[nxt] += dp[state];
                ndp[nxt] %= MOD;
            }
        }
        swap(dp, ndp);
    }
    int64 ans = 0;
    for (int state = 0; state < states; ++state) {
        if (!trie[state].bad) {
            ans += dp[state];
            ans %= MOD;
        }
    }
    cout << ans << '\n';
}


signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

```cpp
const int MOD = 998244353, ALPHABET_SIZE = 26;
int N, K;

// why don't I need output link?
struct Vertex {
    bool is_leaf = false;
    bool bad = false;
    int suffix_link = 0;
    int depth = 0;
    int transition[ALPHABET_SIZE];
    void init() {
        fill(begin(transition), end(transition), 0);
    }
};
vector<Vertex> trie;
void add_string(const string& s) {
    int cur = 0, depth = 0;
    for (char ch : s) {
        int c = ch - 'a';
        depth++;
        if (trie[cur].transition[c] == 0) {
            trie[cur].transition[c] = trie.size();
            Vertex v;
            v.init();
            v.depth = depth;
            trie.push_back(v);
        }
        cur = trie[cur].transition[c];
    }
    trie[cur].is_leaf = true;
}
void push_links() {
    int queue[trie.size()];
    queue[0] = 0;
    for (int state = 0, next_state = 0; state < trie.size(); state++) {
        int v = queue[state];
        int u = trie[v].suffix_link;
        if (v == 0) {
            trie[v].bad = trie[v].is_leaf;
        } else {
            trie[v].bad = trie[v].is_leaf || trie[u].bad;
        }
        for (int c = 0; c < ALPHABET_SIZE; c++) {
            int nxt = trie[v].transition[c];
            if (nxt != 0) {
                trie[nxt].suffix_link = v ? trie[u].transition[c] : 0;
                queue[++next_state] = nxt;
            } else {
                trie[v].transition[c] = trie[u].transition[c];
            }
        }
    }
}

template <int M>
struct Matrix {
    int rows, cols;
    vector<vector<int64>> a;

    Matrix() : rows(0), cols(0) {}

    Matrix(int rows, int cols, int64 value = 0)
        : rows(rows), cols(cols), a(rows, vector<int64>(cols, value % M)) {}

    vector<int64>& operator[](int i) {
        return a[i];
    }

    const vector<int64>& operator[](int i) const {
        return a[i];
    }

    static Matrix identity(int n) {
        Matrix I(n, n);

        for (int i = 0; i < n; i++) {
            I[i][i] = 1;
        }

        return I;
    }

    Matrix operator*(const Matrix& other) const {
        assert(cols == other.rows);

        Matrix result(rows, other.cols);

        for (int i = 0; i < rows; i++) {
            for (int k = 0; k < cols; k++) {
                if (a[i][k] == 0) continue;

                for (int j = 0; j < other.cols; j++) {
                    result[i][j] += a[i][k] * other[k][j] % M;

                    if (result[i][j] >= M) {
                        result[i][j] -= M;
                    }
                }
            }
        }

        return result;
    }

    Matrix pow(int64 exponent) const {
        assert(rows == cols);

        Matrix base = *this;
        Matrix result = Matrix::identity(rows);

        while (exponent > 0) {
            if (exponent & 1) {
                result = result * base;
            }

            base = base * base;
            exponent >>= 1;
        }

        return result;
    }
};

void solve() {
    cin >> N >> K;
    trie.resize(1);
    trie[0].init();
    for (int i = 0; i < K; ++i) {
        string s;
        cin >> s;
        add_string(s);
    }
    push_links();
    int states = trie.size();
    Matrix<MOD> transition(states, states, 0);
    for (int state = 0; state < states; ++state) {
        if (trie[state].bad) continue;
        for (int c = 0; c < ALPHABET_SIZE; ++c) {
            int nxt = trie[state].transition[c];
            if (trie[nxt].bad) continue;
            transition[nxt][state]++;
            transition[nxt][state] %= MOD;
        }
    }
    Matrix<MOD> base(states, 1, 0);
    base[0][0] = 1;
    Matrix<MOD> sol = transition.pow(N) * base;
    int64 ans = 0;
    for (int state = 0; state < states; ++state) {
        if (!trie[state].bad) {
            ans += sol[state][0];
            ans %= MOD;
        }
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

## G. Children Yearn for the Evil Kindergarten

### Solution 1: 

```cpp

```

# Atcoder Beginner Contest 459

## C. Drop Blocks

### Solution 1: 

```cpp

int N, Q;

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
            nodes[idx] += val;
            idx += (idx & -idx);
        }
    }

    T query(int idx) {
        T result = neutral;
        while (idx > 0) {
            result += nodes[idx];
            idx -= (idx & -idx);
        }
        return result;
    }

    T query(int left, int right) {
        return right >= left ? query(right) - query(left - 1) : T(0);
    }
};

void solve() {
    cin >> N >> Q;
    FenwickTree<int> seg;
    int base = 0;
    vector<int> A(N + 1, 1);
    seg.init(Q + 1);
    seg.update(1, N);
    for (int i = 0; i < Q; ++i) {
        int t, x;
        cin >> t >> x;
        if (t == 1) {
            seg.update(A[x], -1);
            if (seg.query(A[x]) == 0) {
                base++;
            }
            seg.update(++A[x], 1);
        } else {
            int ans = seg.query(x + base + 1, Q);
            cout << ans << endl;
        }
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

## D. Adjacent Distinct String

### Solution 1: 

```cpp
string S;

void solve() {
    cin >> S;
    int N = S.size();
    vector<int> freq(26);
    for (char c : S) {
        freq[c - 'a']++;
    }
    if (any_of(freq.begin(), freq.end(), [&](int x) { return x > (N + 1)/ 2; })) {
        cout << "No" << endl;
        return;
    }
    vector<pair<int, char>> chars;
    for (int i = 0; i < 26; i++) {
        if (freq[i] > 0) {
            chars.emplace_back(freq[i], 'a' + i);
        }
    }
    sort(chars.rbegin(), chars.rend());
    string ans(N, '-');
    int idx = 0;
    for (int i = 0; i < N; i += 2) {
        ans[i] = chars[idx].second;
        chars[idx].first--;
        if (chars[idx].first == 0) {
            idx++;
        }
    }
    for (int i = 1; i < N; i += 2) {
        ans[i] = chars[idx].second;
        chars[idx].first--;
        if (chars[idx].first == 0) {
            idx++;
        }
    }
    cout << "Yes" << endl;
    cout << ans << endl;
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

## E. Select from Subtrees

### Solution 1: 

```cpp
const int MOD = 998244353, MAXN = 1e6 + 5;
int N;
vector<vector<int>> adj;
vector<int64> cnt, dp;
vector<int> C, D;

int64 inv(int i, int64 m) {
  return i <= 1 ? i : m - (m / i) * inv(m % i, m) % m;
}

vector<int64> fact, inv_fact;

void factorials(int n, int64 m) {
    fact.assign(n + 1, 1);
    inv_fact.assign(n + 1, 0);
    for (int i = 2; i <= n; i++) {
        fact[i] = (fact[i - 1] * i) % m;
    }
    inv_fact.end()[-1] = inv(fact.end()[-1], m);
    for (int i = n - 1; i >= 0; i--) {
        inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % m;
    }
}

int64 chooseLargeN(int64 n, int r, int64 m) {
    if (n < r) return 0;
    int64 numerator = 1;
    for (int i = 0; i < r; i++) {
        numerator = (numerator * ((n - i) % m)) % m;
    }
    return (numerator * inv_fact[r]) % m;
}

void dfs(int u, int p = -1) {
    cnt[u] = C[u];
    dp[u] = 1;
    for (int v : adj[u]) {
        if (v == p) continue;
        dfs(v, u);
        cnt[u] += cnt[v];
        dp[u] = dp[u] * dp[v] % MOD;
    }
    dp[u] = dp[u] * chooseLargeN(cnt[u], D[u], MOD) % MOD;
    cnt[u] -= D[u]; // take D[u] from subtree u
}

void solve() {
    cin >> N;
    C.resize(N);
    D.resize(N);
    adj.assign(N, vector<int>());
    for (int i = 1; i < N; ++i) {
        int p;
        cin >> p;
        p--;
        adj[p].emplace_back(i);
        adj[i].emplace_back(p);
    }
    for (int i = 0; i < N; ++i) {
        cin >> C[i];
    }
    for (int i = 0; i < N; ++i) {
        cin >> D[i];
    }
    cnt.assign(N, 0);
    dp.assign(N, 0);
    dfs(0);
    cout << dp[0] << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    factorials(MAXN, MOD);
    solve();
    return 0;
}
```

## -1, +1

### Solution 1: 

```cpp

```

## G. golf 2

### Solution 1: 

```cpp

```

# Atcoder Beginner Contest 460

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

##

### Solution 1: 

So after every query, you need the diameter of the subgraph induced by the black vertices, but distance is still measured in the original tree.

```cpp

```

##

### Solution 1: 

```cpp

```

# Atcoder Beginner Contest 461

## C. Variety

### Solution 1: greedy, sorting

```cpp
int N, K, M;

void solve() {
    cin >> N >> K >> M;
    vector<pair<int, int>> gems;
    for (int i = 0; i < N; ++i) {
        int c, v;
        cin >> c >> v;
        gems.emplace_back(v, c);
    }
    sort(gems.rbegin(), gems.rend());
    vector<bool> seen(N + 1, false);
    int64 ans = 0;
    for (int i = 0, j = 0, k = 0; i < N && k < K; ++i) {
        auto [v, c] = gems[i];
        if (j < M && !seen[c]) {
            seen[c] = true;
            ans += v;
            j++;
        } else if (k < K - M) {
            ans += v;
            k++;
        }
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

## D. Count Subgrid Sum K

### Solution 1: 2D subarray counting to 1D sliding window counting, prefix sums, inclusion-exclusion / cumulative counting technique

```cpp
int R, C, K;
vector<vector<int>> grid;

int64 count_at_most(const vector<int> &arr, int K) {
    if (K < 0) return 0;
    int N = arr.size();
    int64 ans = 0;
    int l = 0, sum = 0;
    for (int r = 0; r < N; ++r) {
        sum += arr[r];
        while (sum > K) {
            sum -= arr[l];
            ++l;
        }
        ans += r - l + 1;
    }
    return ans;
}

void solve() {
    cin >> R >> C >> K;
    grid.resize(R, vector<int>(C));
    for (int i = 0; i < R; ++i) {
        string s;
        cin >> s;
        for (int j = 0; j < C; ++j) {
            grid[i][j] = s[j] - '0';
        }
    }
    int64 ans = 0;
    for (int r1 = 0; r1 < R; ++r1) {
        vector<int> arr(C);
        for (int r2 = r1; r2 < R; ++r2) {
            for (int c = 0; c < C; ++c) {
                arr[c] += grid[r2][c];
            }
            ans += count_at_most(arr, K) - count_at_most(arr, K - 1);
        }
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

## E. E Liter

### Solution 1: online queries, last occurrence tracking, fenwick tree over time

Online queries

You process each operation in order and print the answer after each query.

Last occurrence tracking

For every row and column, the code stores the last time it was updated:

lastRow[x]
lastCol[x]

This matters because when the same row or column is painted again, its previous contribution must be removed or adjusted.

Fenwick tree over time

The Fenwick trees track which rows and columns are currently “active” by their most recent update time.

segRow: active rows by last update time
segCol: active columns by last update time

This allows queries like:

how many active columns were updated after this row's previous update?
how many active rows were updated after this column's previous update?

```cpp
int N, Q;

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
            nodes[idx] += val;
            idx += (idx & -idx);
        }
    }

    T query(int idx) {
        T result = neutral;
        while (idx > 0) {
            result += nodes[idx];
            idx -= (idx & -idx);
        }
        return result;
    }

    T query(int left, int right) {
        return right >= left ? query(right) - query(left - 1) : T(0);
    }
};

void solve() {
    cin >> N >> Q;
    vector<int> lastRow(N + 1, -1), lastCol(N + 1, -1);
    FenwickTree<int> segCol, segRow;
    segCol.init(Q + 1);
    segRow.init(Q + 1);
    int64 ans = 0;
    for (int i = 1; i <= Q; ++i) {
        int t, x;
        cin >> t >> x;
        if (t == 1) {
            if (lastRow[x] == -1) {
                ans += N;
            } else {
                ans += segCol.query(lastRow[x], i);
                segRow.update(lastRow[x], -1);
            }
            segRow.update(i, 1);
            lastRow[x] = i;
        } else {
            if (lastCol[x] == -1) {
                ans -= segRow.query(1, i);
            } else {
                ans -= segRow.query(lastCol[x], i);
                segCol.update(lastCol[x], -1);
            }
            segCol.update(i, 1);
            lastCol[x] = i;
        }
        cout << ans << endl;
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

## F. Total Product is N

### Solution 1: number theory, divisors, 0/1 knapsack product variation, combinatorial counting, dynamic programming

1. Every element in a good sequence must be a divisor of N.
2. We first count unordered distinct sets of divisors.
3. Product states are represented only by divisors of N.
4. dp[product][count] counts how many sets form that product.
5. score[product][count] stores the total sum of element-scores for those sets.
6. We include divisor 1 because it is a valid positive integer.
7. We iterate backwards so each divisor is used at most once.
8. Finally, each set of size b gives b! ordered sequences.

This is a product version of 0/1 knapsack. In normal sum knapsack, you might do:
dp[sum] = number of ways to form this sum
Here, we do:
dp[product][count] = number of ways to form this product using count selected divisors
We also need the score, so we keep another DP table:
score[product][count] = sum of scores over all selected sets
The count dimension is important because at the end, each unordered selected set of size count can be ordered in:
count! ways.
Since the score is just the sum of elements, every ordering of the same set has the same score.
So a selected set of size b contributes:
score_of_set * b! to the final answer.

Why iterate backwards?
Each divisor can be used at most once.
So when processing a divisor x, we must not allow the DP update using x to feed into another update using the same x.
That is why we iterate backwards over product states and backwards over count states.
This is the same reason 0/1 knapsack does:
for (sum = target; sum >= x; sum--)
instead of going forward.
Here, the product version is:
for each x:
    for product states backwards:
        for count backwards:
            transition by multiplying by x
The backwards count loop is especially important when x = 1, because multiplying by 1 does not change the product. Without the count dimension and backwards count iteration, 1 could accidentally be reused

```cpp
const int MOD = 998244353, B = 16;
int64 N;

void solve() {
    cin >> N;
    vector<int64> divisors;
    for (int64 i = 1; i * i <= N; ++i) {
        if (N % i) continue;
        divisors.emplace_back(i);
        if (i * i != N) divisors.emplace_back(N / i);
    }
    sort(divisors.begin(), divisors.end());
    int M = divisors.size();
    unordered_map<int64, int> id;
    for (int i = 0; i < M; i++) {
        id[divisors[i]] = i;
    }
    vector<int64> fact(B, 1);
    for (int i = 1; i < B; ++i) {
        fact[i] = fact[i - 1] * i % MOD;
    }
    vector<vector<int64>> dp(M, vector<int64>(B, 0)), score(M, vector<int64>(B, 0));
    dp[0][0] = 1;
    for (int i = 0; i < M; ++i) {
        for (int j = M - 1; j >= 0; --j) {
            for (int b = B - 1; b > 0; --b) {
                int64 x = divisors[i], prod = divisors[j];
                if (prod > N / x) continue;
                int64 new_prod = prod * x;
                if (N % new_prod) continue;
                int k = id[new_prod];
                dp[k][b] = (dp[k][b] + dp[j][b - 1]) % MOD;
                score[k][b] = (score[k][b] + score[j][b - 1] + dp[j][b - 1] * x) % MOD;
            }
        }
    }
    int64 ans = 0;
    for (int b = 0; b < B; ++b) {
        ans = (ans + score[M - 1][b] * fact[b]) % MOD;
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

## G. Graph Problem 2026

### Solution 1: maximum independent set in undirected graph, maximum bipartite matching, Hopcroft-Karp algorithm

```cpp
int N, M;
vector<vector<int>> adj;
vector<int> matchL, matchR, dist;

bool bfs() {
    queue<int> q;
    for (int l = 0; l < N; l++) {
        if (matchL[l] == -1) {
            dist[l] = 0;
            q.emplace(l);
        } else {
            dist[l] = -1;
        }
    }
    bool foundAugmentingPath = false;
    while (!q.empty()) {
        int l = q.front();
        q.pop();
        for (int r : adj[l]) {
            int nextL = matchR[r];

            if (nextL == -1) {
                foundAugmentingPath = true;
            } else if (dist[nextL] == -1) {
                dist[nextL] = dist[l] + 1;
                q.emplace(nextL);
            }
        }
    }
    return foundAugmentingPath;
}

bool dfs(int l) {
    for (int r : adj[l]) {
        int nextL = matchR[r];
        if (nextL == -1 || (dist[nextL] == dist[l] + 1 && dfs(nextL))) {
            matchL[l] = r;
            matchR[r] = l;
            return true;
        }
    }
    dist[l] = -1;
    return false;
}

void solve() {
    cin >> N >> M;
    adj.assign(N, vector<int>());
    matchL.assign(N, -1);
    matchR.assign(N, -1);
    dist.assign(N, -1);
    for (int i = 0; i < M; ++i) {
        int u, v;
        cin >> u >> v;
        --u, --v;
        adj[u].emplace_back(v);
        adj[v].emplace_back(u);
    }
    int matching = 0;
    while (bfs()) {
        for (int l = 0; l < N; l++) {
            if (matchL[l] == -1 && dfs(l)) {
                matching++;
            }
        }
    }
    int maxIndependentSetSize = 2 * N - matching;
    int64 ans = 1013LL * maxIndependentSetSize;
    cout << ans << endl;
}
```

# Atcoder Beginner Contest 462

## C - Not Covered Points

### Solution 1: 

```cpp
int N;
vector<int> X;

void solve() {
    cin >> N;
    X.resize(N + 1);
    for (int i = 0; i < N; ++i) {
        int x, y;
        cin >> x >> y;
        X[x] = y;
    }
    int pmin = N + 1, ans = 0;
    for (int x = 1; x <= N; ++x) {
        int y = X[x];
        if (y < pmin) ans++;
        pmin = min(pmin, y);
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

## D - Accomplice

### Solution 1: difference array, prefix sum, counting pairs, line sweep

```cpp
const int MAXN = 1e6 + 5;
int N, D;
int A[MAXN];

int64 calc(int n) {
    if (n < 2) return 0;
    return 1LL * n * (n - 1) / 2;
}

void solve() {
    cin >> N >> D;
    for (int i = 0; i < N; ++i) {
        int s, t;
        cin >> s >> t;
        if (t - s < D) continue;
        A[s]++;
        A[t - D + 1]--;
    }
    int64 ans = 0;
    for (int i = 1; i < MAXN; ++i) {
        A[i] += A[i - 1];
        ans += calc(A[i]);
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

## E - Alternating Costs

### Solution 1: parity, math, casework, greedy

```cpp
int A, B, X, Y;

int64 calc(int x, int y) {
    int delta = abs(x - y);
    int chosen = min(A, B);
    int64 t0 = 2LL * min(x, y) * chosen;
    int64 t1 = min(1LL * delta / 2 * (A + B), 2LL * delta * chosen);
    return t0 + t1;
}

void solve() {
    cin >> A >> B >> X >> Y;
    X = abs(X), Y = abs(Y);
    if (X % 2 == Y % 2) {
        int64 ans = calc(X, Y);
        cout << ans << endl;
    } else {
        int64 t0 = calc(X - 1, Y) + A;
        int64 t1 = calc(X, Y - 1) + B;
        cout << min(t0, t1) << endl;
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

## F - More ABC

### Solution 1: Langrangian Relaxation, binary search on lambda, Alien's trick, dynamic programming, counting

exactly K of something with dp, probably Alien's trick.

```cpp
const int64 INF = (int64)4e18;
int N, K;
string S;

// A DP value: the penalized cost and the number of "ABC" formed on that path.
struct St {
    int64 cost;
    int64 cnt;
};

// Pick the better of two states: minimize cost; on ties keep the LARGER count.
// Tracking the max count per tie makes count a monotonic function of lambda,
// which is what the binary search in solve() relies on.
static inline St better(const St& a, const St& b) {
    if (a.cost != b.cost) return a.cost < b.cost ? a : b;
    return a.cnt >= b.cnt ? a : b;
}

// Automaton states for matching "ABC": 0 = "", 1 = "A", 2 = "AB".
//   0 --A--> 1,   anything else --> 0
//   1 --B--> 2,   --A--> 1,   anything else --> 0
//   2 --C--> 0 (+1 ABC, reward lambda),   --A--> 1,   anything else --> 0
// At each position we may keep S[i] (cost 0) or replace it (cost 1).
pair<int64, int64> solve_penalty(int64 lambda) {
    St dp[3] = {{0, 0}, {INF, 0}, {INF, 0}};
    for (int i = 0; i < N; ++i) {
        St ndp[3] = {{INF, 0}, {INF, 0}, {INF, 0}};
        int64 cA = S[i] == 'A' ? 0 : 1;
        int64 cB = S[i] == 'B' ? 0 : 1;
        int64 cC = S[i] == 'C' ? 0 : 1;
        // cost to turn S[i] into any letter that is none of A/B/C (resets to "")
        int64 cO = (S[i] == 'A' || S[i] == 'B' || S[i] == 'C') ? 1 : 0;
        for (int s = 0; s < 3; ++s) {
            if (dp[s].cost == INF) continue;
            St cur = dp[s];
            // assign 'A' -> state 1 (from any state)
            ndp[1] = better(ndp[1], {cur.cost + cA, cur.cnt});
            if (s == 1) {
                // assign 'B' -> state 2 (advance the match)
                ndp[2] = better(ndp[2], {cur.cost + cB, cur.cnt});
            } else {
                // 'B' from state 0/2 breaks back to ""
                ndp[0] = better(ndp[0], {cur.cost + cB, cur.cnt});
            }
            if (s == 2) {
                // assign 'C' -> completes "ABC": +1 count, reward lambda
                ndp[0] = better(ndp[0], {cur.cost + cC - lambda, cur.cnt + 1});
            } else {
                // 'C' from state 0/1 just breaks back to ""
                ndp[0] = better(ndp[0], {cur.cost + cC, cur.cnt});
            }
            // assign some other letter -> state ""
            ndp[0] = better(ndp[0], {cur.cost + cO, cur.cnt});
        }
        dp[0] = ndp[0];
        dp[1] = ndp[1];
        dp[2] = ndp[2];
    }
    // The match may end in any state (a trailing "A"/"AB" is fine).
    St res = better(better(dp[0], dp[1]), dp[2]);
    return {res.cost, res.cnt};
}

void solve() {
    cin >> S >> K;
    N = S.size();
    for (int i = 2; i < N; ++i) {
        if (S[i - 2] == 'A' && S[i - 1] == 'B' && S[i] == 'C') {
            K++;
        }
    }
    // Target K is the desired total number of "ABC" occurrences.
    // At most floor(N/3) disjoint "ABC" can exist, so anything beyond is impossible.
    if (K > N / 3) {
        cout << -1 << endl;
        return;
    }
    // count(lambda) is non-decreasing; find the smallest lambda with count >= K.
    // A marginal replacement cost is at most N, so lambda never needs to exceed N.
    // what is up here, well as the penalty lambda increases
    // it will decrease for each "ABC" we formed, and since we are minimizing the dp cost
    // it will be more likely to form more "ABC" to get that minimum cost.
    // so when it is forming too many ABC, we need decrease the lambda
    // and when it is forming too few ABC, we need to increase the lambda.
    int64 lo = 0, hi = N + 1;
    while (lo < hi) {
        int64 mid = lo + (hi - lo) / 2;
        auto [penalizedCost, cnt] = solve_penalty(mid);
        if (cnt >= K) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    auto [penalizedCost, cnt] = solve_penalty(lo);
    int64 ans = penalizedCost + lo * K;
    cout << ans << endl;
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

## G. Completely Wrong

### Solution 1: combinatorics, principle of inclusion-exclusion, polynomial multiplication, number theoretic transform, counting permutations with forbidden positions, set theory, smallest to largest merging of polynomials

A full random process can be represented as a complete ordered sequence of selected balls or colors.

If your permutation is a permutation of balls, then the set is not:

P_i = G_i

It is:
C_{P_i} = G_i

Because P_i is the ball number drawn at operation i, while G_i is a color.

So if I know all the sets for each index i that are valid, then I ultimately want to compute the intersection of all those sets. 

So that is kind of our set up 

S_i is a set of permutations that satisfy the condition at index i. We want to compute the size of the intersection of all S_i.

For a fixed subset:

S={1,4,5}

you are asking:

How many draw-order permutations make operation 1, operation 4, and operation 5 gain a point?

forbidden-position permutation counting problem

So yes, saying:

total permutations minus the union of forbidden sets

is equivalent to saying:

count permutations that lie in the complement of every scoring set

So the full idea is:

1. Convert random drawing into counting permutations.
2. Use inclusion-exclusion over positions where we score.
3. Group selected positions by goal color.
4. For each color, build a polynomial describing how many scoring positions of that color we choose.
5. Multiply all color polynomials.
6. Coefficient $x^i$ gives the total signed contribution for selecting $i$ scoring positions.
7. Multiply by $(N-i)!$ for the remaining free positions.
8. Divide by $N!$.

For one fixed color $k$:

$$F_k(x) = \sum_{j=0}^{\min(X_k, Y_k)} (-1)^j \binom{Y_k}{j} \binom{X_k}{j} j!\, x^j$$

The coefficient of $x^j$ is:

$$(-1)^j \binom{Y_k}{j} \binom{X_k}{j} j!$$

This does not directly count final permutations where exactly $j$ balls of color $k$ score.

It counts the contribution for choosing $j$ positions of target color $k$ that we force to score inside inclusion-exclusion.

That distinction matters because inclusion-exclusion counts “at least these positions score,” not “exactly these positions score.”

the signed number of ways to choose $j$ scoring constraints involving color $k$, and assign actual balls of color $k$ to satisfy those constraints.

To calculate the answer, you turn your inclusion-exclusion formula into a polynomial product.

Let:

- $X_k = \#\{\text{balls with color } k\}$
- $Y_k = \#\{\text{operations } i : G_i = k\}$

For each color $k$, define:

$$F_k(x) = \sum_{j=0}^{\min(X_k, Y_k)} (-1)^j \binom{Y_k}{j} \binom{X_k}{j} j!\, x^j$$

This polynomial says: for color $k$, if I force exactly $j$ scoring positions of this color, how many signed ways are there to choose those positions and assign actual balls to them?

Then multiply all color polynomials:

$$F(x) = \prod_{k=1}^{N} F_k(x)$$

Suppose:

$$F(x) = a_0 + a_1 x + a_2 x^2 + \cdots + a_N x^N$$

Then $a_i$ represents the total signed contribution of all ways to force exactly $i$ scoring positions across all colors.

Once those $i$ scoring positions are fixed, the remaining $N-i$ balls can be placed freely in $(N-i)!$ ways.

So the number of valid permutations (permutations with zero score) is:

$$\text{valid} = \sum_{i=0}^{N} a_i\, (N-i)!$$

Finally, because every draw order is equally likely and there are $N!$ total permutations:

$$\text{answer} = \frac{\text{valid}}{N!}$$

Modulo 998244353, division by $N!$ means multiplying by the modular inverse of $N!$:

$$\text{answer} = \left(\sum_{i=0}^{N} a_i\, (N-i)!\right) \cdot (N!)^{-1} \pmod{998244353}$$

Putting it all together, the compressed closed form is:

$$\text{answer} = \frac{1}{N!} \sum_{i=0}^{N} [x^i]\!\left(\prod_{k=1}^{N} \sum_{j=0}^{\min(X_k,\,Y_k)} (-1)^j \binom{Y_k}{j}\binom{X_k}{j} j!\, x^j\right)(N-i)!$$

```cpp
const int64 MOD = 998244353;

namespace NTT{
    const int P = 998244353, R = 3, IR = (P + 1) / 3;

    int L, N, IN;
    vector<int> rev;

    int qpow(int b, int k) {
        int ret = 1;
        while(k > 0) {
            if(k & 1) ret = static_cast<int64>(ret) * b % P;
            b = static_cast<int64>(b) * b % P;
            k >>= 1;
        }
        return ret;
    }

    void initNTT(int l) {
        L = 0;                                  
        while((1 << L) < l) L ++;
        N = 1 << L;
        IN = qpow(N, P - 2);

        rev = vector<int>(N);

        for(int i = 0; i < N; i ++)
            rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (L - 1));
    }

    void NTT(vector<int> &a, bool type) {
        a.resize(N);
        for(int i = 0; i < N; i ++)
            if(i < rev[i])
                swap(a[i], a[rev[i]]);

        for(int i = 1; i < N; i *= 2) {
            int64 g = qpow(type ? R : IR, (P - 1) / (i * 2));
            for(int j = 0; j < N; j += i * 2) {
                int64 gn = 1;
                for(int k = 0; k < i; k ++, gn = gn * g % P) {
                    int x = a[j + k], y = a[i + j + k] * gn % P;
                    a[j + k] = (x + y) % P;
                    a[i + j + k] = (x - y + P) % P;
                }
            }
        }
    }

    vector<int> convolution(vector<int> a, vector<int> b) {
        int len = (int)a.size() + (int)b.size() - 1;
        NTT(a, 1);
        NTT(b, 1);
        for(int i = 0; i < N; i ++)
            a[i] = a[i] * static_cast<int64>(b[i]) % P;
        NTT(a, 0);
        a.resize(len);
        for(auto &x : a)
            x = static_cast<int64>(x) * IN % P;
        return a;
    }
}

vector<int> polynomialMultiplication(const vector<int>& a, const vector<int>& b) {
    NTT::initNTT(a.size() + b.size() - 1);
    return NTT::convolution(a, b);
}

int64 inv(int i, int64 m) {
  return i <= 1 ? i : m - (m / i) * inv(m % i, m) % m;
}

vector<int64> fact, inv_fact;

void factorials(int n, int64 m) {
    fact.assign(n + 1, 1);
    inv_fact.assign(n + 1, 0);
    for (int i = 2; i <= n; i++) {
        fact[i] = (fact[i - 1] * i) % m;
    }
    inv_fact.end()[-1] = inv(fact.end()[-1], m);
    for (int i = n - 1; i >= 0; i--) {
        inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % m;
    }
}

int64 choose(int n, int r, int64 m) {
    if (n < 0 || r < 0 || n < r) return 0;
    return (fact[n] * inv_fact[r] % m) * inv_fact[n - r] % m;
}

int N;
vector<int> C, G;

void solve() {
    cin >> N;
    C.resize(N);
    G.resize(N);
    for (int i = 0; i < N; i++) cin >> C[i];
    for (int i = 0; i < N; i++) cin >> G[i];

    // X[k] = #balls of color k, Y[k] = #operations wanting color k
    vector<int> X(N + 1, 0), Y(N + 1, 0);
    for (int i = 0; i < N; i++) X[C[i]]++;
    for (int i = 0; i < N; i++) Y[G[i]]++;

    factorials(N, MOD);

    // Build f_k(x) for each color present in both C and G.
    // f_k[j] = (-1)^j * C(X_k, j) * C(Y_k, j) * j!
    // Min-heap merge by size for O(N log^2 N).
    vector<vector<int>> polys;
    for (int k = 1; k <= N; k++) {
        if (X[k] == 0 || Y[k] == 0) continue; // f_k = [1], skip
        int d = min(X[k], Y[k]);
        vector<int> p(d + 1);
        for (int j = 0; j <= d; j++) {
            int64 v = choose(X[k], j, MOD) * choose(Y[k], j, MOD) % MOD;
            v = v * fact[j] % MOD;
            if (j & 1) v = (MOD - v) % MOD;
            p[j] = v;
        }
        polys.emplace_back(move(p));
    }

    int number_polynomials = polys.size();

    vector<int> f;
    if (polys.empty()) {
        f = {1};
    } else {
        // (size, id) min-heap over a growing pool of polynomials
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> minheap;
        for (int i = 0; i < number_polynomials; ++i) {
            minheap.emplace(polys[i].size(), i);
        }
        while (minheap.size() > 1) {
            auto [s1, i1] = minheap.top(); minheap.pop();
            auto [s2, i2] = minheap.top(); minheap.pop();
            vector<int> prod = polynomialMultiplication(polys[i1], polys[i2]);
            int id = polys.size();
            polys.push_back(move(prod));
            minheap.emplace(polys[id].size(), id);
        }
        f = polys[minheap.top().second];
    }

    // numerator = sum_i c_i * (N - i)!
    int64 numerator = 0;
    for (int i = 0; i < f.size(); ++i) {
        numerator = (numerator + 1LL * f[i] * fact[N - i]) % MOD;
    }
    // probability = numerator / N!
    int64 ans = numerator * inv_fact[N] % MOD;
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

# Atcoder Beginner Contest 463

## C - Tallest at the Moment

### Solution 1: 

```cpp
int N, Q;
vector<int> ans;
vector<pair<int, int>> heights, queries;

void solve() {
    cin >> N;
    for (int i = 0; i < N; ++i) {
        int h, l;
        cin >> h >> l;
        heights.emplace_back(h, l);
    }
    cin >> Q;
    ans.resize(Q);
    for (int i = 0; i < Q; ++i) {
        int t;
        cin >> t;
        queries.emplace_back(t, i);
    }
    sort(queries.rbegin(), queries.rend());
    int smax = 0;
    for (const auto &[t, i] : queries) {
        while (!heights.empty() && heights.back().second > t) {
            smax = max(smax, heights.back().first);
            heights.pop_back();
        }
        ans[i] = smax;
    }
    for (int x : ans) {
        cout << x << endl;
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

## D - Maximize the Gap

### Solution 1: 

```cpp
int N, K, M;
vector<int> dp, values, leftEndpoint;
unordered_map<int, int> id;

bool possible(int target) {
    dp.assign(M + 1, 0);
    for (int i = 0; i < M; ++i) {
        dp[i + 1] = dp[i];
        int d = leftEndpoint[i] - target;
        int j = upper_bound(values.begin(), values.end(), d) - values.begin();
        dp[i + 1] = max(dp[i + 1], dp[j] + 1);
    }
    return dp[M] >= K;
}

void solve() {
    cin >> N >> K;
    vector<pair<int, int>> intervals;
    for (int i = 0; i < N; ++i) {
        int l, r;
        cin >> l >> r;
        intervals.emplace_back(l, r);
        values.emplace_back(r);
    }
    sort(values.begin(), values.end());
    values.erase(unique(values.begin(), values.end()), values.end());
    M = values.size();
    for (int i = 0; i < M; ++i) {
        id[values[i]] = i;
    }
    leftEndpoint.assign(M, -1);
    for (const auto &[l, r] : intervals) {
        leftEndpoint[id[r]] = max(leftEndpoint[id[r]], l);
    }
    int lo = 0, hi = 1e9;
    while (lo < hi) {
        int mid = lo + (hi - lo + 1) / 2;
        if (possible(mid)) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    if (!lo) {
        cout << -1 << endl;
        return;
    }
    cout << lo << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## E - Roads and Gates

### Solution 1: 

A plain Dijkstra approach works here because all edge weights are positive.

However, explicitly adding every warp edge is infeasible: each city can warp to every other city, so there are $N^2$ warp edges. With $N = 2 \times 10^5$, this is far too large.

The fix is to compress warp transitions using virtual nodes.

Key idea:

Warping from city $i$ to city $j$ costs

$$
X_i + Y + X_j.
$$

Model this with two virtual nodes, `warp_in` and `warp_out`:

- city $i \to$ `warp_in` with cost $X_i$
- `warp_in \to warp_out` with cost $Y$
- `warp_out \to$ city $j$ with cost $X_j$

So instead of adding $N^2$ warp edges, we add only $2N + 1$ extra edges.

```cpp
int N, M, Y;
vector<vector<pair<int, int>>> adj;
vector<int> X;

namespace Graph {
    template<typename CostType, typename Transition>
    vector<CostType> dijkstra(int src, Transition transitionFunction) {
        priority_queue<pair<CostType, int>, vector<pair<CostType, int>>, greater<pair<CostType, int>>> minheap;
        vector<CostType> dist(N + 2, numeric_limits<CostType>::max());
        minheap.emplace(0, src);
        dist[src] = 0;
        while (!minheap.empty()) {
            auto [cost, u] = minheap.top();
            minheap.pop();
            if (dist[u] < cost) continue;
            for (auto [v, w] : adj[u]) {
                CostType ncost = transitionFunction(cost, w);
                if (dist[v] <= ncost) continue;
                dist[v] = ncost;
                minheap.emplace(ncost, v);
            }
        }
        return dist;
    }
}

void solve() {
    cin >> N >> M >> Y;
    adj.assign(N + 2, vector<pair<int, int>>());
    for (int i = 0; i < M; ++i) {
        int u, v, w;
        cin >> u >> v >> w;
        --u, --v;
        adj[u].emplace_back(v, w);
        adj[v].emplace_back(u, w);
    }
    X.resize(N);
    for (int i = 0; i < N; ++i) {
        cin >> X[i];
    }
    adj[N].emplace_back(N + 1, Y);
    adj[N + 1].emplace_back(N, Y);
    for (int i = 0; i < N; ++i) {
        adj[i].emplace_back(N, X[i]);
        adj[N + 1].emplace_back(i, X[i]);
    }
    vector<int64> dist = Graph::dijkstra<int64>(0, [](int64 cost, int w) {
        return cost + w;
    });
    for (int i = 1; i < N; ++i) {
        cout << dist[i] << ' ';
    }
    cout << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## F - Senshuraku

### Solution 1: combinatorics, probability, binomial coefficients, uniform tie-breaking, grouping, casework

The clean way to understand the equations is: first decide the final winning score, then count how many players tie at that score, then multiply by 1 / number_of_candidates.

Yes, there are only two cases:

Final champion score = W + 1
Final champion score = W

For any fixed player, the answer is:

sum over possible final top scores:
    probability this player is tied for first
    *
    probability they are selected among the tied players

If there are w champion candidates, and this player is one of them, then the chance this player is chosen is:

1 / w

So the core formula is:

answer[player] += P(player is a candidate and total candidates = w) * 1/w

Compact summary

For every possible w:

answer += P(player is candidate AND total candidates = w) * 1/w

The binomial coefficient counts how many other flexible matches produce candidates.

The power of 2 in the denominator counts how many fair coin-flip match outcomes must line up.

The 1/w is the final uniform tie-break among w candidates.

## 3. Match Groups

Every match falls into one of six groups:

\[
m_0 = (W, W)
\]

\[
m_1 = (W, W-1)
\]

\[
m_2 = (W, 0)
\]

\[
m_3 = (W-1, W-1)
\]

\[
m_4 = (W-1, 0)
\]

\[
m_5 = (0, 0)
\]

These groups are the core compression. Instead of calculating probabilities for each player directly, we calculate probabilities for each group and role.

Players in the same group and same role have identical probability.

---

## 4. General Probability Formula

For a fixed candidate count \(w\), the probability that a player wins is:

\[
P(\text{player wins})
=
P(\text{player is candidate and total candidates} = w)
\cdot
\frac{1}{w}
\]

The factor:

\[
\frac{1}{w}
\]

comes from uniform tie-breaking among the \(w\) players tied for maximum score.

So the algorithm sums:

\[
\sum_w P(\text{player is candidate and total candidates} = w)
\cdot
\frac{1}{w}
\]

---

## 5. Case 1: Champion Score is \(W+1\)

A player can finish with \(W+1\) only if they started with \(W\) and won their final match.

### Group behavior for score \(W+1\)

| Group | Match Type | Contribution to \(W+1\) candidates |
|---|---|---|
| \(m_0\) | \((W, W)\) | Always creates exactly \(1\) candidate |
| \(m_1\) | \((W, W-1)\) | Creates \(1\) candidate with probability \(\frac{1}{2}\) |
| \(m_2\) | \((W, 0)\) | Creates \(1\) candidate with probability \(\frac{1}{2}\) |
| \(m_3\) | \((W-1, W-1)\) | Creates \(0\) candidates |
| \(m_4\) | \((W-1, 0)\) | Creates \(0\) candidates |
| \(m_5\) | \((0, 0)\) | Creates \(0\) candidates |

Define:

\[
x = m_1 + m_2
\]

These are the flexible matches that may or may not create a \(W+1\) candidate.

The candidate count is:

\[
w = m_0 + r
\]

where \(r\) is the number of successful flexible matches.

Therefore:

\[
w \in [m_0, m_0 + m_1 + m_2]
\]

but we only consider:

\[
w \ge 1
\]

because a champion-score case must have at least one candidate.

---

### 5.1 Contribution from \(m_0 = (W, W)\)

In a \((W, W)\) match, one of the two players always wins and reaches \(W+1\).

For a specific player in this group:

\[
P(\text{this player reaches } W+1) = \frac{1}{2}
\]

To get total candidate count \(w\), we need:

\[
w - m_0
\]

successful flexible matches among the \(x\) matches.

The probability of that is:

\[
\binom{x}{w - m_0} \cdot \frac{1}{2^x}
\]

Therefore, the contribution for a specific player in \(m_0\) is:

\[
\boxed{
\text{contribution}_{m_0}
=
\frac{1}{2}
\cdot
\frac{\binom{x}{w - m_0}}{2^x}
\cdot
\frac{1}{w}
}
\]

Both players in \(m_0\) get the same contribution.

---

### 5.2 Contribution from \(m_1 = (W, W-1)\)

Only the \(W\) player can reach \(W+1\).

For that specific \(W\) player to be a candidate, they must win their own match.

Their match is one of the \(x\) flexible matches.

So one successful flexible match is already fixed. We now need:

\[
w - m_0 - 1
\]

more successful flexible matches from the remaining:

\[
x - 1
\]

matches.

Thus:

\[
P(\text{this player is candidate and total candidates } = w)
=
\binom{x-1}{w - m_0 - 1}
\cdot
\frac{1}{2^x}
\]

The denominator is \(2^x\) because all \(x\) flexible matches are still random, including this player's own match.

So the contribution is:

\[
\boxed{
\text{contribution}_{m_1,W}
=
\frac{\binom{x-1}{w - m_0 - 1}}{2^x}
\cdot
\frac{1}{w}
}
\]

The \(W-1\) player in \(m_1\) cannot reach \(W+1\), so they get no contribution in this case.

---

### 5.3 Contribution from \(m_2 = (W, 0)\)

This is almost identical to \(m_1\).

Only the \(W\) player can reach \(W+1\).

So:

\[
\boxed{
\text{contribution}_{m_2,W}
=
\frac{\binom{x-1}{w - m_0 - 1}}{2^x}
\cdot
\frac{1}{w}
}
\]

The \(0\) player cannot reach \(W+1\).

---

### 5.4 Groups \(m_3, m_4, m_5\)

These groups cannot create \(W+1\) candidates.

So their contribution in the \(W+1\) case is:

\[
0
\]

---

## 6. Case 2: Champion Score is \(W\)

This case is possible only if:

\[
m_0 = 0
\]

because if there is a \((W, W)\) match, one of those two players must win and reach \(W+1\).

So when:

\[
m_0 > 0
\]

the \(W\) case is impossible.

---

### Group behavior for score \(W\)

For the champion score to be exactly \(W\), no player can reach \(W+1\). Therefore, every \(W\) player must lose their final match.

| Group | Match Type | Contribution to \(W\) candidates |
|---|---|---|
| \(m_1\) | \((W, W-1)\) | If \(W\) loses, both players finish at \(W\), so \(2\) candidates |
| \(m_2\) | \((W, 0)\) | If \(W\) loses, the \(W\) player stays at \(W\), so \(1\) candidate |
| \(m_3\) | \((W-1, W-1)\) | Exactly one player wins and reaches \(W\), so \(1\) candidate |
| \(m_4\) | \((W-1, 0)\) | Creates \(1\) candidate with probability \(\frac{1}{2}\) |
| \(m_5\) | \((0, 0)\) | Creates \(0\) candidates |

Define the forced number of candidates:

\[
K = 2m_1 + m_2 + m_3
\]

These candidates are forced once all \(W\) players lose.

The group \(m_4\) is flexible. Each \((W-1,0)\) match creates one candidate if the \(W-1\) player wins.

So:

\[
w = K + r
\]

where \(r\) is the number of successful \(m_4\) matches.

Therefore:

\[
w \in [K, K + m_4]
\]

---

### Shared probability denominator

To make champion score exactly \(W\):

- In every \(m_1\) match, the \(W\) player must lose.
- In every \(m_2\) match, the \(W\) player must lose.
- In \(m_4\), some number of \(W-1\) players may win.

The total number of relevant fair coin flips is:

\[
m_1 + m_2 + m_4
\]

So the shared denominator is:

\[
2^{m_1 + m_2 + m_4}
\]

For a fixed \(w\), we need:

\[
w - K
\]

successful \(m_4\) matches.

The base probability is:

\[
\frac{\binom{m_4}{w-K}}{2^{m_1 + m_2 + m_4}}
\]

Then we multiply by:

\[
\frac{1}{w}
\]

for tie-breaking.

So define:

\[
\text{base}(w)
=
\frac{\binom{m_4}{w-K}}{2^{m_1 + m_2 + m_4}}
\cdot
\frac{1}{w}
\]

---

### 6.1 Contribution from \(m_1 = (W, W-1)\)

In this group, the \(W\) player must lose to avoid creating a \(W+1\) champion.

If the \(W\) player loses:

\[
W \rightarrow W
\]

and:

\[
W-1 \rightarrow W
\]

So both players become \(W\)-candidates.

Therefore, both players in \(m_1\) receive:

\[
\boxed{
\text{contribution}_{m_1}
=
\frac{\binom{m_4}{w-K}}{2^{m_1 + m_2 + m_4}}
\cdot
\frac{1}{w}
}
\]

---

### 6.2 Contribution from \(m_2 = (W, 0)\)

The \(W\) player must lose to avoid creating \(W+1\).

If the \(W\) player loses, they still remain at score \(W\), so they are a candidate.

The \(0\) player cannot reach \(W\).

So only the \(W\) player receives:

\[
\boxed{
\text{contribution}_{m_2,W}
=
\frac{\binom{m_4}{w-K}}{2^{m_1 + m_2 + m_4}}
\cdot
\frac{1}{w}
}
\]

The \(0\) player receives:

\[
0
\]

---

### 6.3 Contribution from \(m_3 = (W-1, W-1)\)

In a \((W-1,W-1)\) match, exactly one of the two players wins and reaches \(W\).

So the match always contributes exactly one \(W\)-candidate, but a specific player is that candidate with probability:

\[
\frac{1}{2}
\]

Therefore, each player in \(m_3\) receives:

\[
\boxed{
\text{contribution}_{m_3}
=
\frac{1}{2}
\cdot
\frac{\binom{m_4}{w-K}}{2^{m_1 + m_2 + m_4}}
\cdot
\frac{1}{w}
}
\]

---

### 6.4 Contribution from \(m_4 = (W-1, 0)\)

Only the \(W-1\) player can become a \(W\)-candidate.

For a specific \(W-1\) player in \(m_4\), their own match must be one of the successful \(m_4\) matches.

So one success is fixed.

Among the remaining:

\[
m_4 - 1
\]

matches, we need:

\[
w - K - 1
\]

more successful matches.

Thus:

\[
\boxed{
\text{contribution}_{m_4,W-1}
=
\frac{\binom{m_4 - 1}{w-K-1}}{2^{m_1 + m_2 + m_4}}
\cdot
\frac{1}{w}
}
\]

The \(0\) player receives:

\[
0
\]

---

### 6.5 Contribution from \(m_5 = (0,0)\)

Neither player can reach \(W\) or \(W+1\), so both receive:

\[
0
\]

---

## 7. Algorithm Summary

### Step 1: Find \(W\)

\[
W = \max_i A_i
\]

### Step 2: Classify each match

For each match, compress both players into one of:

\[
W,\quad W-1,\quad 0
\]

Then count how many matches belong to:

\[
m_0,m_1,m_2,m_3,m_4,m_5
\]

Also remember which group and role each player belongs to.

### Step 3: Precompute combinations

Use factorials and inverse factorials to compute:

\[
\binom{n}{r}
\]

in \(O(1)\).

Also precompute powers of two and modular inverses.

### Step 4: Process the \(W+1\) case

Let:

\[
x = m_1 + m_2
\]

Loop over:

\[
w = \max(1,m_0), \max(1,m_0)+1, \dots, m_0+x
\]

For each \(w\), update:

\[
m_0
\]

\[
m_1 \text{ W-side}
\]

\[
m_2 \text{ W-side}
\]

using the formulas above.

### Step 5: Process the \(W\) case

Only if:

\[
m_0 = 0
\]

Define:

\[
K = 2m_1 + m_2 + m_3
\]

Loop over:

\[
w = K, K+1, \dots, K+m_4
\]

For each \(w\), update:

\[
m_1 \text{ both sides}
\]

\[
m_2 \text{ W-side}
\]

\[
m_3 \text{ both sides}
\]

\[
m_4 \text{ W-1 side}
\]

### Step 6: Assign answers to players

After the group probabilities are computed, each player gets the probability associated with their group and role.

If a player is in group \(g\) and role \(s\), then:

\[
ans_i = prob[g][s]
\]

This is what makes the algorithm fast.

```cpp
const int MOD = 998244353, MAXN = 4e5 + 5;
int N, W;
vector<int> A;
vector<int64> pow2;

int64 inv(int i, int64 m) {
  return i <= 1 ? i : m - (m / i) * inv(m % i, m) % m;
}

vector<int64> fact, inv_fact;

void factorials(int n, int64 m) {
    fact.assign(n + 1, 1);
    inv_fact.assign(n + 1, 0);
    for (int i = 2; i <= n; i++) {
        fact[i] = (fact[i - 1] * i) % m;
    }
    inv_fact.end()[-1] = inv(fact.end()[-1], m);
    for (int i = n - 1; i >= 0; i--) {
        inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % m;
    }
}

int64 choose(int n, int r, int64 m) {
    if (n < 0 || r < 0 || n < r) return 0;
    return (fact[n] * inv_fact[r] % m) * inv_fact[n - r] % m;
}

// 2 represents W, 1 represents W - 1, 0 represents less than W - 1

int groupId(int x, int y) {
    if (x > y) swap(x, y);
    if (x == W && y == W) return 0;
    if (x == W - 1 && y == W) return 1;
    if (x < W - 1 && y == W) return 2;
    if (x == W - 1 && y == W - 1) return 3;
    if (x < W - 1 && y == W - 1) return 4;
    return 5;
}

void solve() {
    cin >> N;
    int M = 2 * N;
    A.resize(M);
    for (int i = 0; i < N; ++i) {
        cin >> A[2 * i] >> A[2 * i + 1];
    }
    W = *max_element(A.begin(), A.end());
    vector<int64> freq(6, 0);
    vector<pair<int64, int64>> prob(6);
    for (int i = 0; i < N; ++i) {
        int g = groupId(A[2 * i], A[2 * i + 1]);
        freq[g]++;
    }
    // W + 1 is the champion score
    for (int i = freq[0]; i <= freq[0] + freq[1] + freq[2]; ++i) {
        int x = freq[1] + freq[2];
        // contribution of m0
        int64 P = inv(2, MOD) * choose(x, i - freq[0], MOD) % MOD;
        P = P * inv(pow2[x], MOD) % MOD;
        P = P * inv(i, MOD) % MOD;
        prob[0].first = (prob[0].first + P) % MOD;
        prob[0].second = (prob[0].second + P) % MOD;
        // contribution of m1 or m2
        if (!x) continue;
        P = inv(2, MOD) * choose(x - 1, i - freq[0] - 1, MOD) % MOD;
        P = P * inv(pow2[x - 1], MOD) % MOD;
        P = P * inv(i, MOD) % MOD;
        prob[1].second = (prob[1].second + P) % MOD;
        prob[2].second = (prob[2].second + P) % MOD;
    }
    if (freq[0] == 0) {
        // W is the champion score
        int k = 2 * freq[1] + freq[2] + freq[3];
        for (int i = k; i <= k + freq[4]; ++i) {
            // contribution of m1 (W - 1, W)
            int64 x = inv(pow2[freq[1] + freq[2] + freq[4]], MOD);
            int64 P = choose(freq[4], i - k, MOD) * x % MOD;
            P = P * inv(i, MOD) % MOD;
            prob[1].first = (prob[1].first + P) % MOD;
            prob[1].second = (prob[1].second + P) % MOD;
            // contribution of m2 (0, W)
            P = choose(freq[4], i - k, MOD) * x % MOD;
            P = P * inv(i, MOD) % MOD;
            prob[2].second = (prob[2].second + P) % MOD;
            // contribution of m3 (W - 1, W - 1)
            P = choose(freq[4], i - k, MOD) * x % MOD;
            P = P * inv(i, MOD) % MOD;
            P = P * inv(2, MOD) % MOD;
            prob[3].first = (prob[3].first + P) % MOD;
            prob[3].second = (prob[3].second + P) % MOD;
            // contribution of m4 (0, W - 1)
            if (!freq[4]) continue;
            P = choose(freq[4] - 1, i - k - 1, MOD) * x % MOD;
            P = P * inv(i, MOD) % MOD;
            prob[4].second = (prob[4].second + P) % MOD;
        }
    }
    vector<int64> ans(M);
    for (int i = 0; i < N; ++i) {
        int x = 2 * i, y = 2 * i + 1;
        int g = groupId(A[x], A[y]);
        if (A[x] > A[y]) swap(x, y);
        ans[x] = prob[g].first;
        ans[y] = prob[g].second;
    }
    for (int64 x : ans) {
        cout << x << " ";
    }
    cout << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    factorials(MAXN, MOD);
    pow2.resize(MAXN);
    pow2[0] = 1;
    for (int i = 1; i < MAXN; ++i) {
        pow2[i] = pow2[i - 1] * 2 % MOD;
    }
    solve();
    return 0;
}
```

## G - Random Walk Distance

### Solution 1: 1D random walk, binomial distribution, prefix sums, Mo’s algorithm

After N steps, only the number of +1 moves matters.
Final position is 2i−N.
The final position distribution is binomial.
Absolute value can be split at X.
The expectation reduces to binomial prefix sums.
Mo’s algorithm lets you maintain those prefix sums while moving between queries.

The real “why” behind the solution is:

The random part becomes binomial, and the absolute value turns into prefix binomial sums.

Then the constraints force the final optimization:

Answer all prefix-binomial queries offline with cheap transitions.

```cpp

const int MOD = 998244353, MAXV = 2e5 + 5;
int T;
// s0 = f(curN, curM)
// s1 = g(curN, curM)
int64 s0, s1;
vector<int> N, X, M;
vector<int64> pow2;

int64 mul(int64 a, int64 b) {
    return (a * b) % MOD;
}

int block_size, curN, curM;

int64 inv(int64 a, int64 m) {
    return a <= 1 ? a : m - (m / a) * inv(m % a, m) % m;
}

vector<int64> fact, inv_fact;

void factorials(int n, int64 m) {
    fact.assign(n + 1, 1);
    inv_fact.assign(n + 1, 1);

    for (int i = 2; i <= n; i++) {
        fact[i] = fact[i - 1] * i % m;
    }

    inv_fact[n] = inv(fact[n], m);

    for (int i = n - 1; i >= 0; i--) {
        inv_fact[i] = inv_fact[i + 1] * (i + 1) % m;
    }
}

int64 choose(int n, int r, int64 m = MOD) {
    if (n < 0 || r < 0 || n < r) return 0;
    return fact[n] * inv_fact[r] % m * inv_fact[n - r] % m;
}

struct Query {
    int l, r, idx;
    Query(int l, int r, int idx) : l(l), r(r), idx(idx) {}

    bool operator<(const Query &other) const {
        int b1 = l / block_size, b2 = other.l / block_size;
        if (b1 != b2) return b1 < b2;
        if (b1 & 1) return r > other.r;
        return r < other.r;
    }
};

void addN() {
    // Move from N to N + 1.
    //
    // f(N + 1, M) = 2f(N, M) - C(N, M - 1)
    // g(N + 1, M) = 2g(N, M) + f(N, M) - M C(N, M - 1)

    int64 boundary = choose(curN, curM - 1);
    int64 oldS0 = s0;
    int64 oldS1 = s1;
    s0 = (2LL * oldS0 - boundary + MOD) % MOD;
    s1 = (mul(2, oldS1) + oldS0 - mul(curM, boundary) + MOD) % MOD;
    curN++;
}

void addM() {
    // Move from M to M + 1.
    //
    // Add the term with index M:
    // f(N, M + 1) = f(N, M) + C(N, M)
    // g(N, M + 1) = g(N, M) + M C(N, M)

    int64 c = choose(curN, curM);
    s0 = (s0 + c) % MOD;
    s1 = (s1 + mul(curM, c)) % MOD;
    curM++;
}

void removeN() {
    // Move from N to N - 1.
    //
    // This reverses addN().
    //
    // f(N, M) = (f(N + 1, M) + C(N, M - 1)) / 2
    // g(N, M) = (g(N + 1, M) - f(N, M) + M C(N, M - 1)) / 2

    curN--;
    int64 boundary = choose(curN, curM - 1);
    s0 = mul(s0 + boundary, inv(2, MOD));
    s1 = mul(s1 - s0 + mul(curM, boundary), inv(2, MOD));
}

void removeM() {
    // Move from M to M - 1.
    //
    // Remove the term with index M - 1.
    // f(N, M - 1) = f(N, M) - C(N, M - 1)
    // g(N, M - 1) = g(N, M) - (M - 1) C(N, M - 1)

    curM--;
    int64 c = choose(curN, curM);
    s0 = (s0 - c + MOD) % MOD;
    s1 = (s1 - mul(curM, c) + MOD) % MOD;
}

pair<int64, int64> getAnswer() {
    return {s0, s1};
}

vector<pair<int64, int64>> mo_s_algorithm(vector<Query> queries) {
    int Q = queries.size();
    block_size = max<int>(1, MAXV / max(1.0, sqrt((double)Q)));
    vector<pair<int64, int64>> answers(Q);
    sort(queries.begin(), queries.end());

    curN = 0, curM = 0;
    s0 = 0, s1 = 0;
    for (const Query& q : queries) {
        while (curN < q.l) addN();
        while (curN > q.l) removeN();

        while (curM < q.r) addM();
        while (curM > q.r) removeM();
        answers[q.idx] = getAnswer();
    }
    return answers;
}

void solve() {
    cin >> T;
    N.resize(T);
    X.resize(T);
    M.resize(T);
    vector<Query> queries;
    for (int i = 0; i < T; ++i) {
        cin >> N[i] >> X[i];
        M[i] = clamp((N[i] + X[i] + 1) / 2, 0, N[i]);
        queries.emplace_back(N[i], M[i], i);
    }
    vector<pair<int64, int64>> answers = mo_s_algorithm(queries);
    for (int i = 0; i < T; ++i) {
        int64 ans = 0;
        if (abs(X[i]) >= N[i]) {
            ans = abs(X[i]);
        } else {
            auto [f, g] = answers[i];
            int64 inside = (mul(N[i] + X[i], f) - mul(2, g) + MOD) % MOD;
            ans = (mul(inv(pow2[N[i] - 1], MOD), inside) - X[i] + MOD) % MOD;
        }
        cout << ans << endl;
    }
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    factorials(MAXV, MOD);
    pow2.resize(MAXV + 1, 1);
    for (int i = 0; i < MAXV; ++i) {
        pow2[i + 1] = mul(pow2[i], 2);
    }
    solve();
    return 0;
}
```