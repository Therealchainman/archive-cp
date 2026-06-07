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