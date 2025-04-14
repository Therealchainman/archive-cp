# Atcoder Regular Contest 186

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

```cpp

```

# Atcoder Regular Contest 187

## Add and Swap

### Solution 1: 

```cpp

```

## Sum of CC

### Solution 1: 

```cpp

```

# Atcoder Regular Contest 188

## Symmetric Painting

### Solution 1:  number theory, modular arithmetic, gcd, parity

1.  Really need to know the rules of gcd and modular arithmetic, and how you can move about a circle. 

```cpp
int N, K;

void solve() {
    cin >> N >> K;
    int g = gcd(N, 2 * K);
    if (g == 1 || (g == 2 && K % 2 != (K + N / 2) % 2)) {
        cout << "Yes" << endl;
    } else {
        cout << "No" << endl;
    }
}

signed main() {
    int T;
    cin >> T;
    while (T--) {
        solve();
    }
    return 0;
}
```

# Atcoder Regular Contest 189

## A - Reversi 2 

### Solution 1:  parity, double factorial, factorials, combinatorics, interleaving, multinomial theorem

1. So you have at some point I have groups of sizes s1, s2, s3 and so want to calculate number of ways to interleave these.
1. Where si really represents number of operations, number of operations is always size of sequence of same values divided by 2, floor division
1. There are some obvious cases where it is false though. 
1. Then calculating the number of ways internally for any group of equal digit can be done with double factorial, draw out example to see why.

```cpp
const int MOD = 998244353;
int N;
vector<int> A;

int inv(int i, int m) {
  return i <= 1 ? i : m - (int)(m/i) * inv(m % i, m) % m;
}

vector<int> fact, inv_fact;
void factorials(int n, int m) {
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

int multinomial(const vector<int> &items) {
    int cnt = 0;
    for (int x : items) {
        cnt += x / 2;
    }
    int ans = fact[cnt];
    for (int x : items) {
        ans = (ans * inv_fact[x / 2]) % MOD;
    }
    return ans;
}

void solve() {
    cin >> N;
    A.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    A.emplace_back(2);
    if (A[0] == 0) {
        cout << 0 << endl;
        return;
    }
    factorials(N + 1, MOD);
    int last = 0;
    vector<int> groups;
    for (int i = 1; i <= N; i++) {
        if (A[i] != A[i - 1]) {
            int sz = i - last;
            if (sz % 2 == 0) {
                cout << 0 << endl;
                return;
            }
            groups.emplace_back(sz);
            last = i;
        }
    }
    int ans = multinomial(groups);
    for (int x : groups) {
        for (int i = x - 2; i > 0; i -= 2) {
            ans = (ans * i) % MOD;
        }
    }
    cout << ans << endl;
}

signed main() {
    solve();
    return 0;
}
```

## 

### Solution 1: 

```cpp

```

# Atcoder Regular Contest 191

## Replace Digits

### Solution 1:  greedy, constructive, implementation, strings, lexicographically sorting

1. The catch me is that the last element in T if not used must still be used so put it in the least worst place in S. Which is whereevery S[i] is equal to to T.back() or if there isn't just replace the last element in S with T.back().

```cpp
int N, M;
string S, T;

int decode(char c) {
    return c - '0';
}

char encode(int i) {
    return i + '0';
}

void solve() {
    cin >> N >> M >> S >> T;
    vector<vector<int>> adj(10, vector<int>());
    for (int i = 0; i < M; i++) {
        adj[decode(T[i])].emplace_back(i);
    }
    bool lastRemains = true;
    for (int i = 0; i < N; ++i) {
        int cur = decode(S[i]);
        for (int j = 9; j > cur; --j) {
            if (!adj[j].empty()) {
                S[i] = encode(j);
                if (adj[j].back() == M - 1) lastRemains = false;
                adj[j].pop_back();
                break;
            }
        }
    }
    for (int i = 0; i < N && lastRemains; ++i) {
        if (S[i] == T.back()) {
            S[i] = T.back();
            lastRemains = false;
        }
    }
    if (lastRemains) S.back() = T.back();
    cout << S << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## XOR = MOD

### Solution 1: 

```cpp

```

# Atcoder Regular Contest 192

## 

### Solution 1: 

```cpp

```

## 

### Solution 1: 

```cpp

```

# Atcoder Regular Contest 193

## 

### Solution 1: 

```cpp

```

## 

### Solution 1: 

```cpp

```

# Atcoder Regular Contest 194 Div 2

## Operations on a Stack

### Solution 1: 

```cpp
const int64 INF = 1e18;
int N;
vector<int> A;
vector<int64> dp;

void solve() {
    cin >> N;
    A.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    dp.assign(N + 1, -INF);
    dp[0] = 0;
    for (int i = 0; i < N; i++) {
        dp[i + 1] = max(dp[i] + A[i], i > 0 ? dp[i - 1] : -INF);
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

## Minimum Cost Sort

### Solution 1: fenwick tree, sum of natural numbers, greedy, 

```cpp
int N;
vector<int> A;

int64 calc(int64 n) {
    return n * (n + 1) / 2;
}

struct FenwickTree {
    vector<int> nodes;
    int neutral = 0;

    void init(int n) {
        nodes.assign(n + 1, neutral);
    }

    void update(int idx, int val) {
        while (idx < (int)nodes.size()) {
            nodes[idx] += val;
            idx += (idx & -idx);
        }
    }

    int query(int left, int right) {
        return right >= left ? query(right) - query(left - 1) : 0;
    }

    int query(int idx) {
        int result = neutral;
        while (idx > 0) {
            result += nodes[idx];
            idx -= (idx & -idx);
        }
        return result;
    }
};

void solve() {
    cin >> N;
    A.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    int64 ans = 0;
    FenwickTree ft;
    ft.init(N);
    for (int i = 0; i < N; i++) {
        int cnt = ft.query(A[i]);
        ans += calc(i) - calc(cnt);
        ft.update(A[i], 1);
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

## Cost to Flip

### Solution 1: fenwick tree, greedy, sorting

1. Uses fenwick tree for counting and prefix sums
1. The hard part is realizing that you take the states that are of type (1, 1) and transition them from 1 -> 0 and 0 -> 1. in the decreasing order of the value, so you optimally can take the largest. 

```cpp
struct FenwickTree {
    vector<int64> nodes;
    int neutral = 0;

    void init(int n) {
        nodes.assign(n + 1, neutral);
    }

    void update(int idx, int64 val) {
        while (idx < (int)nodes.size()) {
            nodes[idx] += val;
            idx += (idx & -idx);
        }
    }

    int64 query(int left, int right) {
        return right >= left ? query(right) - query(left - 1) : 0;
    }

    int64 query(int idx) {
        int64 result = neutral;
        while (idx > 0) {
            result += nodes[idx];
            idx -= (idx & -idx);
        }
        return result;
    }
};

const int MAXN = 1e6 + 5;
int N;
vector<int> A, B, costs;

void solve() {
    cin >> N;
    A.resize(N);
    B.resize(N);
    costs.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    for (int i = 0; i < N; i++) {
        cin >> B[i];
    }
    for (int i = 0; i < N; i++) {
        cin >> costs[i];
    }
    FenwickTree ftSumX, ftSumY, ftCntX, ftCntY;
    ftSumX.init(MAXN);
    ftSumY.init(MAXN);
    ftCntX.init(MAXN);
    ftCntY.init(MAXN);
    vector<int64> X, Y, Z;
    int64 zsum = 0;
    for (int i = 0; i < N; i++) {
        if (A[i] == 1 && B[i] == 0) {
            X.emplace_back(costs[i]);
        } else if (A[i] == 0 && B[i] == 1) {
            Y.emplace_back(costs[i]);
        } else if (A[i] == 1 && B[i] == 1) {
            Z.emplace_back(costs[i]);
            zsum += costs[i];
        }
    }
    sort(X.rbegin(), X.rend());
    sort(Y.begin(), Y.end());
    int64 base = 0;
    for (int i = 0; i < X.size(); i++) {
        base += X[i] * i;
        ftSumX.update(X[i], X[i]);
        ftCntX.update(X[i], 1);
    }
    for (int i = 0; i < Y.size(); i++) {
        base += Y[i] * (Y.size() - i);
        ftSumY.update(Y[i], Y[i]);
        ftCntY.update(Y[i], 1);
    }
    int64 cnt = X.size() + Y.size();
    int64 ans = base + zsum * cnt;
    sort(Z.rbegin(), Z.rend());
    for (int i = 0; i < Z.size(); i++) {
        base += ftSumX.query(Z[i] - 1) + ftCntX.query(Z[i], MAXN) * Z[i];
        base += ftSumY.query(Z[i] - 1) + (ftCntY.query(Z[i], MAXN) + 1) * Z[i];
        cnt += 2;
        zsum -= Z[i];
        ans = min(ans, base + zsum * cnt);
        ftSumX.update(Z[i], Z[i]);
        ftCntX.update(Z[i], 1);
        ftSumY.update(Z[i], Z[i]);
        ftCntY.update(Z[i], 1);
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

# Atcoder Regular Contest 195

## 

### Solution 1: 

```cpp

```

## 

### Solution 1: 

```cpp

```

# Atcoder Regular Contest 196

## Adjacent Delete

### Solution 1: greedy, sorting, rolling median, rolling difference between larger half and smaller half

```cpp
int N;
vector<int> A;

struct MedianBalancer {
    vector<int64> result;
    multiset<int64> left, right;
    int64 leftSum, rightSum;
    void init(const vector<int>& arr, int k) {
        int N = arr.size();
        leftSum = rightSum = 0;
        result.assign(N + 1, 0);
        for (int i = 0; i < N; i++) {
            if (i % 2 == 0) {
                result[i] = rightSum - leftSum;;
            }
            add(arr[i]);
        }
    }
    void balance() {
        while (left.size() > right.size() + 1) {
            auto it = prev(left.end());
            int val = *it;
            leftSum -= val;
            left.erase(it);
            rightSum += val;
            right.insert(val);
        }
        while (left.size() < right.size()) {
            auto it = right.begin();
            int val = *it;
            rightSum -= val;
            right.erase(it);
            leftSum += val;
            left.insert(val);
        }
    }
    void add(int num) {
        if (left.empty() || num <= *prev(left.end())) {
            left.insert(num);
            leftSum += num;
        } else {
            right.insert(num);
            rightSum += num;
        }
        balance();
    }
    void remove(int num) {
        if (left.find(num) != left.end()) {
            auto it = left.find(num);
            int64 val = *it;
            leftSum -= val;
            left.erase(it);
        } else {
            auto it = right.find(num);
            int64 val = *it;
            rightSum -= val;
            right.erase(it);
        }
        balance();
    }
};

void solve() {
    cin >> N;
    A.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    int64 ans = 0;
    if (N % 2 == 0) {
        sort(A.begin(), A.end());
        for (int i = 0; i < N; i++) {
            if (i < N / 2) ans -= A[i];
            else ans += A[i];
        }
        cout << ans << endl;
        return;
    }
    MedianBalancer prefRMD;
    prefRMD.init(A, N);
    reverse(A.begin(), A.end());
    MedianBalancer suffRMD;
    suffRMD.init(A, N);
    reverse(suffRMD.result.begin(), suffRMD.result.end());
    for (int i = 0; i < N; i++) {
        ans = max(ans, prefRMD.result[i] + suffRMD.result[i + 1]);
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

# Atcoder Regular Contest 197

## 

### Solution 1: 

```cpp

```

## 

### Solution 1: 

```cpp

```

