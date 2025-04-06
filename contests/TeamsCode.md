# TeamsCode

# TeamsCode Spring 2025 Advanced

## Cell Towers

### Solution 1: 

I think monotonic stack idea, but meh, then dp? I tried greedy

```cpp

```

## Walk

### Solution 1: bipartite graph, combinatorics

```cpp
const int64 MOD = 998244353, MAXN = 2e6 + 5;
int A, B, C, D;

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
    cin >> A >> B >> C >> D;
    int64 sum = A + B + C + D;
    if (A + D != (sum + 1) / 2) {
        cout << 0 << endl;
        return;
    }
    if (B + C != (sum - 1) / 2) {
        cout << 0 << endl;
        return;
    }
    if (A < 2) {
        cout << 0 << endl;
        return;
    }
    int64 ans = choose(A + D - 2, D, MOD) * choose(B + C, C, MOD) % MOD;
    cout << ans << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    int T;
    cin >> T;
    factorials(MAXN, MOD);
    while (T--) {
        solve();
    }
    return 0;
}
```

## Not Japanese Triangle

1. did I try hard enough? I need inspiration

### Solution 1: 

```cpp

```

## Robot Racing

### Solution 1: greedy prefix accumulation algorithm, reverse order

1. You have to see the pattern that makes this one solvable, how the size of groups is 1 2 3 k k k k
2. So whenever you add an element and you can't go to k + 1 groups you can move up the 1 2 3 and the current one so you add whatever the current k is equal which I called mul
3. And then if you can go to k + 1, you increase by all elements already processed since you increase by one group for everything. 
4. This problem is heavily simplified because the elements are sorted, so just note that and you can find the pattern.

```cpp
int N, L;
vector<int64> A;

void solve() {
    cin >> N >> L;
    A.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    reverse(A.begin(), A.end());
    int64 cur = 0, mul = 0, ans = 0;
    for (int i = 0; i < N; i++) {
        if ((mul + 1) * A[i] <= L) {
            mul++;
            cur += i + 1;
        } else {
            cur += mul;
        }
        ans += cur;
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

## 

### Solution 1: 

```cpp

```