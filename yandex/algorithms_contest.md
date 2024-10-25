# Algorithms Contest

# Qualification Round 2024

## A. Two out of Three

### Solution 1:  number theory, lowest common multiple, inclusion exclusion principle

You have to subtract lcm(a, b, c) * 3, because these will appear in all the numbers, but it has to be exactly 2 of them that divide into a number.  So if all three do you need to subtract it from each pair.

```cpp
const int INF = 1e18;
int a, b, c, n;

int lcm(int x, int y) {
    return (x / gcd(x, y)) * y;
}

void solve() {
    cin >> a >> b >> c >> n;
    int l1 = lcm(a, b), l2 = lcm(b, c), l3 = lcm(a, c), l4 = lcm(l1, c);
    int lo = 1, hi = INF;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        int cnt = mid / l1 + mid / l2 + mid / l3 - 3 * (mid / l4);
        if (cnt < n) lo = mid + 1;
        else hi = mid;
    }
    cout << lo << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    // freopen("input.txt", "r", stdin);
    // freopen("output.txt", "w", stdout);
    solve();
    return 0;
}

```

## B. Hills, Greeks, Gags

### Solution 1:  prefix and suffix counts

```cpp
const int INF = 1e18;
int N;
vector<int> arr, pref, suf;

void solve() {
    cin >> N;
    arr.resize(N);
    for (int i = 0; i < N; i++) cin >> arr[i];
    pref.assign(N, 0);
    suf.assign(N, 0);
    for (int i = 1; i < N; i++) {
        if (arr[i] > arr[i - 1]) pref[i] = pref[i - 1] + 1;
    }
    for (int i = N - 2; i >= 0; i--) {
        if (arr[i] > arr[i + 1]) suf[i] = suf[i + 1] + 1;
    }
    int ans = 0;
    for (int i = 1; i < N - 1; i++) {
        if (arr[i] > arr[i - 1] && arr[i] > arr[i + 1]) ans += (pref[i] * suf[i]);
    }
    cout << ans << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    // freopen("input.txt", "r", stdin);
    // freopen("output.txt", "w", stdout);
    int T;
    cin >> T;
    while (T--) solve();
    return 0;
}
```

## C. Ancient Dances

### Solution 1:  greedy

```cpp
const int INF = 1e18;
string S;

void solve() {
    cin >> S;
    int ans = 0, l = 0, p = 0;
    // replace all ? with R
    for (char ch : S) {
        if (ch == 'L') p--;
        else p++;
        l = min(l, p);
        ans = max(ans, p + abs(l));
    }
    // replace all ? with L
    l = p = 0;
    for (char ch : S) {
        if (ch == 'R') p--;
        else p++;
        l = min(l, p);
        ans = max(ans, p + abs(l));
    }
    cout << ans << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    // freopen("input.txt", "r", stdin);
    // freopen("output.txt", "w", stdout);
    solve();
    // int T;
    // cin >> T;
    // while (T--) solve();
    return 0;
}
```

## D. The Legend of Icarus

### Solution 1:  binary search, and powers of two

```cpp
const int INF = 1e18;

bool ask(int x) {
    cout << x << endl;
    cout.flush();
    string s;
    cin >> s;
    cout.flush();
    return s == "ok";
}

void answer(int x) {
    cout << "! " << x << endl;
    cout.flush();
}

void solve() {
    int lo = 1, hi = 1;
    while (!ask(hi)) {
        lo = hi;
        hi *= 2LL;
    }
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (ask(mid)) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    answer(lo);
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    // freopen("input.txt", "r", stdin);
    // freopen("output.txt", "w", stdout);
    solve();
    // int T;
    // cin >> T;
    // while (T--) solve();
    return 0;
}
```

## E. Binary Weights of the Maya

### Solution 1:  recursive dynamic programming

too slow, time comlexity of this is like O(N*LOG*M).  that is ignoring the operations on the map.  

```cpp
const int BITS = 25, MOD = 1e9 + 7;
int N, M;
vector<int> A, P;
map<pair<int, int>, int> dp;

int dfs(int i, int sum) {
    if (sum == N) {
        return 1;
    }
    if (i == BITS) return 0;
    if (dp.count({i, sum})) return dp[{i, sum}];
    int ans = 0;
    for (int j = 0; j < M; j++) {
        if (sum + A[j] * P[i] > N) break;
        ans = (ans + dfs(i + 1, sum + A[j] * P[i])) % MOD;
    }
    return dp[{i, sum}] = ans;
}

void solve() {
    cin >> N >> M;
    A.resize(M);
    for (int i = 0; i < M; i++) {
        cin >> A[i];
    }
    sort(A.begin(), A.end());
    P.resize(BITS);
    P[0] = 1;
    for (int i = 1; i < BITS; i++) {
        P[i] = (P[i - 1] * 2LL) % MOD;
    }
    int ans = dfs(0, 0);
    cout << ans << endl;

}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    // freopen("input.txt", "r", stdin);
    // freopen("output.txt", "w", stdout);
    solve();
    // int T;
    // cin >> T;
    // while (T--) solve();
    return 0;
}
```