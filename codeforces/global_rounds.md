# Codeforces Global Round 25

## C. Ticket Hoarding

### Solution 1: greedy, sorting

```py
def main():
    n, m, k = map(int, input().split())
    arr = list(map(int, input().split()))
    ans = 0
    queries = sorted([(arr[i], i) for i in range(n)])
    tarr = []
    for cost, i in queries:
        take = min(k, m)
        if take == 0: break
        ans += cost * take
        k -= take
        tarr.append((i, take))
    tarr.sort()
    pen = 0
    for _, take in tarr:
        ans += take * pen
        pen += take
    print(ans)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## D. Buying Jewels

### Solution 1: 

```py

```

## E. No Palindromes

### Solution 1:  palindromes, splitting into non-palindromes

possibly could try every partition with string hashing? 

```py
def checker(s, p):
    return not check(s[:p]) and not check(s[p:])

def check(s):
    return s == s[::-1]

def main():
    s = input()
    n = len(s)
    if all(ch == s[0] for ch in s): return print("NO")
    if not check(s):
        print("YES")
        print(1)
        print(s)
        return 
    for i in range(n):
        if s[i] != s[0]: break
    if checker(s, i + 1):
        print("YES")
        print(2)
        print(s[:i + 1], s[i + 1:])
        return
    i += 1
    if checker(s, i + 1):
        print("YES")
        print(2)
        print(s[:i + 1], s[i + 1:])
    else:
        print("NO")

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## G. Clacking Balls

### Solution 1: 

```py

```

## I. Growing Trees

### Solution 1: 

```py

```

# Codeforces Global Round 26

##

### Solution 1: 

```cpp

```

## 

### Solution 1: 

```cpp

```

## E. Shuffle

### Solution 1:  maximum independent set, dp on tree, reroot dp

```cpp
int N, ans;
vector<vector<int>> adj, dp, dpp;
vector<int> deg;

void dfs1(int u, int p) {
    dp[u][1] = 1;
    for (int v : adj[u]) {
        if (v == p) continue;
        dfs1(v, u);
        dp[u][0] += max(dp[v][0], dp[v][1]);
        dp[u][1] += dp[v][0];
    }
}

void dfs2(int u, int p) {
    int cand;
    if (deg[u] > 1) {
        cand = max(dpp[u][0] + max(dp[u][0], dp[u][1]), dpp[u][1] + dp[u][0]);
    } else {
        cand = max(dpp[u][0] + max(dp[u][0] + 1, dp[u][1]), dpp[u][1] + dp[u][0] + 1);
    }
    ans = max(ans, cand);
    for (int v : adj[u]) {
        if (v == p) continue;
        dpp[v][0] = max(dpp[u][0], dpp[u][1]) + dp[u][0] - max(dp[v][0], dp[v][1]);
        dpp[v][1] = dpp[u][0] + dp[u][1] - dp[v][0];
        dfs2(v, u);
    }
}

void solve() {
    cin >> N;
    adj.assign(N, vector<int>());
    deg.assign(N, 0);
    for (int i = 0; i < N - 1; i++) {
        int u, v;
        cin >> u >> v;
        u--, v--;
        adj[u].push_back(v);
        adj[v].push_back(u);
        deg[u]++;
        deg[v]++;
    }
    dp.assign(N, vector<int>(2, 0));
    dpp.assign(N, vector<int>(2, 0));
    ans = 0;
    dfs1(0, -1);
    dfs2(0, -1);
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

# Codeforces Global Round 27

## B. Everyone Loves Tres

### Solution 1: 

```cpp
int n;

void solve() {
    cin >> n;
    if (n == 1 || n == 3) {
        cout << -1 << endl;
        return;
    }
    if (n % 2 == 0) {
        string s = "";
        for (int i = 0; i < n - 2; i++) {
            s += '3';
        }
        s += "66";
        cout << s << endl;
    } else {
        string s = "";
        for (int i = 0; i < n - 4; i++) {
            s += '3';
        }
        s += "6366";
        cout << s << endl;
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

## C. Alya and Permutation

### Solution 1: 

```cpp
int N;

int msb_position(int n) {
    if (n == 0) return -1; 
    int pos = 0;
    while (n > 1) {
        n >>= 1;
        pos++;
    }
    return pos;
}

vector<int> solve_odd(int n) {
    vector<int> A(n);
    iota(A.begin(), A.end(), 1);
    if ((n + 1) % 4 == 0) {
        swap(A[n - 2], A[n - 3]);
    } else {
        swap(A[n - 4], A[n - 5]);
    }
    return A;
}

void solve() {
    cin >> N;
    if (N == 6) {
        cout << 7 << endl;
        vector<int> A = {1, 2, 6, 5, 3, 4};
        for (int x : A) {
            cout << x << " ";
        }
        cout << endl;
        return;
    }
    vector<int> A;
    if (N & 1) {
        A = solve_odd(N);
        cout << N << endl;
    } else {
        int b = msb_position(N);
        int last = 1 << b;
        vector<int> infix = solve_odd(last - 1);
        infix.emplace_back(last);
        vector<int> prefix(N - last);
        iota(prefix.begin(), prefix.end(), last + 1);
        prefix.insert(prefix.end(), infix.begin(), infix.end());
        A = prefix;
        int ans = A.end()[-2] | A.end()[-1];
        cout << ans << endl;
    }
    for (int x : A) {
        cout << x << " ";
    }
    cout << endl;
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

## D. Yet Another Real Number Problem

### Solution 1: 

```cpp
const int MAXN = 1e7, MOD = 1e9 + 7;
int N;
vector<int> arr, odds, powers;
int POW2[MAXN];

// position of msb
int msb(int x) {
    int cnt = 0;
    while (x > 0) {
        x >>= 1;
        cnt++;
    }
    return cnt;
}

int floor(int x, int y) {
    return x / y;
}

// how many times does it take to divide y into x
// the number of twos you'd need to make y greater than x. 
int difference(int x, int y) {
    int cnt = floor(x, y);
    cnt = msb(cnt);
    return cnt;
}

void solve() {
    cin >> N;
    arr.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> arr[i];
    }
    odds.resize(N);
    powers.assign(N, 0);
    for (int i = 0; i < N; i++) {
        int x = arr[i];
        while (x % 2 == 0) {
            x /= 2;
            powers[i]++;
        }
        odds[i] = x;
    }
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> minheap;
    int sum = 0;
    int v, p;
    for (int i = 0; i < N; i++) {
        int x = arr[i];
        while (!minheap.empty() && difference(minheap.top().first, odds[i]) <= powers[i]) {
            tie(v, p) = minheap.top();
            powers[i] += p;
            int delta = ((v * POW2[p]) % MOD - v + MOD) % MOD;
            sum = (sum - delta + MOD) % MOD;
            minheap.pop();
        }
        int add = (odds[i] * POW2[powers[i]]) % MOD;
        sum = (sum + add) % MOD;
        minheap.emplace(odds[i], powers[i]);
        cout << sum << " ";
    }
    cout << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    POW2[0] = 1;
    for (int i = 1; i < MAXN; i++) {
        POW2[i] = (POW2[i - 1] * 2) % MOD;
    }
    int T;
    cin >> T;
    while (T--) {
        solve();
    }
    return 0;
}
```

## E. Monster

### Solution 1: 

```cpp

```