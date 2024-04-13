# Codeforces Round 936 Div 2

## C. Tree Cutting

### Solution 1: 

```cpp
const int MAXN = 1e5 + 5;
int N, K, cut, cnt;
vector<vector<int>> adj;

int dfs(int u, int p, int x) {
    int sz = 1;
    for (int v : adj[u]) {
        if (v == p) continue;
        int csz = dfs(v, u, x);
        if (csz >= x && cnt < K) {
            cut += csz;
            cnt++;
        } else {
            sz += csz;
        }
    }
    return sz;
}

void solve() {
    cin >> N >> K;
    adj.assign(N, vector<int>());
    for (int i = 0; i < N - 1; i++) {
        int u, v;
        cin >> u >> v;
        u--, v--;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    int l = 1, r = N;
    while (l < r) {
        int m = (l + r + 1) / 2;
        cut = 0;
        cnt = 0;
        dfs(0, -1, m);
        if (N - cut < m || cnt < K) {
            r = m - 1;
        } else {
            l = m;
        }
    }
    cout << l << endl;
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

## 

### Solution 1: 

```py

```

## 

### Solution 1: 

```py

```

## E. Girl Permutation

### Solution 1:  combinatorics, modular inverse, permutations, combinations

```py
MAXN = 2 * int(1e5) + 5
mod = int(1e9) + 7

def mod_inverse(v):
    return pow(v, mod - 2, mod)

def factorials(n):
    fact, inv_fact = [1] * (n + 1), [0] * (n + 1)
    for i in range(2, n + 1):
        fact[i] = (fact[i - 1] * i) % mod
    inv_fact[-1] = mod_inverse(fact[-1])
    for i in reversed(range(n)):
        inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % mod
    return fact, inv_fact

def choose(n, k):
    return (fact[n] * inv_fact[k] * inv_fact[n - k]) % mod

def main():
    n, m1, m2 = map(int, input().split())
    pmax = list(map(int, input().split()))
    smax = list(map(int, input().split()))
    ans = choose(n - 1, smax[0] - 1)
    if smax[0] != pmax[-1] or smax[-1] != n or pmax[0] != 1: ans = 0
    for i in range(m1 - 2, -1, -1):
        num = pmax[i + 1] - 1
        left, right = pmax[i], pmax[i + 1] - pmax[i] - 1
        perm = fact[right]
        ans = (ans * choose(num - 1, left - 1) * perm) % mod
    for i in range(1, m2):
        num = n - smax[i - 1]
        left, right = smax[i] - smax[i - 1] - 1, n - smax[i]
        perm = fact[left]
        ans = (ans * choose(num - 1, right) * perm) % mod
    print(ans)
    
if __name__ == '__main__':
    fact, inv_fact = factorials(MAXN)
    T = int(input())
    for _ in range(T):
        main()    
```

## 

### Solution 1: 

```py

```

## 

### Solution 1: 

```py

```

