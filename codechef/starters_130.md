# Starters 130

## Append Array

### Solution 1:  prime sieve, multiplicity, binary search, preprocess

```py
def prime_sieve(lim):
    multi_sum = [0] * lim
    for i in range(2, lim):
        if multi_sum[i] > 0: continue 
        for j in range(i, lim, i):
            num = j
            while num % i == 0:
                multi_sum[j] += 1
                num //= i
    return multi_sum

import bisect
MAXN = 2_000_005
MAXM = 22

def main():
    N, M, K = map(int, input().split())
    A = list(map(int, input().split()))
    mx1 = mx2 = cur = 0
    for num in A:
        cur += num
        if msum[num] > mx1:
            mx2 = mx1
            mx1 = msum[num]
        elif msum[num] > mx2:
            mx2 = msum[num]
    multis = [0] * MAXM
    for i in range(MAXM): # iterates over sum of multiplicities
        j = bisect.bisect_right(marr[i], M) - 1
        if j < 0: continue
        multis[i] = marr[i][j]
    ans = 0
    for i in range(MAXM):
        if not multis[i]: continue
        for j in range(i + 1):
            if not multis[j]: continue
            if i >= mx1:
                m1 = i
                m2 = max(mx1, j)
            else:
                m1 = max(mx1, i)
                m2 = max(mx2, i)
            ans = max(ans, cur + multis[i] + multis[j] * (K - 1) - m1 - m2)
    print(ans)
if __name__ == "__main__":
    msum = prime_sieve(MAXN)
    marr = [[] for _ in range(MAXM)]
    for i in range(1, MAXN):
        marr[msum[i]].append(i)
    T = int(input())
    for _ in range(T):
        main()
```

## Minimize the Difference

### Solution 1:  binary search, dynammic programming, optimizations

python TLEs

```py
MOD = 998244353
def main():
    N = int(input())
    A = list(map(int, input().split()))
    M = max(A)
    l, r = M, 2 * M
    def possible(target):
        dp = [1] * (target + 1)
        for a in A:
            ndp = [0] * (target + 1)
            for i in range(target + 1):
                if i - a >= 0:
                    ndp[i] = dp[i - a]
                if i + a <= target:
                    ndp[i] |= dp[i + a]
            dp = ndp
        return any(x for x in dp)
    while l < r:
        m = (l + r) >> 1
        if possible(m):
            r = m
        else:
            l = m + 1
    dp = [1] * (l + 1)
    for a in A:
        ndp = [0] * (l + 1)
        for i in range(l + 1):
            if i - a >= 0:
                ndp[i] = (ndp[i] + dp[i - a]) % MOD
            if i + a <= l:
                ndp[i] = (ndp[i] + dp[i + a]) % MOD
        dp = ndp
    ans = 0
    for x in dp:
        ans = (ans + x) % MOD
    print(l, ans)
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

```cpp
const int MOD = 998244353;

bool possible(const vector<int>& A, int target) {
    vector<int> dp(target + 1, 1);
    vector<int> ndp(target + 1, 0);
    for (int a : A) {
        ndp.assign(target + 1, 0);
        for (int i = 0; i <= target; ++i) {
            if (i - a >= 0) {
                ndp[i] = dp[i - a];
            }
            if (i + a <= target) {
                ndp[i] |= dp[i + a];
            }
        }
        swap(dp, ndp);
    }
    return accumulate(dp.begin(), dp.end(), 0) > 0;
}

void solve() {
int N;
    cin >> N;
    vector<int> A(N);
    for (int& a : A) {
        cin >> a;
    }
    int M = *max_element(A.begin(), A.end());
    int l = M, r = 2 * M;
    while (l < r) {
        int m = (l + r) >> 1;
        if (possible(A, m)) {
            r = m;
        } else {
            l = m + 1;
        }
    }
    vector<int> dp(l + 1, 1);
    vector<int> ndp(l + 1, 0);
    for (int a : A) {
        ndp.assign(l + 1, 0);
        for (int i = 0; i <= l; ++i) {
            if (i - a >= 0) {
                ndp[i] = (ndp[i] + dp[i - a]) % MOD;
            }
            if (i + a <= l) {
                ndp[i] = (ndp[i] + dp[i + a]) % MOD;
            }
        }
        swap(dp, ndp);
    }
    int ans = 0;
    for (int x : dp) {
        ans = (ans + x) % MOD;
    }
    cout << l << " " << ans << endl;
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

## Permutation Cycle

### Solution 1:  permutation cycle, undirected graph, cycles, subset sum with dp, optimization on subset sum with dp, disjoint cycles

dynamic programming that considers taking just one of each distinct element, and calculates the minimum size of subset that equals each sum in range from 1 to N.

```cpp
const int MAXN = 1e5 + 5, INF = 1e9;
int N, K, adj[MAXN], cmap[MAXN];
vector<int> vis, cycles, freq;

void solve() {
    cin >> N >> K;
    for (int i = 0; i < N; i++) {
        cin >> adj[i];
        adj[i]--;
    }
    vis.assign(N, 0);
    cycles.clear();
    freq.assign(N + 1, 0);
    for (int i = 0; i < N; i++) {
        if (vis[i]) continue;
        int u = i;
        int clen = 0;
        vector<int> nodes;
        while (true) {
            nodes.push_back(u);
            u = adj[u];
            if (vis[u]) break;
            vis[u] = 1;
            clen++;
        }
        for (int j : nodes) cmap[j] = clen;
        freq[clen]++;
        if (freq[clen] == 1) cycles.push_back(clen);
    }
    sort(cycles.begin(), cycles.end());
    vector<vector<int>> dp1(cycles.size(), vector<int>(N + 1, INF));
    dp1[0][0] = 0;
    for (int i = 0; i < cycles.size(); i++) {
        int c = cycles[i];
        if (i == 0) {
            dp1[i][c] = 1;
            continue;
        }
        for (int j = 0; j <= N - c; j++) {
            dp1[i][j + c] = min(dp1[i][j + c], dp1[i - 1][j] + 1); // take the cycle
            dp1[i][j] = min(dp1[i][j], dp1[i - 1][j]); // don't take the cycle
        }
    }
    vector<int> dp2(N + 1, INF);
    dp2[0] = 0;
    // Now we need to do something similar but for if you take freq - 1 of each distinct element.
    for (int c : cycles) {
        int f = freq[c] - 1;
        if (f == 0) continue;
        for (int j = 0; j < c; j++) {
            int mn = INF;
            vector<array<int, 2>> v;
            int i = 0, r = 0;
            for (int k = j; k <= N; k += c, r++) {
                while (i < v.size()) {
                    auto [x, y] = v[i];
                    if (y >= r - f) break;
                    i++;
                }
                int mn = dp2[k];
                if (i < v.size()) {
                    dp2[k] = min(dp2[k], v[i][0] + r - v[i][1]);
                }
                while (v.size()) {
                    auto [x, y] = v.back();
                    if (x - y >= mn - r) v.pop_back();
                    else break;
                }
                v.push_back({mn, r});
                i = min(i, (int)v.size() - 1);
            }
        }
    }
    vector<int> dp_combined(N + 1, INF);
    // make each cycle length equal to size K
    for (int i = cycles.size() - 1; i >= 0; i--) {
        int c = cycles[i];
        int req = K - c;
        if (req < 0) continue;
        if (i == 0) {
            dp_combined[c] = dp2[req];
            continue;
        }
        for (int j = 0; j <= req; j++) {
            dp_combined[c] = min(dp_combined[c], dp1[i - 1][j] + dp2[req - j]);
        }
        // update dp2
        for (int j = N - c; j >= 0; j--) {
            dp2[j + c] = min(dp2[j + c], dp2[j] + 1);
        }
    }
    for (int c : cycles) {
        // you take greedily the greatest until you are over the subset sum target
        // and determine if that is better than taking up to the subset sum
        int req = K - c;
        int cnt = 1;
        for (int i = cycles.size() - 1; i >= 0; i--) {
            if (req <= 0) break;
            int f = freq[cycles[i]];
            if (cycles[i] == c) f--;
            for (int j = 0; j < f; j++) {
                if (req <= 0) break;
                cnt++;
                req -= cycles[i];
            }
        }
        dp_combined[c] = min(dp_combined[c], cnt);
    }
    for (int i = 0; i < N; i++) {
        cout << dp_combined[cmap[i]] << " ";
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