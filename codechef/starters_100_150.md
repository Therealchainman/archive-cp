# Starters 124

```py
import math
def main():
    L, R = map(int, input().split())
    N = R - L + 1
    ans = [x for x in range(L, R + 1)]
    for i in range(1, N, 2):
        ans[i - 1], ans[i] = ans[i], ans[i - 1]
    if N > 1 and N & 1: ans[-2], ans[-1] = ans[-1], ans[-2]
    if all(math.gcd(ans[i], L + i) == 1 for i in range(N)):
        print(*ans)
    else:
        print(-1)

if __name__ == "__main__":
    T = int(input())
    for _ in range(T):
        main()
```

```py
import string
def main():
    S = input()
    N = len(S)
    ans = N
    for ch in string.ascii_lowercase:
        k = S.count(ch)
        if k == 0: continue
        cnt = 0
        for i in range(N):
            if S[i] == ch: cnt += 1
            if i >= k - 1:
                ans = min(ans, k - cnt)
                if S[i - k + 1] == ch: cnt -= 1
    print(ans)
    
if __name__ == "__main__":
    T = int(input())
    for _ in range(T):
        main()
```

```cpp
const int MAXN = 2e5 + 5;
int N, Q, u, v, t;
vector<vector<int>> adj;
vector<map<int, int>> freq;
int ans[MAXN], act[MAXN], score[MAXN];

void dfs(int u, int p, int depth) {
    for (int v : adj[u]) {
        if (v == p) continue;
        dfs(v, u, depth + 1);
        if (freq[v].size() > freq[u].size()) {
            swap(freq[u], freq[v]);
            swap(score[u], score[v]);
        }
        for (auto &[d, c] : freq[v]) {
            if (freq[u].find(d) == freq[u].end()) {
                freq[u][d] = 0;
            }
            if (freq[u][d] == 1) score[u]--;
            freq[u][d] += c;
            if (freq[u][d] == 1) score[u]++;
        }
    }
    ans[u] = score[u];
    if (act[u] == 1) {
        if (freq[u].find(depth) == freq[u].end()) freq[u][depth] = 0;
        freq[u][depth]++;
        if (freq[u][depth] == 1) score[u]++;
        if (freq[u][depth] == 2) score[u]--;
    }
}

void solve() {
    cin >> N >> Q;
    int u, v;
    adj.assign(N, vector<int>());
    for (int i = 0; i < N - 1; i++) {
        cin >> u >> v;
        u--; v--;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    freq.assign(N, map<int, int>());
    memset(ans, 0, sizeof(ans));
    memset(act, 0, sizeof(act));
    memset(score, 0, sizeof(score));
    int t;
    vector<int> queries;
    while (Q--) {
        cin >> t >> u;
        u--;
        if (t == 1) {
            queries.push_back(u);
        } else {
            act[u] ^= 1;
        }
    }
    dfs(0, -1, 0);
    for (int u : queries) {
        cout << ans[u] << endl;
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

```cpp
const int MAXN = 2e5 + 5, LOG = 20;
int N, Q, timer, in[MAXN], out[MAXN], depth[MAXN], max_dep, marked[MAXN], par[MAXN], last[MAXN];
int tour_depth[2 * MAXN];
vector<set<pair<int, int>>> nei;
vector<vector<int>> adj;
vector<int> tour;


int neutral = 0;
struct FenwickTree {
    vector<int> nodes;
    
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
        return query(right) - query(left - 1);
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

FenwickTree ft;

void dfs(int u, int p, int dep) {
    in[u] = ++timer;
    depth[u] = dep;
    par[u] = p;
    max_dep = max(max_dep, dep);
    tour_depth[tour.size()] = dep;
    tour.push_back(u);
    for (int v : adj[u]) {
        if (v != p) {
            dfs(v, u, dep + 1);
            tour_depth[tour.size()] = dep;
            tour.push_back(u);
        }
    }
    out[u] = ++timer;
    last[u] = tour.size() - 1;
}

struct RMQ {
    vector<vector<int>> st;
    void init() {
        int n = tour.size();
        st.assign(n, vector<int>(LOG));
        for (int i = 0; i < n; i++) {
            st[i][0] = i;
        }
        for (int j = 1; j < LOG; j++) {
            for (int i = 0; i + (1 << j) <= n; i++) {
                int x = st[i][j - 1];
                int y = st[i + (1 << (j - 1))][j - 1];
                st[i][j] = (tour_depth[x] < tour_depth[y] ? x : y);
            }
        }
    }

    int query(int a, int b) {
        int l = min(a, b), r = max(a, b);
        int j = 31 - __builtin_clz(r - l + 1);
        int x = st[l][j];
        int y = st[r - (1 << j) + 1][j];
        return (tour_depth[x] < tour_depth[y] ? x : y);
    }
};
RMQ rmq;
void update(set<pair<int, int>>::iterator it, int val) {
    int v1 = 0, v2 = 0;
    int v = it->second;
    int d = depth[v];
    if (it != nei[d].begin()) {
        it--;
        v1 = it->second;
        it++;
    }
    it++;
    if (it != nei[d].end()) {
        v2 = it->second;
    }
    int lca1 = tour[rmq.query(last[v], last[v1])], lca2 = tour[rmq.query(last[v], last[v2])];
    if (depth[lca1] < depth[lca2]) {
        swap(lca1, lca2);
    }
    v = par[v];
    ft.update(in[v], val);
    ft.update(in[lca1], -val);
}

void solve() {
    cin >> N >> Q;
    adj.assign(N + 1, vector<int>());
    for (int i = 0; i < N - 1; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    adj[0].push_back(1);
    adj[1].push_back(0);
    timer = 0; max_dep = 0;
    memset(par, 0, sizeof(par));
    dfs(0, -1, 0);
    nei.assign(max_dep + 1, set<pair<int, int>>());
    ft.init(timer);
    rmq.init();
    memset(marked, 0, sizeof(marked));
    while (Q--) {
        int t, u;
        cin >> t >> u;
        if (t == 1) {
            int resp = ft.query(in[u], out[u]);
            cout << resp << endl;
        } else {
            int d = depth[u];
            marked[u] ^= 1;
            if (marked[u]) {
                auto it = nei[d].upper_bound({in[u], 0});
                if (it != nei[d].end()) {
                    update(it, -1);
                }
                if (it != nei[d].begin()) {
                    it--;
                    update(it, -1);
                }
                nei[d].insert({in[u], u});
                it = nei[d].upper_bound({in[u], 0});
                if (it != nei[d].end()) {
                    update(it, 1);
                }
                if (it != nei[d].begin()) {
                    it--;
                    update(it, 1);
                }
                if (it != nei[d].begin()) {
                    it--;
                    update(it, 1);
                }
            } else {
                auto vit = nei[d].find({in[u], u});
                auto it = nei[d].upper_bound({in[u], 0});
                if (it != nei[d].end()) {
                    update(it, -1);
                }
                if (it != nei[d].begin()) {
                    it--;
                    update(it, -1);
                }
                if (it != nei[d].begin()) {
                    it--;
                    update(it, -1);
                }
                nei[d].erase(vit);
                it = nei[d].upper_bound({in[u], 0});
                if (it != nei[d].end()) {
                    update(it, 1);
                }
                if (it != nei[d].begin()) {
                    it--;
                    update(it, 1);
                }
            }
        }
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

# Starters 125


## Binary Minimal

### Solution 1:  greedy

```py
def main():
    N, K = map(int, input().split())
    S = list(map(int, input()))
    if S.count(1) > K:
        for i in range(N):
            if K == 0: break
            if S[i] == 1:
                S[i] = 0
                K -= 1
        print("".join(map(str, S)))
    else:
        print("0" * (N - K))

if __name__ == "__main__":
    T = int(input())
    for _ in range(T):
        main()
```

## Bucket Game

### Solution 1:  greedy

```py
def main():
    N = int(input())
    arr = list(map(int, input().split()))
    A = B = cnt = s = 0
    for num in arr:
        if num == 1:
            if A > B: B += 1
            else: A += 1
        else:
            s += num
            cnt += 1
    if A > B:
        if s & 1:
            B += cnt
        else:
            A += cnt
    else:
        if s & 1:
            A += cnt
        else:
            B += cnt
    if A == B: print("Draw")
    elif A > B: print("Alice")
    else: print("Bob")

if __name__ == "__main__":
    T = int(input())
    for _ in range(T):
        main()
```

## Operating on A

### Solution 1:  prefix sums

```py
def main():
    N = int(input())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    ans = "YES" if sum(A) == sum(B) else "NO"
    print(ans)

if __name__ == "__main__":
    T = int(input())
    for _ in range(T):
        main()
```

# Starters 127

## Expected Components

### Solution 1:  combinatorics, tree, connected components, multiplicative modular inverse, 

```py
MOD = int(1e9) + 7
def mod_inverse(x):
    return pow(x, MOD - 2, MOD)
def divide(x, y):
    return (x * y) % MOD
def main():
    N, K = map(int, input().split())
    black = [0] * N
    arr = map(int, input().split())
    for v in arr:
        v -= 1
        black[v] = 1
    cww = cwb = cbb = 0
    for _ in range(N - 1):
        u, v = map(int, input().split())
        u -= 1; v -= 1
        if black[u] and black[v]: cbb += 1
        elif black[u] or black[v]: cwb += 1
        else: cww += 1
    ans = [0] * (K + 1)
    inv_bb = mod_inverse(K * (K - 1))
    inv_wb = mod_inverse(K)
    for i in range(1, K + 1):
        pbb = divide((K - i) * (K - i - 1), inv_bb)
        pwb = divide(K - i, inv_wb)
        expected_rem_edges = (cww + cbb * pbb + cwb * pwb) % MOD
        ans[i] = (N - i - expected_rem_edges) % MOD
    print(*ans[1:])

if __name__ == "__main__":
    T = int(input())
    for _ in range(T):
        main()
```

# Starters 129

## SumXor

### Solution 1:  bit manipulation, xor, bit frequency for integers less than N, prefix and suffix product, dynamic programming, counting

```py
BITS = 30
MOD = 998244353
def count_bits(x, b):
    cycle_len = 2 ** (b + 1)
    cycle_cnt = x // cycle_len
    cycle_rem = x % cycle_len - 2 ** b + 1
    return cycle_cnt * 2 ** b + max(0, cycle_rem)

def main():
    N = int(input())
    L = list(map(int, input().split()))
    R = list(map(int, input().split()))
    # figure this out how to get bit frequency for an interval
    bfreq = [[0] * BITS for _ in range(N)]
    for i in range(N):
        for j in range(BITS):
            bfreq[i][j] = count_bits(R[i], j) % MOD
            if L[i] > 0: bfreq[i][j] = (bfreq[i][j] - count_bits(L[i] - 1, j)) % MOD
    pfreq = [0] * N
    sfreq = [0] * N
    for i in range(N):
        pfreq[i] = R[i] - L[i] + 1
        if i > 0: pfreq[i] = (pfreq[i] * pfreq[i - 1]) % MOD 
    for i in reversed(range(N)):
        sfreq[i] = R[i] - L[i] + 1
        if i < N - 1: sfreq[i] = (sfreq[i] * sfreq[i + 1]) % MOD
    ans = 0
    for b in range(BITS):
        for l in range(N):
            x = 1 # number of ways for b to not be set in last element
            y = 0 # number of ways for b to be set in last element
            for r in range(l, N):
                cnt_b = bfreq[r][b]
                cnt_no_b = R[r] - L[r] + 1 - cnt_b
                x, y = (x * cnt_no_b + y * cnt_b) % MOD, (x * cnt_b + y * cnt_no_b) % MOD
                ways = y
                if l > 0: ways = (ways * pfreq[l - 1]) % MOD
                if r < N - 1: ways = (ways * sfreq[r + 1]) % MOD
                ans = (ans + ways * (1 << b)) % MOD
    print(ans)

if __name__ == "__main__":
    T = int(input())
    for _ in range(T):
        main()
```

## Red Blue Decomposition

### Solution 1:  tree, dfs, flipping, parity

```cpp
const int RED = 0, BLUE = 1;
int N;
vector<vector<int>> adj;
vector<int> color, flip;

int dfs1(int u, int p) {
    int sz = 1;
    int act = 0;
    for (int v : adj[u]) {
        if (v == p) continue;
        int csz = dfs1(v, u);
        if (csz & 1) {
            if (act) flip[v] ^= 1;
            act ^= 1;
        }
        sz += csz;
    }
    if (sz % 2 == 0) color[u] = BLUE;
    return sz;
}

void dfs2(int u, int p, int f) {
    for (int v : adj[u]) {
        if (v == p) continue;
        dfs2(v, u, flip[v] ^ f);
    }
    if (flip[u]) color[u] ^= 1;
}

void solve() {
    cin >> N;
    adj.assign(N, {});
    for (int i = 0; i < N - 1; i++) {
        int u, v;
        cin >> u >> v;
        u--, v--;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    color.assign(N, RED);
    flip.assign(N, 0);
    dfs1(0, -1);
    dfs2(0, -1, 0);
    string ans;
    ans.reserve(N);
    for (int i = 0; i < N; i++) {
        ans += (color[i] == RED ? 'R' : 'B');
    }
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

# Starters 131

## Chef needs to return some videotapes

### Solution 1:  square root decomposition, sorted list (set), binary search, prefix sum per block

```cpp
const int MAXN = 1e5 + 5, B = 450;
int N, Q, last, t, a, b, lo, hi;
int C[MAXN], L[MAXN];
vector<int> psum, prv;
vector<int> block_order;
vector<set<int>> indices;

void recalc(int bid) {
    lo = bid * B, hi = min(N, (bid + 1) * B);
    sort(block_order.begin() + lo, block_order.begin() + hi, [&](int x, int y) {
        return prv[x] < prv[y];
    });
    for (int i = lo; i < hi; i++) {
        psum[i] = L[block_order[i]];
        if (i > lo) psum[i] += psum[i - 1];
    }
}

void solve() {
    cin >> N >> Q;
    indices.assign(N + 1, set<int>());
    for (int i = 0; i < N; i++) {
        cin >> L[i] >> C[i];
        C[i]--;
        indices[C[i]].insert(i);
    }
    prv.assign(N + 1, -1);
    last = 0;
    psum.assign(N, 0);
    block_order.resize(N);
    iota(block_order.begin(), block_order.end(), 0);
    vector<int> seen(N, -1);
    for (int i = 0; i < N; i++) {
        prv[i] = seen[C[i]];
        seen[C[i]] = i;
    }
    for (int i = 0; i < N; i += B) {
        recalc(i / B);
    }
    for (int q = 0; q < Q; q++) {
        cin >> t >> a >> b;
        if (t == 1) {
            int l, r;
            l = a ^ last;
            r = b ^ last;
            l--; r--;
            int lb = l / B, rb = r / B;
            last = 0;
            for (int bid = lb + 1; bid < rb; bid++) {
                lo = bid * B, hi = min(N, (bid + 1) * B);
                int idx = lower_bound(block_order.begin() + lo, block_order.begin() + hi, l, [&](int x, int y) {
                    return prv[x] < y;
                }) - block_order.begin();
                if (idx > lo) last += psum[idx - 1];
            }
            for (int i = l; i < min(r + 1, (lb + 1) * B); i++) {
                if (prv[i] < l) last += L[i];
            }
            if (lb != rb) {
                for (int i = rb * B; i <= r; i++) {
                    if (prv[i] < l) last += L[i];
                }
            }
            cout << last << endl;
        } else if (t == 2) {
            int x, y;
            x = a ^ last;
            y = b ^ last;
            x--;
            L[x] = y;
            recalc(x / B);
        } else {
            int i, y, j, k;
            i = a ^ last;
            y = b ^ last;
            i--; 
            y--;
            j = N; k = N;
            auto it = indices[C[i]].find(i);
            if (next(it) != indices[C[i]].end()) {
                k = *next(it);
                prv[k] = prv[i];
            }
            indices[C[i]].erase(i);
            C[i] = y;
            indices[C[i]].insert(i);
            auto it2 = indices[y].find(i);
            if (next(it2) != indices[y].end()) {
                j = *next(it2);
                prv[j] = i;
            }
            if (it2 == indices[C[i]].begin()) {
                prv[i] = -1;
            } else {
                prv[i] = *prev(it2);
            }
            recalc(i / B);
            if (j < N) recalc(j / B);
            if (k < N) recalc(k / B);
        }
    }
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    while (T--) {
        solve();
    }
    return 0;
}
```

# Starters 132

## Mex Path

### Solution 1:  dijkstra, directed graph, mex, forward edges, backward edges

```cpp
const int MAXN = 5e5 + 5;
int N;
int arr[MAXN];
vector<vector<pair<int, int>>> adj;
vector<int> vis, last;

int dijkstra(int src, int dst) {
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    vector<bool> vis(N, false);
    pq.emplace(0, src);
    while (!pq.empty()) {
        auto [cost, u] = pq.top();
        pq.pop();
        if (u == dst) return cost;
        if (vis[u]) continue;
        vis[u] = true;
        for (auto [v, w] : adj[u]) {
            pq.emplace(cost + w, v);
        }
    }
    return -1;
}

void solve() {
    cin >> N;
    adj.assign(N, vector<pair<int, int>>());
    for (int i = 0; i < N; i++) cin >> arr[i];
    vis.assign(N + 2, 0);
    int mex = 0;
    // forward edges that start from first node
    for (int i = 0; i < N; i++) {
        vis[arr[i]] = 1;
        while (vis[mex]) mex++;
        adj[0].emplace_back(i, mex);
    }
    fill(vis.begin(), vis.end(), 0);
    mex = 0;
    // forward edges that point to last node
    for (int i = N - 1; i >= 0; i--) {
        vis[arr[i]] = 1;
        while (vis[mex]) mex++;
        adj[i].emplace_back(N - 1, mex);
    }
    // BACKWARD EDGES
    for (int i = 1; i < N; i++) {
        adj[i].emplace_back(i - 1, 0);
    }
    // FORWARD EDGES
    last.assign(N + 1, -1);
    for (int i = 0; i < N; i++) {
        if (last[arr[i]] != -1) {
            adj[last[arr[i]] + 1].emplace_back(i - 1, arr[i]);
        }
        last[arr[i]] = i;
    }
    int ans = dijkstra(0, N - 1);
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

# Starters 133

## Too Far For Comfort

### Solution 1:  factorials, modular inverse, combinatorics, prefix sum, calculate nearest occurrence of repeat

```cpp
const int MOD = 998244353;
int N, M;
vector<int> arr, marked, psum;

int inv(int i) {
  return i <= 1 ? i : MOD - (int)(MOD/i) * inv(MOD % i) % MOD;
}

vector<int> fact, inv_fact;

void factorials(int n) {
    fact.assign(n + 1, 1);
    inv_fact.assign(n + 1, 0);
    for (int i = 2; i <= n; i++) {
        fact[i] = (fact[i - 1] * i) % MOD;
    }
    inv_fact.end()[-1] = inv(fact.end()[-1]);
    for (int i = n - 1; i >= 0; i--) {
        inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % MOD;
    }
}

int choose(int n, int r) {
    if (n < r) return 0;
    return (((fact[n] * inv_fact[r]) % MOD) * inv_fact[n - r]) % MOD;
}

int sum_(int l, int r) {
    return psum[r] - (l > 0 ? psum[l - 1] : 0);
}

void solve() {
    cin >> N >> M;
    arr.resize(N);
    marked.assign(M + 1, 0);
    psum.assign(N, 0);
    for (int i = 0; i < N; i++) {
        cin >> arr[i];
        marked[arr[i]] = 1;
        psum[i] = arr[i] > 0 ? 1 : 0;
        if (i > 0) psum[i] += psum[i - 1];
    }
    vector<int> nxt(N + 1, N);
    vector<int> last(M + 1, N);
    for (int i = N - 1; i >= 0; i--) {
        if (arr[i] > 0) {
            nxt[i] = min(nxt[i + 1], last[arr[i]]);
            last[arr[i]] = i;
        } else {
            nxt[i] = nxt[i + 1];
        }
    }
    factorials(max(N, M));
    int start = count(marked.begin() + 1, marked.end(), 1);
    int ans = 0;
    for (int k = max(1LL, start); k <= N; k++) {
        int base = choose(M - start, k - start);
        int mul = 1;
        for (int s = 0; s < N; s += k) {
            int len_ = min(s + k, N) - s;
            int cnt = sum_(s, min(s + k - 1, N - 1));
            int z = len_ - cnt;
            if (nxt[s] < min(s + k, N)) {
                mul = 0;
                break;
            }
            mul = (mul * (fact[z] * choose(k - cnt, z)) % MOD) % MOD;
        }
        ans = (ans + (base * mul) % MOD) % MOD;
    }
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

## Fireworks

### Solution 1:  tree dp, reroot tree, diameter of tree, iterate over sqrt(N) degrees

```cpp
const int MAXN = 1e5 + 5;
int N, x_deg;
vector<vector<int>> adj;
vector<int> deg, mx1, mx2, vis, lpath, trav1, trav2, res;

// calculate mx1 and mx2 for all subtrees
void dfs1(int u, int p = -1) {
    mx1[u] = mx2[u] = 1;
    for (int v : adj[u]) {
        if (v == p) continue;
        dfs1(v, u);
        if (mx1[v] + 1 > mx1[u]) {
            trav2[u] = trav1[u];
            trav1[u] = v;
            mx2[u] = mx1[u];
            mx1[u] = mx1[v] + 1;
        } else if (mx2[v] + 1 > mx2[u]) {
            mx2[u] = mx2[v] + 1;
            trav2[u] = v;
        }
    }
    if (deg[u] == x_deg) {
        mx1[u] = mx2[u] = 0;
    }
}

// reroot the tree to calculate longest path from each node
void dfs2(int u, int p = -1, int mxp = 0) {
    lpath[u] = max(mxp, max(mx1[u], mx2[u]));
    for (int v : adj[u]) {
        if (v == p) continue;
        if (deg[u] == x_deg) {
            dfs2(v, u, 0);
        } else if (trav1[u] == v) {
            dfs2(v, u, max(mxp, mx2[u]) + 1);
        } else if (trav2[u] == v) {
            dfs2(v, u, max(mxp, mx1[u]) + 1);
        } else {
            dfs2(v, u, max(mxp, max(mx1[u], mx2[u])) + 1);
        }
    }
    if (deg[u] == x_deg) lpath[u] = 0;
}

// calculate answer for longest path, star graph from each node with deg[u] == x_deg
void dfs3(int u, int p = -1) {
    for (int v : adj[u]) {
        if (deg[u] == x_deg) {
            res[u] += lpath[v];
        }
        if (v == p) continue;
        dfs3(v, u);
    }
}

void solve() {
    cin >> N;
    adj.assign(N, vector<int>());
    deg.assign(N + 1, 0);
    for (int i = 0; i < N - 1; i++) {
        int u, v;
        cin >> u >> v;
        u--; v--;
        adj[u].push_back(v);
        adj[v].push_back(u);
        deg[u]++; deg[v]++;
    }
    res.assign(N, 1);
    vector<int> degrees;
    for (int i = 0; i < N; i++) {
        if (deg[i] > 0) degrees.push_back(deg[i]);
    }
    sort(degrees.begin(), degrees.end());
    degrees.erase(unique(degrees.begin(), degrees.end()), degrees.end());
    for (int degree : degrees) {
        mx1.assign(N, 0);
        mx2.assign(N, 0);
        lpath.assign(N, 0);
        trav1.assign(N, -1);
        trav2.assign(N, -1);
        x_deg = degree;
        dfs1(0);
        dfs2(0);
        dfs3(0);
    }
    for (int i = 0; i < N; i++) {
        cout << res[i] << " ";
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

# Starters 134

## Prison Escape

### Solution 1:  matrix, grid, dijkstra, flood fill, connected components

```cpp
int N, M;
vector<vector<int>> grid, dist;
vector<bool> vis;

int mat_id(int i, int j) {
    return i * M + j;
}

pair<int, int> mat_ij(int id) {
    return {id / M, id % M};
}

bool in_bounds(int i, int j) {
    return i >= 0 && i < N && j >= 0 && j < M;
}

vector<pair<int, int>> neighborhood(int i, int j) {
    return {{i - 1, j}, {i + 1, j}, {i, j - 1}, {i, j + 1}};
}

void dijkstra() {
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    vis.assign(N * M, false);
    for (int i = 0; i < N; i++) {
        pq.emplace(grid[i][0], mat_id(i, 0));
        pq.emplace(grid[i][M - 1], mat_id(i, M - 1));
    }
    for (int i = 0; i < M; i++) {
        pq.emplace(grid[0][i], mat_id(0, i));
        pq.emplace(grid[N - 1][i], mat_id(N - 1, i));
    }
    while (!pq.empty()) {
        auto [cost, u] = pq.top();
        pq.pop();
        if (vis[u]) continue;
        vis[u] = true;
        auto [i, j] = mat_ij(u);
        dist[i][j] = cost;
        for (auto [ni, nj] : neighborhood(i, j)) {
            if (!in_bounds(ni, nj)) continue;
            if (vis[mat_id(ni, nj)]) continue;
            pq.emplace(cost + grid[ni][nj], mat_id(ni, nj));
        }
    }
}

void solve() {
    cin >> N >> M;
    grid.assign(N, vector<int>(M));
    dist.assign(N, vector<int>(M));
    for (int i = 0; i < N; i++) {
            string row;
            cin >> row;
        for (int j = 0; j < M; j++) {
            grid[i][j] = row[j] - '0';
        }
    }
    dijkstra();
    vis.assign(N * M, false);
    int ans = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            if (grid[i][j] || vis[mat_id(i, j)]) continue;
            ans = max(ans, dist[i][j]);
        }
    }
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

## Count Good RBS

## Chef and Bit Tree

### Solution 1:  tree, lca, bitwise xor operation, bit manipulation, min bitwise xor pair, by sorting, bit manipulation relation, pigeonhole principle

```cpp
const int INF = 1e5, MAXN = 1e3 + 5;
int N, Q, marked[MAXN];
vector<int> par, dep, A;
vector<vector<int>> adj;

void dfs(int u, int p = -1) {
    par[u] = p;
    for (int v : adj[u]) {
        if (v == p) continue;
        dep[v] = dep[u] + 1;
        dfs(v, u);
    }
}

void solve() {
    cin >> N >> Q;
    par.resize(N);
    dep.assign(N, 0);
    adj.assign(N, vector<int>());
    for (int i = 0; i < N - 1; i++) {
        int u, v;
        cin >> u >> v;
        u--, v--;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    A.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
        A[i] /= 2;
    }
    dfs(0);
    memset(marked, 0, sizeof(marked));
    for (int i = 1; i <= Q; i++) {
        int t, u, v;
        cin >> t >> u >> v;
        if (t == 1) {
            u--;
            A[u] = v / 2;
        } else {
            u--, v--;
            int ans = INF;
            while (u != v) {
                if (dep[u] < dep[v]) swap(u, v);
                if (marked[A[u]] == i) {
                    ans = 0;
                    break;
                } else {
                    marked[A[u]] = i;
                }
                u = par[u];
            }
            if (marked[A[u]] == i) ans = 0;
            else marked[A[u]] = i;
            if (ans != 0) {
                int prv = INF;
                for (int j = 0; j < MAXN; j++) {
                    if (marked[j] == i) {
                        if (prv != INF) ans = min(ans, prv ^ j);
                        prv = j;
                    } 
                }
            }
            cout << ans << endl;
        }
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

## Permutation Cycle Queries

### Solution 1:  prefix sum, permutations, binary search, sort, permutation cycles

```cpp
int N, Q, M;
vector<int> A, B, psum1, psum2, psum3, pprod, sprod;

int sum_(vector<int> &psum, int l, int r) {
    if (l > r) return 0;
    return (psum[r] - (l == 0 ? 0 : psum[l - 1]) + M) % M;
}

void solve() {
    cin >> N >> Q >> M;
    A.resize(N);
    B.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
        B[i] = A[i];
    }
    sort(B.begin(), B.end());
    pprod.assign(N + 1, 1);
    sprod.assign(N + 2, 1);
    for (int i = 1; i <= N; i++) {
        pprod[i] = (pprod[i - 1] * i) % M;
    }
    for (int i = N; i > 0; i--) {
        sprod[i] = (sprod[i + 1] * i) % M;
    }
    psum1.assign(N, 0);
    psum2.assign(N, 0);
    psum3.assign(N, 0);
    for (int i = 0; i < N; i++) {
        psum1[i] = ((B[i] * pprod[i]) % M * sprod[i + 2]) % M; // B[i] / i
        if (i < N - 1) psum3[i] = ((B[i] * pprod[i + 1]) % M * sprod[i + 3]) % M; // B[i] / i + 1
        if (i > 0) {
            psum2[i] = ((B[i] * pprod[i - 1]) % M * sprod[i + 1]) % M; // B[i] / i - 1
            psum1[i] = (psum1[i] + psum1[i - 1]) % M;
            psum2[i] = (psum2[i] + psum2[i - 1]) % M;
            psum3[i] = (psum3[i] + psum3[i - 1]) % M;
        }
    }
    while (Q--) {
        int idx, u, ans, i, j;
        cin >> idx >> u;
        idx--;
        i = lower_bound(B.begin(), B.end(), A[idx]) - B.begin();
        j = upper_bound(B.begin(), B.end(), u) - B.begin();
        if (j > i) {
            j--;
            int s1 = sum_(psum1, 0, i - 1);
            int s2 = sum_(psum2, i + 1, j);
            int s3 = sum_(psum1, j + 1, N - 1);
            int v = (u * pprod[j] % M * sprod[j + 2]) % M;
            ans = (s1 + s2 + v + s3) % M;
        } else {
            int s1 = sum_(psum1, 0, j - 1);
            int s2 = sum_(psum3, j, i - 1);
            int s3 = sum_(psum1, i + 1, N - 1);
            int v = (u * pprod[j] % M * sprod[j + 2]) % M;
            ans = (s1 + v + s2 + s3) % M;
        }
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

# Starters 135

## Graph Cost

### Solution 1:  suffix min, greedy

```cpp
const int INF = 1e9;
int N;
vector<int> A, smin;

void solve() {
    cin >> N;
    A.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    smin.assign(N, INF);
    for (int i = N - 1; i >= 0; i--) {
        smin[i] = A[i];
        if (i + 1 < N) smin[i] = min(smin[i], smin[i + 1]);
    }
    int ans = 0;
    int i = 0;
    for (int j = 0; j < N; j++) {
        if (A[j] <= A[i] || A[j] == smin[j]) {
            ans += (j - i) * max(A[i], A[j]);
            i = j;
        }
    }
    ans += (N - i - 1) * max(A[i], A.end()[-1]);
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

## Limit of MEX

### Solution 1:  monotonic stack, size of left and right, combinatorics

```cpp
int N;
vector<int> A, R, L, last;

int calc(int n) {
    return n * (n - 1) / 2;
}

void solve() {
    cin >> N;
    A.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    int ans = calc(N);
    R.resize(N);
    L.resize(N);
    stack<int> stk;
    for (int i = 0; i < N; i++) {
        while (!stk.empty() && A[i] >= A[stk.top()]) {
            stk.pop();
        }
        L[i] = i - (stk.empty() ? -1 : stk.top());
        stk.push(i);
    }
    while (!stk.empty()) {
        stk.pop();
    }
    for (int i = N - 1; i >= 0; i--) {
        while (!stk.empty() && A[i] > A[stk.top()]) {
            stk.pop();
        }
        R[i] = (stk.empty() ? N : stk.top()) - i;
        stk.push(i);
    }
    for (int i = 0; i < N; i++) {
        ans += A[i] * L[i] * R[i];
    }
    last.assign(N + 1, 0);
    for (int i = 0; i < N; i++) {
        if (!last[A[i]]) ans -= calc(N);
        ans += calc(i - last[A[i]]);
        last[A[i]] = i + 1;
    }
    for (int i = 0; i <= N; i++) {
        if (!last[i]) continue;
        ans += calc(N - last[i]);
    }
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

## Milky-Dark Chocolates

### Solution 1:  dynammic programming, prefix sum, rearrangement of variables to simplify

```cpp
const int INF = 1e18;
int N, K;
vector<int> A, B, dp, ndp, pA, pB;

void solve() {
    cin >> N >> K;
    A.resize(N);
    B.resize(N);
    pA.resize(N);
    pB.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
        pA[i] = A[i];
        if (i > 0) pA[i] += pA[i - 1];
    }
    for (int i = 0; i < N; i++) {
        cin >> B[i];
        pB[i] = B[i];
        if (i > 0) pB[i] += pB[i - 1];
    }
    dp.assign(N + 1, INF);
    dp[0] = 0;
    for (int len = 0; len < K; len++) {
        ndp.assign(N + 1, INF);
        int minB = dp[0], minA = dp[0];
        for (int i = 1; i <= N; i++) {
            ndp[i] = min(minA + pA[i - 1], minB + pB[i - 1]);
            minB = min(minB, dp[i] - pB[i - 1]);
            minA = min(minA, dp[i] - pA[i - 1]);
        }
        swap(dp, ndp);
    }
    cout << dp.end()[-1] << endl;
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

## Envious Pile

### Solution 1:  dfs, spanning tree, path by dfs backtracking, undirected graph, connected components, greedy

```cpp
int N, W;
vector<vector<pair<int, int>>> adj;
vector<int> A, ans;
vector<bool> vis;

bool dfs(int u, int p = -1, int idx = -1) {
    if (idx != -1) {
        ans.push_back(idx);
    }
    if (u < A[0]) return true;
    for (auto [v, i]: adj[u]) {
        if (v == p || vis[v]) continue;
        vis[v] = true;
        if (dfs(v, u, i)) return true;
    }
    ans.pop_back();
    return false;
}

void solve() {
    cin >> N >> W;
    A.resize(N);
    int MAX = 0;
    for (int i = 0; i < N; i++) {
        cin >> A[i];
        MAX = max(MAX, A[i]);
    }
    adj.assign(MAX + 1, vector<pair<int, int>>());
    vis.assign(MAX + 1, false);
    // CONSTRUCT THE UNDIRECTED GRAPH
    for (int x = 1; x <= MAX; x++) {
        for (int i = 0; i < N; i++) {
            if (A[i] <= x) continue;
            adj[x].push_back({A[i] - x, i});
            adj[A[i] - x].push_back({x, i});
        }
    }
    ans.clear();
    if (!dfs(W)) {
        cout << -1 << endl;
        return;
    }
    cout << ans.size() + N << endl;
    for (int x: ans) {
        cout << x + 1 << " ";
    }
    for (int i = 1; i <= N; i++) {
        cout << i << " ";
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

# Starters 149

## Maximise Sum

### Solution 1: sort, greedy

1. if it is better to invert do that

```cpp

int N;
vector<int> arr;

void solve() {
    cin >> N;
    arr.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> arr[i];
    }
    sort(arr.begin(), arr.end());
    for (int i = 1; i < N; i += 2) {
        if (-arr[i] - arr[i - 1] > arr[i] + arr[i - 1]) {
            arr[i] = -arr[i];
            arr[i - 1] = -arr[i - 1];
        }
    }
    int ans = accumulate(arr.begin(), arr.end(), 0LL);
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

## Chef Loves Beautiful Strings (Easy Version)

### Solution 1:  math, formula

1. Derive the formula to count the number.

```cpp
#include <bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'

int N;
string S;

int summation(int n) {
    return n * (n + 1) / 2;
}

void solve() {
    cin >> N >> S;
    int x = 0;
    for (int i = 1; i < N; i++) {
        if (S[i] != S[i - 1]) x++;
    }
    int ans = max(0LL, N - x - 1) * x + summation(x - 1);
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

## Kill Monsters (Hard Version)

### Solution 1: math, sort, two pointers

```cpp
#include <bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'

int N, X, K;
vector<int> arr;

int calc(int cur) {
    int ans = 0;
    for (int x : arr) {
        if (cur > x) {
            ans++;
            cur = x;
        }
    }
    return ans;
}

void solve() {
    cin >> N >> X >> K;
    arr.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> arr[i];
    }
    sort(arr.begin(), arr.end(), greater<int>());
    int base = calc(X);
    int ans = 0;
    int prv = 0;
    vector<int> arr2, health;
    health.push_back(X);
    for (int x : arr) {
        if (X > x) {
            X = x;
            health.push_back(x);
        } else {
            if (x != prv) arr2.push_back(x);
            prv = x;
        }
    }
    int l = 0, r = 0;
    for (int i = 0; i < health.size(); i++) {
        while (r < arr2.size() && arr2[r] >= health[i]) r++;
        while (l < arr2.size() && arr2[l] >= K * health[i]) l++;
        ans = max(ans, base + r - l);
    }
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