# Codeforces Educational 163 Div 2

## E. Clique Partition

### Solution 1:  brute force to find pattern

Brute forced solution to find the patterns, when assuming the maximum size of a clique is K, and if you N > K, then you will need to form ceil(N / K) cliques

```py
N = 10
K = 5
arr = range(N)
cnt = 0
manhattan_distance = lambda x1, y1, x2, y2: abs(x1 - x2) + abs(y1 - y2)
def check(labels):
    partitions = []
    for k in range(0, N, K):
        partitions.append(labels[k : k + K])
    for part in partitions:
        for i in range(len(part)):
            for j in range(i):
                if manhattan_distance(i, part[i], j, part[j]) > K: return False
    return True
solutions = []
for p in permutations(arr):
    if check(p):
        solutions.append(p)

print(f"{len(solutions):,}")
```

The solution with an observed pattern from above, with assumption max size of clique is k

```py
def ceil(x, y):
    return (x + y - 1) // y

def solve(n, size):
    labels = [0] * n
    m = ceil(size, 2)
    pat = list(range(m, 0, -1)) + list(reversed(range(m + 1, size + 1)))
    for i in range(n):
        labels[i] = pat[i % size]
    return labels

def main():
    n, k = map(int, input().split())
    ans = [0] * n
    cnt = ceil(n, k)
    labels = []
    rem = n % k
    labels += (solve(n - rem, k))
    labels += (solve(rem, rem))
    cliques = -1
    for i in range(n):
        if i % k == 0: cliques += 1
        labels[i] += cliques * k
        ans[i] = cliques + 1
    print(*labels)
    print(cnt)
    print(*ans)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## F. Rare Coins

### Solution 1:  combinatorics, combinations, prefix sum, binomial coefficients, 

```py
MOD = 998244353
def mod_inverse(x):
    return pow(x, MOD - 2, MOD)
def factorials(n):
    fact, inv_fact = [1] * (n + 1), [0] * (n + 1)
    for i in range(2, n + 1):
        fact[i] = (fact[i - 1] * i) % MOD
    inv_fact[-1] = mod_inverse(fact[-1])
    for i in reversed(range(n)):
        inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % MOD
    return fact, inv_fact
def main():
    N, Q = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    total_g = sum(A)
    psum_g = list(accumulate(A))
    total_s = sum(B)
    psum_s = list(accumulate(B))
    fact, inv_fact = factorials(total_s)
    def choose(n, k):
        return (fact[n] * inv_fact[n - k] * inv_fact[k]) % MOD if n >= k else 0
    dp = [0] * (total_s + 1)
    for i in range(total_s + 1):
        dp[i] = choose(total_s, i)
        if i > 0: dp[i] = (dp[i] + dp[i - 1]) % MOD
    inv_total_outcomes = mod_inverse(pow(2, total_s, MOD))
    ans = [0] * Q
    for i in range(Q):
        l, r = map(int, input().split())
        l -= 1; r -= 1
        cg = psum_g[r]
        if l > 0: cg -= psum_g[l - 1]
        delta = 2 * cg - total_g
        cs = psum_s[r]
        if l > 0: cs -= psum_s[l - 1]
        ub = min(total_s, cs + delta - 1)
        # probability is (number of desired outcomes) / (total number of outcomes)
        ans[i] = (dp[ub] * inv_total_outcomes) % MOD if ub >= 0 else 0
    print(*ans)

if __name__ == '__main__':
    main()
```

## G. MST with Matching

### Solution 1:  konig's theorem, minimum spanning tree, tree is bipartite graph, minimum vertex cover = maximum matching

```cpp
const int INF = 1e18;
int N, C;
vector<vector<pair<int, int>>> adj;

struct Edge {
    int u, v, w;
    bool operator<(const Edge &e) const {
        return w < e.w;
    }
};

struct UnionFind {
    vector<int> parents, size;
    void init(int n) {
        parents.resize(n);
        iota(parents.begin(),parents.end(),0);
        size.assign(n,1);
    }

    int find(int i) {
        if (i==parents[i]) {
            return i;
        }
        return parents[i]=find(parents[i]);
    }

    bool union_(int i, int j) {
        i = find(i), j = find(j);
        if (i!=j) {
            if (size[j]>size[i]) {
                swap(i,j);
            }
            size[i]+=size[j];
            parents[j]=i;
            return true;
        }
        return false;
    }
};

void solve() {
    cin >> N >> C;
    adj.assign(N, {});
    vector<Edge> edges;
    for (int u = 0; u < N; u++) {
        for (int v = 0; v < N; v++) {
            int w;
            cin >> w;
            if (w > 0 && v > u) {
                edges.push_back({u, v, w});
            }
        }
    }
    int ans = INF;
    sort(edges.begin(), edges.end());
    for (int mask = 1; mask < 1 << N; mask++) {
        UnionFind uf;
        uf.init(N);
        int cost = __builtin_popcount(mask) * C;
        int num_edges = 0;
        for (Edge e : edges) {
            if (((mask >> e.u) & 1) || ((mask >> e.v) & 1)) {
                if (uf.union_(e.u, e.v)) {
                    cost += e.w;
                    num_edges++;
                }
            }
        }
        if (num_edges == N - 1) ans = min(ans, cost);
    }
    cout << ans << endl;
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


