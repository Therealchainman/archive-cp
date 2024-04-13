# Atcoder Beginner Contest 348

## D - Medicines on Grid 

### Solution 1:  grid, created directed graph, find path from source node to target node within the directed graph, that is target is reachable from source

```py
from itertools import product
from collections import deque
def main():
    R, C = map(int, input().split())
    grid = [list(input()) for _ in range(R)]
    M = int(input())
    med = {}
    adj = [[] for _ in range(M + 1)]
    for i in range(M):
        r, c, v = map(int, input().split())
        r -= 1; c -= 1
        med[(r, c)] = (v, i)
    target = M
    neighborhood = lambda r, c: [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
    in_bounds = lambda r, c: 0 <= r < R and 0 <= c < C
    def bfs(r, c):
        E, i = med[(r, c)]
        vis = [[0] * C for _ in range(R)]
        q = deque([(r, c, E)])
        while q:
            r, c, e = q.popleft()
            for nr, nc in neighborhood(r, c):
                if not in_bounds(nr, nc) or vis[nr][nc] or grid[nr][nc] == "#": continue
                vis[nr][nc] = 1
                ne = e - 1
                if ne > 0: q.append((nr, nc, ne))
                if grid[nr][nc] == "T":
                    adj[i].append(target)
                elif (nr, nc) in med:
                    adj[i].append(med[(nr, nc)][1])
    # CONSTRUCT DIRECTED GRAPH WITH MEDICINE AND TARGET
    for r, c in med:
        bfs(r, c)
    vis = [0] * (M + 1)
    q = []
    for r, c in product(range(R), range(C)):
        if grid[r][c] == "S" and (r, c) in med:
            vis[med[(r, c)][1]] = 1
            q.append(med[(r, c)][1])
            break
    while q:
        u = q.pop()
        if u == target: return print("Yes")
        for v in adj[u]:
            if vis[v]: continue 
            vis[v] = 1
            q.append(v)
    print("No")

if __name__ == '__main__':
    main()
```

## E - Minimize Sum of Distances 

### Solution 1:  reroot dp tree, dp on tree

```cpp
const int INF = INT64_MAX, MAXN = 1e5 + 5;
int labels[MAXN], sums[MAXN], costs[MAXN], psums[MAXN], pcosts[MAXN], dp[MAXN];
int N;
vector<vector<int>> adj;

void dfs1(int u, int p) {
    sums[u] = labels[u];
    for (int v : adj[u]) {
        if (v == p) continue;
        dfs1(v, u);
        sums[u] += sums[v];
        costs[u] += sums[v] + costs[v];
    }
}
void dfs2(int u, int p) {
    dp[u] = costs[u] + pcosts[u];
    for (int v : adj[u]) {
        if (v == p) continue;
        psums[v] = psums[u] + sums[u] - sums[v];
        pcosts[v] = pcosts[u] + costs[u] - costs[v] - sums[v] + psums[v];
        dfs2(v, u);
    }
}

signed main() {
    cin >> N;
    adj.assign(N, {});
    for (int i = 0; i < N - 1; ++i) {
        int u, v;
        cin >> u >> v;
        u--; v--;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    for (int i = 0; i < N; ++i) {
        cin >> labels[i];
    }
    memset(sums, 0LL, sizeof(sums));
    memset(costs, 0LL, sizeof(costs));
    dfs1(0, -1);
    memset(psums, 0LL, sizeof(psums));
    memset(pcosts, 0LL, sizeof(pcosts));
    dfs2(0, -1);
    int ans = INF;
    for (int i = 0; i < N; i++) {
        ans = min(ans, dp[i]);
    }
    cout << ans << endl;
    return 0;
}
```

## F - Oddly Similar 

### Solution 1:  bitset, bit manipulation, bitwise xor to track odd counts

```py
const int MAXN = 2e3 + 5;
int N, M, A[MAXN][MAXN];

signed main() {
    cin >> N >> M;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            cin >> A[i][j];
        }
    }
    vector<bitset<MAXN>> values(1000, bitset<MAXN>());
    vector<bitset<MAXN>> bitmasks(N, bitset<MAXN>());
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            values[A[j][i]].set(j);
        }
        for (int j = 0; j < N; j++) {
            bitmasks[j] ^= values[A[j][i]];
        }
        for (int j = 0; j < 1000; j++) {
            values[j].reset();
        }
    }
    int ans = 0;
    for (int i = 0; i < N; i++) {
        ans += bitmasks[i].count();
    }
    if (M & 1) ans -= N;
    cout << ans / 2 << endl;
    return 0;
}
```

## G - Max (Sum - Max) 

### Solution 1: 

```py

```

```cpp
#include <bits/stdc++.h>

using namespace std;
using ll = long long;

constexpr ll inf = 1ll << 60;

int n;
vector<pair<ll, ll>> x; // x : (a, b)

void read() {
    cin >> n;
    x.resize(n);
    for (auto &[a, b] : x) {
        cin >> a >> b;
    }
    sort(x.begin(), x.end(), [&](const auto &a, const auto &b) {
        return a.second < b.second;
    });
}

template <class F>
vector<ll> monotone_maxima(F &f, int h, int w) {
    vector<ll> ret(h);
    auto sol = [&](auto &&self, const int l_i, const int r_i, const int l_j, const int r_j) -> void {
        const int m_i = (l_i + r_i) / 2;
        int max_j = l_j;
        ll max_val = -inf;
        for (int j = l_j; j <= r_j; ++j) {
            const ll v = f(m_i, j);
            if (v > max_val) {
                max_j = j;
                max_val = v;
            }
        }
        ret[m_i] = max_val;

        if (l_i <= m_i - 1) {
            self(self, l_i, m_i - 1, l_j, max_j);
        }
        if (m_i + 1 <= r_i) {
            self(self, m_i + 1, r_i, max_j, r_j);
        }
    };
    sol(sol, 0, h - 1, 0, w - 1);
    return ret;
}

/*
what is a and b in this array? 
a is a vector of values for the right interval
b is a vector of values for the left interval

monotone_maxima
*/
vector<ll> max_plus_convolution(const vector<ll> &a, const vector<ll> &b) {
    const int n = (int)a.size(), m = (int)b.size();
    auto f = [&](int i, int j) {
        if (i < j or i - j >= m) {
            return -inf;
        }
        return a[j] + b[i - j];
    };

    return monotone_maxima(f, n + m - 1, n);
}

vector<ll> sol(const int l, const int r) {
    if (r - l == 1) {
        const vector<ll> ret = {-inf, x[l].first - x[l].second};
        return ret;
    }
    const int m = (l + r) / 2;
    const auto res_l = sol(l, m);
    const auto res_r = sol(m, r);

    vector<ll> sorted_l(m - l);
    for (int i = l; i < m; ++i) {
        sorted_l[i - l] = x[i].first;
    }
    sort(sorted_l.begin(), sorted_l.end(), greater());
    for (int i = 1; i < m - l; ++i) {
        sorted_l[i] += sorted_l[i - 1];
    }
    sorted_l.insert(sorted_l.begin(), -inf);
    // O(n)
    auto res = max_plus_convolution(res_r, sorted_l);

    for (int i = 0; i < (int)res_l.size(); ++i) {
        res[i] = max(res[i], res_l[i]);
    }
    for (int i = 0; i < (int)res_r.size(); ++i) {
        res[i] = max(res[i], res_r[i]);
    }
    return res;
}

void process() {
    auto ans = sol(0, n);
    for (int i = 1; i <= n; ++i) {
        cout << ans[i] << endl;
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    read();
    process();
}
```