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

```py

```