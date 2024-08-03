# Atcoder Beginner Contest 363

## 

### Solution 1: 

```cpp

```

## F - Range Connect MST 

### Solution 1:  union find, merge sets, sort, functional graph, next array

```cpp
struct Item {
    int c, l, r;
    Item(int c, int l, int r) : c(c), l(l), r(r) {}
    bool operator<(const Item &other) const {
        return c < other.c;
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

    bool same(int i, int j) {
        i = find(i), j = find(j);
        if (i!=j) {
            if (size[j]>size[i]) {
                swap(i,j);
            }
            size[i]+=size[j];
            parents[j]=i;
            return false;
        }
        return true;
    }
};

int N, Q;
vector<Item> arr;

void solve() {
    cin >> N >> Q;
    for (int i = 0; i < Q; i++) {
        int c, l, r;
        cin >> l >> r >> c;
        l--; r--;
        arr.emplace_back(c, l, r);
    }
    sort(arr.begin(), arr.end());
    UnionFind dsu;
    dsu.init(N);
    vector<int> nxt(N);
    iota(nxt.begin(), nxt.end(), 0);
    int ans = 0;
    for (auto &[c, l, r] : arr) {
        int u = l;
        ans += c;
        while (u < r) {
            // find the last node in the current set
            u = nxt[dsu.find(u)];
            // merge with start of next set. 
            if (u + 1 <= r) {
                ans += c;
                int v = nxt[dsu.find(u + 1)]; // determine last of next set
                dsu.same(u, u + 1);
                nxt[dsu.find(u)] = v; // set last of current merged set to the last of the next set.
            }
            u++;
        }
    }
    if (dsu.size[dsu.find(0)] == N) {
        cout << ans << endl;
    } else {
        cout << -1 << endl;
    }
}

signed main() {
    solve();
    return 0;
}
```

## G - Last Major City 

### Solution 1:  minimum steiner tree problem, bitmask dp, min heap, dijkstra, enumerate submasks

```cpp
const int INF = 1e16;
int N, M, K;
vector<vector<pair<int, int>>> adj;
vector<vector<int>> dp;

void solve() {
    cin >> N >> M >> K;
    K--;
    dp.assign(1 << K, vector<int>(N, INF));
    for (int i = 0; i < K; i++) {
        dp[1 << i][i] = 0; // fixed terminal nodes for steiner tree
    }
    adj.assign(N, vector<pair<int, int>>());
    // construct weighted graph
    for (int i = 0; i < M; i++) {
        int u, v, w;
        cin >> u >> v >> w;
        u--; v--;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }
    for (int mask = 1; mask < (1 << K); mask++) {
        for (int submask = mask; submask > 0; submask = (submask - 1) & mask) {
            for (int i = 0; i < N; i++) {
                dp[mask][i] = min(dp[mask][i], dp[submask][i] + dp[mask - submask][i]); // mask - submask works because it is a submask, this gets the set difference
            }
        }
        // dijkstra part to find shortest path given this bitmask or set of elements in a steiner tree
        // And calculate the shortest path to be able to reach vertex v.
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> minheap;
        for (int i = 0; i < N; i++) {
            minheap.emplace(dp[mask][i], i);
        }
        // shortest distance from any node in the mask or set of nodes (steiner tree) to any other node outside of the current steiner tree. 
        while (!minheap.empty()) {
            auto [dist, u] = minheap.top();
            minheap.pop();
            if (dist > dp[mask][u]) continue;
            for (auto [v, w] : adj[u]) {
                if (dp[mask][u] + w < dp[mask][v]) {
                    dp[mask][v] = dp[mask][u] + w;
                    minheap.emplace(dp[mask][v], v);
                }
            }
        }
    }
    for (int i = K; i < N; i++) {
        cout << dp.end()[-1][i] << endl;
    }
}

signed main() {
    solve();
    return 0;
}
```