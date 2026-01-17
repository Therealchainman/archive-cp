# disjoint Sets Union

# Step 1

## Disjoint Sets Union

### Solution 1: union find

```cpp
struct UnionFind {
    vector<int> parents, size;
    UnionFind(int n) {
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

    void unite(int i, int j) {
        i = find(i), j = find(j);
        if (i!=j) {
            if (size[j]>size[i]) {
                swap(i,j);
            }
            size[i]+=size[j];
            parents[j]=i;
        }
    }

    bool same(int i, int j) {
        return find(i) == find(j);
    }
};

int N, M;

void solve() {
    cin >> N >> M;
    UnionFind dsu(N);
    while (M--) {
        string s;
        int u, v;
        cin >> s >> u >> v;
        u--, v--;
        if (s == "union") {
            dsu.unite(u, v);
        } else {
            if (dsu.same(u, v)) {
                cout << "YES" << endl;
            } else {
                cout << "NO" << endl;
            }
        }
    }
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Disjoint Sets Union 2

### Solution 1: union find, calculate min, max in disjoint sets

```cpp
struct UnionFind {
    vector<int> parents, size, mn, mx;
    UnionFind(int n) {
        parents.resize(n);
        iota(parents.begin(),parents.end(),0);
        size.assign(n,1);
        mn.resize(n);
        mx.resize(n);
        iota(mn.begin(), mn.end(), 0);
        iota(mx.begin(), mx.end(), 0);
    }

    int find(int i) {
        if (i==parents[i]) {
            return i;
        }
        return parents[i]=find(parents[i]);
    }

    void unite(int i, int j) {
        i = find(i), j = find(j);
        if (i!=j) {
            if (size[j]>size[i]) {
                swap(i,j);
            }
            size[i]+=size[j];
            parents[j]=i;
            mn[i] = min(mn[i], mn[j]);
            mx[i] = max(mx[i], mx[j]);
        }
    }

    bool same(int i, int j) {
        return find(i) == find(j);
    }
};

int N, M;

void solve() {
    cin >> N >> M;
    UnionFind dsu(N);
    while (M--) {
        string s;
        cin >> s;
        if (s == "union") {
            int u, v;
            cin >> u >> v;
            u--, v--;
            dsu.unite(u, v);
        } else {
            int u;
            cin >> u;
            u--;
            u = dsu.find(u);
            cout << dsu.mn[u] + 1 << " " << dsu.mx[u] + 1 << " " << dsu.size[u] << endl;
        }
    }
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Experience

### Solution 1: union find, prefix sum, order

I need to remove path compression from dsu.

When you query for value you take the node and follow it parent links and when you do that you track the rank or at which query index that this parent link was created. 

This is important because then in that parent node it will have different query indexes where integer was added to it, you want to query with lower bound for the appropriate index.  

That is if you have order = [0, 1, 4, 6, 8] but the lastEdge = 5, that means this parent connection wasn't created until 5, which means technically only the values added at 6, 8 are going to apply to it so basically just subtract off the prefix sum from psum[0...2]. This gives the values that applied to this node, which doesn't happen until these are connected, the query index when the parent link is created. 

```cpp
struct UnionFind {
    vector<int> parent, size, linkIndex;
    vector<vector<int>> psum, order;
    UnionFind(int n) {
        parent.resize(n);
        iota(parent.begin(),parent.end(),0);
        size.assign(n,1);
        linkIndex.assign(n, -1);
        order.assign(n, vector<int>(1, 0));
        psum.assign(n, vector<int>(1, 0));
    }

    int find(int i) {
        return i == parent[i] ? i : find(parent[i]);
    }

    void unite(int i, int j, int k) {
        i = find(i), j = find(j);
        if (i!=j) {
            if (size[j]>size[i]) {
                swap(i,j);
            }
            size[i] += size[j];
            parent[j] = i;
            linkIndex[j] = k;
            // i -> j
        }
    }

    void add(int u, int val, int i) {
        u = find(u);
        order[u].emplace_back(i);
        psum[u].emplace_back(psum[u].back() + val);
    }

    bool same(int i, int j) {
        return find(i) == find(j);
    }

    int valuation(int u, int i) {
        int ans = 0, lastEdge = 0;
        while (true) {
            int idx = upper_bound(order[u].begin(), order[u].end(), lastEdge) - order[u].begin();
            int suf = psum[u].back() - psum[u][idx - 1];
            ans += suf;
            if (u == parent[u]) break;
            lastEdge = linkIndex[u];
            u = parent[u];
        }
        return ans;
    }
};

int N, M;

void solve() {
    cin >> N >> M;
    UnionFind dsu(N);
    for (int i = 1; i <= M; ++i) {
        string s;
        cin >> s;
        int u;
        cin >> u;
        u--;
        if (s == "add") {
            int val;
            cin >> val;
            dsu.add(u, val, i);
        } else if (s == "join") {
            int v;
            cin >> v;
            v--;
            dsu.unite(u, v, i);
        } else {
            cout << dsu.valuation(u, i) << endl;
        }
    }
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Cutting a Graph

### Solution 1: union find, offline query, reverse, 

```cpp
struct UnionFind {
    vector<int> parents, size;
    UnionFind(int n) {
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

    void unite(int i, int j) {
        i = find(i), j = find(j);
        if (i!=j) {
            if (size[j]>size[i]) {
                swap(i,j);
            }
            size[i]+=size[j];
            parents[j]=i;
        }
    }

    bool same(int i, int j) {
        return find(i) == find(j);
    }
};

int N, M, K;

void solve() {
    cin >> N >> M >> K;
    for (int i = 0; i < M; ++i) {
        int u, v;
        cin >> u >> v;
    }
    UnionFind dsu(N);
    vector<tuple<string, int, int>> events;
    for (int i = 0; i < K; ++i) {
        string s;
        int u, v;
        cin >> s >> u >> v;
        u--, v--;
        events.emplace_back(s, u, v);
    }
    reverse(events.begin(), events.end());
    vector<string> ans;
    for (auto [s, u, v] : events) {
        if (s == "ask") {
            if (dsu.same(u, v)) {
                ans.emplace_back("YES");
            } else {
                ans.emplace_back("NO");
            }
        } else {
            dsu.unite(u, v);
        }
    }
    reverse(ans.begin(), ans.end());
    for (const string &s : ans) {
        cout << s << endl;
    }
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Monkeys

### Solution 1: union find, disjoint sets, offline query, reverse, small to large merging

```cpp
struct UnionFind {
    vector<int> parents, size;
    vector<vector<int>> components;
    UnionFind(int n) {
        parents.resize(n);
        iota(parents.begin(),parents.end(),0);
        size.assign(n,1);
        components.assign(n, vector<int>());
        for (int i = 0; i < n; ++i) {
            components[i].emplace_back(i);
        }
    }

    int find(int i) {
        return i == parents[i] ? i : find(parents[i]);
    }

    void unite(int i, int j) {
        i = find(i), j = find(j);
        if (i!=j) {
            if (size[j]>size[i]) {
                swap(i,j);
            }
            size[i]+=size[j];
            parents[j]=i;
            for (int v : components[j]) {
                components[i].emplace_back(v);
            }
        }
    }

    bool same(int i, int j) {
        return find(i) == find(j);
    }
};

int N, M;
vector<int> A[2];
vector<bool> mL, mR;

void solve() {
    cin >> N >> M;
    for (int i = 0; i < 2; ++i) {
        A[i].assign(N, -1);
    }
    UnionFind dsu(N);
    for (int i = 0; i < N; ++i) {
        cin >> A[0][i] >> A[1][i];
        if (A[0][i] != -1) A[0][i]--;
        if (A[1][i] != -1) A[1][i]--;
    }
    mL.assign(N, false);
    mR.assign(N, false);
    vector<pair<int, int>> events;
    for (int i = 0; i < M; ++i) {
        int p, h;
        cin >> p >> h;
        p--, h--;
        if (h == 0) {
            mL[p] = true;
        } else {
            mR[p] = true;
        }
        events.emplace_back(p, h);
    }
    for (int i = 0; i < N; ++i) {
        if (!mL[i] && A[0][i] != -1) dsu.unite(i, A[0][i]);
        if (!mR[i] && A[1][i] != -1) dsu.unite(i, A[1][i]);
    }
    vector<int> ans(N, -1);
    for (int i = M - 1; i >= 0; --i) {
        auto [p, h] = events[i];
        int w = dsu.find(0), u = dsu.find(p), v = dsu.find(A[h][p]);
        if (u == v) continue;
        if (u == w) {
            for (int x : dsu.components[v]) {
                ans[x] = i;
            }
        }
        else if (v == w) {
            for (int x : dsu.components[u]) {
                ans[x] = i;
            }
        }
        dsu.unite(u, v);
    }
    for (const int x : ans) {
        cout << x << endl;
    }
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```