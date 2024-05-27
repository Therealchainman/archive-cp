# Atcoder Beginner Contest 355

## E - Guess the Sum 

### Solution 1:  shortest path, bfs, undirected graph, parent array for backtracking

```cpp
const int MOD = 100, MAX = (1 << 18);
int N, L, R, T, ans;

int upper(int target, int i) {
    int lo = 0, hi = MAX;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (mid * (1 << i) < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

int lower(int target, int i) {
    int lo = 0, hi = MAX;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (mid * (1 << i) - 1 <= target) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

void solve() {
    cin >> N >> L >> R;
    ans = 0;
    vector<pair<int, int>> ranges, nranges;
    ranges.emplace_back(L, R);
    for (int i = 18; i >= 0; i--) {
        nranges.clear();
        for (auto [l, r] : ranges) {
            cout << l << " " << r << endl;
            int s = upper(l, i), e = lower(r, i);
            if (s >= e) {
                nranges.emplace_back(l, r);
                continue;
            }
            // cout << i << " " << s << " " << e << endl;
            cout.flush();
            for (int j = s; j < e; j++) {
                cout << "? " << i << " " << j << endl;
                cout.flush();
                cin >> T;
                ans = (ans + T) % MOD;
            }
            int l1 = (1 << i) * s, r1 = (1 << i) * e - 1;
            if (l1 > l) {
                nranges.emplace_back(l, l1 - 1);
            }
            if (r1 < r) {
                nranges.emplace_back(r1 + 1, r);
            }
        }
        swap(ranges, nranges);
    }
    cout << "! " << ans << endl;
    cout.flush();
}

signed main() {
    solve();
    return 0;
}
```

## F - MST Query 

### Solution 1:  union find, union find for different edge weight graphs

```cpp
int N, Q;

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

void solve() {
    cin >> N >> Q;
    int ans = 10 * (N - 1);
    vector<UnionFind> dsus(10);
    for (auto &dsu : dsus) {
        dsu.init(N);
    }
    for (int i = 0; i < N - 1; i++) {
        int u, v, w;
        cin >> u >> v >> w;
        u--; v--;
        for (int j = w; j < 10; j++) {
            if (!dsus[j].same(u, v)) {
                ans--;
            }
        }
    }
    while (Q--) {
        int u, v, w;
        cin >> u >> v >> w;
        u--; v--;
        for (int j = w; j < 10; j++) {
            if (!dsus[j].same(u, v)) {
                ans--;
            }
        }
        cout << ans << endl;
    }
}

signed main() {
    solve();
    return 0;
}
```

## G - Baseball 

### Solution 1: 

```cpp

```