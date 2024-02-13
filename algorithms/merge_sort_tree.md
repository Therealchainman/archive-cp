# Merge Sort Tree

Mixture of Segment tree and Merge Sort Algorithm 

If you understand segment tree and merge sort algorithm, this algorithm just follows.  Just think of each segment holding a sorted vector and you will then merge those vectors.  And each node in the segment tree holds a sorted voector is what that means.  so then you can perform binary search on that sorted vector for a range query. 

## Algorithm

I don't know why this algorithm reqires the tree to consists of 4 * n, but it does some reason, verified on a problem.

All of these below you have to follow, and build(1, 0, n - 1),  so it is 0-indexed, and your queries should be from 0 to n - 1.  And I need to recall why it sets u = 1,  But to query you need to include these as well

## practice problems

[K-query online]<https://www.spoj.com/problems/KQUERYO/>

Find the number of elements greater than k in segments/interval of an array.  Can do this with upper bound and merge sort tree.  Use upper bound to get the count and query for the total count of elements above k.

```cpp
const int N = 1e5 + 10;
vector<int> tree[4 * N]; 
int n, m, arr[N], i, j, k, a, b, c;

struct MergeSortTree {
    void build(int u, int left, int right) {
        if (left == right) {
            tree[u].push_back(arr[left]);
            return;
        }
        int mid = (left + right) >> 1;
        int left_segment = u << 1;
        int right_segment = left_segment | 1;
        build(left_segment, left, mid);
        build(right_segment, mid + 1, right);
        merge(tree[left_segment].begin(), tree[left_segment].end(), tree[right_segment].begin(), tree[right_segment].end(), back_inserter(tree[u]));
    }
    int query(int u, int left, int right, int i, int j, int k) {
        if (i > right || left > j) return 0;
        if (i <= left && right <= j) {
            int idx = upper_bound(tree[u].begin(), tree[u].end(), k) - tree[u].begin();
            return tree[u].size() - idx;
        }
        int mid = (left + right) >> 1;
        int left_segment = u << 1;
        int right_segment = left_segment | 1;
        return query(left_segment, left, mid, i, j, k) + query(right_segment, mid + 1, right, i, j, k);
    }
};

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);    cout.tie(0);
    cin >> n;
    for (int i = 0; i < n; i++) {
        cin >> arr[i];
    }
    cin >> m;
    MergeSortTree mst;
    mst.build(1, 0, n - 1);

    int ans = 0;
    while (m--) {
        cin >> a >> b >> c;
        i = a ^ ans;
        j = b ^ ans;
        k = c ^ ans;
        ans = mst.query(1, 0, n - 1, i - 1, j - 1, k);
        cout << ans << endl;
    }
    return 0;
}
```

[Unusual Entertainment]<https://www.spoj.com/problems/MKTHNUM/>

Tree and undirected graph problem that requires a depth first search with tracking of entry and exit times of nodes from starting from root node.  

Then you can use merge sort tree to search the range in the permutation or array.  And you are looking for if there exists a descendent which you can find based on the tin and tout of node x.  Then you can determine if a tin of elements in the segment of permutation contain a tin within [tin[x], tout[x]]

```cpp
vector<vector<int>> tree, adj;
int n, q, u, v, l, r, x, timer;
vector<int> tin, tout, arr;

struct MergeSortTree {
    void build(int u, int left, int right) {
        if (left == right) {
            tree[u].push_back(tin[arr[left]]);
            return;
        }
        int mid = (left + right) >> 1;
        int left_segment = u << 1;
        int right_segment = left_segment | 1;
        build(left_segment, left, mid);
        build(right_segment, mid + 1, right);
        merge(tree[left_segment].begin(), tree[left_segment].end(), tree[right_segment].begin(), tree[right_segment].end(), back_inserter(tree[u]));
    }
    int query(int u, int left, int right, int i, int j, int k1, int k2) {
        if (i > right || left > j) return 0;
        if (i <= left && right <= j) {
            int lb = lower_bound(tree[u].begin(), tree[u].end(), k1) - tree[u].begin();
            int ub = upper_bound(tree[u].begin(), tree[u].end(), k2) - tree[u].begin();
            return ub - lb;
        }
        int mid = (left + right) >> 1;
        int left_segment = u << 1;
        int right_segment = left_segment | 1;
        return query(left_segment, left, mid, i, j, k1, k2) + query(right_segment, mid + 1, right, i, j, k1, k2);
    }
};

void dfs(int u, int parent) {
    tin[u] = timer++;
    for (int v : adj[u]) {
        if (v == parent) continue;
        dfs(v, u);
    }
    tout[u] = timer++;
}

void solve() {
    cin >> n >> q;
    // nodes are 0-indexed
    adj.assign(n, vector<int>());
    for (int i = 1; i < n; i++) {
        cin >> u >> v;
        u--;
        v--;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    arr.resize(n);
    for (int i = 0; i < n; i++) {
        cin >> arr[i];
        arr[i]--;
    }
    timer = 0;
    tin.resize(n);
    tout.resize(n);
    dfs(0, -1);
    tree.assign(4 * n, vector<int>());
    MergeSortTree mst;
    mst.build(1, 0, n - 1);
    while (q--) {
        cin >> l >> r >> x;
        l--; r--; x--;
        // arr index is 0-indexed
        cout << (mst.query(1, 0, n - 1, l, r, tin[x], tout[x]) > 0 ? "YES" : "NO") << endl;
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
        cout << endl;
    }
    return 0;
}
```

[2080. Range Frequency Queries]<https://leetcode.com/problems/range-frequency-queries/>

This is in format that would work on leetcode platform.

```cpp
struct MergeSortTree {
    vector<int> values;
    vector<vector<int>> tree;
    int N;
    void init(vector<int>& arr, int l, int r) {
        N = arr.size();
        values.resize(N);
        tree.assign(4 * N, vector<int>());
        for (int i = 0; i < N; i++) {
            values[i] = arr[i];
        }
        build(1, 0, N - 1);
    }
    void build(int u, int left, int right) {
        if (left == right) {
            tree[u].push_back(values[left]);
            return;
        }
        int mid = (left + right) >> 1;
        int left_segment = u << 1;
        int right_segment = left_segment | 1;
        build(left_segment, left, mid);
        build(right_segment, mid + 1, right);
        merge(tree[left_segment].begin(), tree[left_segment].end(), tree[right_segment].begin(), tree[right_segment].end(), back_inserter(tree[u]));
    }
    int query(int u, int left, int right, int i, int j, int k) {
        if (i > right || left > j) return 0;
        if (i <= left && right <= j) {
            int lb = lower_bound(tree[u].begin(), tree[u].end(), k) - tree[u].begin();
            int ub = upper_bound(tree[u].begin(), tree[u].end(), k) - tree[u].begin();
            return ub - lb;
        }
        int mid = (left + right) >> 1;
        int left_segment = u << 1;
        int right_segment = left_segment | 1;
        return query(left_segment, left, mid, i, j, k) + query(right_segment, mid + 1, right, i, j, k);
    }
    int query(int L, int R, int k) {
        return query(1, 0, N - 1, L, R, k);
    }
};
class RangeFreqQuery {
public:
    MergeSortTree mst;
    RangeFreqQuery(vector<int>& arr) {
        int n = arr.size();
        mst.init(arr);
    }
    
    int query(int left, int right, int value) {
        return mst.query(left, right, value);
    }
};
```

## Solving another problem involving sum
You can also find the cumulative sum of all elements in a range, for when all elements are less than or equal to some value X.

```cpp
const int N = 2e5 + 10;
vector<int> tree[4 * N], psum[4 * N];
int n, arr[N], a, b, c;

struct MergeSortTree {
    void build(int u, int left, int right) {
        if (left == right) {
            tree[u].push_back(arr[left]);
            psum[u].push_back(arr[left]);
            return;
        }
        int mid = (left + right) >> 1;
        int left_segment = u << 1;
        int right_segment = left_segment | 1;
        build(left_segment, left, mid);
        build(right_segment, mid + 1, right);
        merge(tree[left_segment].begin(), tree[left_segment].end(), tree[right_segment].begin(), tree[right_segment].end(), back_inserter(tree[u]));
        int l = 0, r = 0, nl = tree[left_segment].size(), nr = tree[right_segment].size(), cur = 0;
        while (l < nl or r < nr) {
            if (l < nl and r < nr) {
                if (tree[left_segment][l] <= tree[right_segment][r]) {
                    cur += tree[left_segment][l];
                    l += 1;
                } else {
                    cur += tree[right_segment][r];
                    r += 1;
                }
            } else if (l < nl) {
                cur += tree[left_segment][l];
                l += 1;
            } else {
                cur += tree[right_segment][r];
                r += 1;
            }
            psum[u].push_back(cur);
        }
    }
    // not greater than k, so <= k we want
    int query(int u, int left, int right, int i, int j, int k) {
        if (i > right || left > j) return 0; // NO OVERLAP
        if (i <= left && right <= j) { // COMPLETE OVERLAP
            int idx = upper_bound(tree[u].begin(), tree[u].end(), k) - tree[u].begin();
            return idx > 0 ? psum[u][idx - 1] : 0;
        }
        // PARTIAL OVERLAP
        int mid = (left + right) >> 1;
        int left_segment = u << 1;
        int right_segment = left_segment | 1;
        return query(left_segment, left, mid, i, j, k) + query(right_segment, mid + 1, right, i, j, k);
    }
};
```