# Centroid Decomposition

Centroid decomposition is a recursive technique that splits a tree into smaller pieces by repeatedly removing “centroid” nodes—nodes whose removal leaves every connected component of size ≤½ the original.  The result is a new “centroid-tree” which has height O(log n).

Centroid Decomposition is a divide-and-conquer technique on trees. It is used to solve various problems that involve paths in a tree, such as counting paths with certain properties, finding distances, or answering queries on tree paths.

---

## Data Structures

- `tree`: adjacency list of the original tree, size n+1  
- `cd`: adjacency list of the centroid-tree, size n+1  
- `subSize[v]`: size of the (active) subtree rooted at v  
- `removed[v]`: boolean flag marking whether v has been “deleted” in the decomposition 

Important to note that this you need the nodes to be 1-indexed, cause it makes a dummy node labeled as 0 to be the root of the centroid tree.

### calcSums
Compute the size of each (remaining) subtree rooted at node, storing it in subSize[node]. This size count excludes any nodes already “removed” (marked in removed[]).

### findCentroid

Given a connected component of (un‐deleted) nodes of total size sz, find its centroid: a node which, if chosen as root and “removed,” leaves no child‐subtree of size ≥ sz/2.

### centroidDecomposition

Recursively assemble the centroid‐decomposition tree. Each time you pick a centroid of some remaining component, you:
1. Record it as a child of its parent‐centroid in the cd adjacency list.
1. Mark it deleted in removed[], so subsequent centroids ignore it.
1. For each of the resulting smaller components (each neighbor subtree under that centroid), recompute sizes, find that component’s centroid, and recurse.

```cpp
int N, K;
vector<vector<int>> adj;
vector<bool> removed;

struct CentroidDecomposition {
    int n;
    vector<int> subSize;
    CentroidDecomposition(int n) : n(n) {
        subSize.assign(n, 0);
    }
    int getSize(int u, int p = -1) {
        subSize[u] = 1;
        for (int v : adj[u]) {
            if (v == p || removed[v]) continue;
            subSize[u] += getSize(v, u);
        }
        return subSize[u];
    }
    int getCentroid(int u, int p, int totalSize) {
        for (int v : adj[u]) {
            if (v == p || removed[v]) continue;
            if (2 * subSize[v] > totalSize) {
                return getCentroid(v, u, totalSize);
            }
        }
        return u;
    }
    template<class ProcessCentroid>
    void decompose(int u, ProcessCentroid processCentroid) {
        int totalSize = getSize(u);
        int c = getCentroid(u, -1, totalSize);
        processCentroid(c);
        removed[c] = true;
        for (int v : adj[c]) {
            if (removed[v]) continue;
            decompose(v, processCentroid);
        }
    }
};

struct CountPathsExactlyK {
    int k;
    int64 ans = 0;
    vector<int> cnt;
    CountPathsExactlyK(int k) : k(k) {
        cnt.assign(k + 1, 0);
    }
    void collectDistances(int u, int p, int depth, vector<int>& distances) {
        if (depth > k) return;
        distances.emplace_back(depth);
        for (int v : adj[u]) {
            if (v == p || removed[v]) continue;
            collectDistances(v, u, depth + 1, distances);
        }
    }
    void operator()(int c) {
        vector<int> touched;
        cnt[0] = 1;
        touched.emplace_back(0);
        for (int v : adj[c]) {
            if (removed[v]) continue;
            vector<int> distances;
            collectDistances(v, c, 1, distances);
            for (int d : distances) {
                ans += cnt[k - d];
            }
            for (int d : distances) {
                cnt[d]++;
                touched.emplace_back(d);
            }
        }
        for (int d : touched) {
            cnt[d] = 0;
        }
    }
};

void solve() {
    cin >> N >> K;
    adj.assign(N, vector<int>());
    for (int i = 0; i < N - 1; ++i) {
        int u, v;
        cin >> u >> v;
        --u, --v;
        adj[u].emplace_back(v);
        adj[v].emplace_back(u);
    }
    removed.assign(N, false);
    CentroidDecomposition cd(N);
    CountPathsExactlyK counter(K);
    cd.decompose(0, ref(counter));
    cout << counter.ans << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```


## A great explanation of time complexity

After removing the centroid, every remaining component has size at most n / 2. 

so the recursion depth is at most O(logn)
because a node can only belong to components of sizes like: n, n/2, n/4, n/8,... until it reaches size 1. So each node participates in at most O(logn) decomposition levels. 
Therefore n nodes * log n levels = O(nlogn)
The important thing is that even though you recurse into multiple components, those components are disjoint.  So at one decomposition level, the total number of nodes processed across all recursive calls is still at most n. 