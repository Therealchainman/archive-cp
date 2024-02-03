# Codeforces Round 923 Div 3

## A. Make it White

### Solution 1:  min and max

```py
def main():
    n = int(input())
    s = input()
    first, last = n, 0
    for i in range(n):
        if s[i] == "B":
            first = min(first, i)
            last = max(last, i)
    print(last - first + 1)
 
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## B. Following the String

### Solution 1:  deque, strings, dictionary of deque

```py
from collections import defaultdict
import string
 
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    ans = [None] * n
    letters = defaultdict(list)
    letters[0] = list(string.ascii_lowercase)
    for i in range(n):
        ch = letters[arr[i]].pop()
        ans[i] = ch
        letters[arr[i] + 1].append(ch)
    print("".join(ans))
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## C. Choose the Different Ones!

### Solution 1:  set, set intersection, set difference

```py
def main():
    n, m, k = map(int, input().split())
    A = set(filter(lambda x: 1 <= x <= k, map(int, input().split())))
    B = set(filter(lambda x: 1 <= x <= k, map(int, input().split())))
    shared = A & B
    arr1, arr2 = list(A - shared), list(B - shared)
    for x in shared:
        if len(arr1) < k // 2: arr1.append(x)
        else: arr2.append(x)
    ans = "YES" if len(arr1) == len(arr2) == k // 2 else "NO"
    print(ans)
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## D. Find the Different Ones!

### Solution 1:  sort, line sweep, set, distinct adjacent always

probably could binary search as well

```py
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    q = int(input())
    starts, ends = [None] * q, [None] * q
    ans = [[-1] * 2 for _ in range(q)]
    for i in range(q):
        l, r = map(int, input().split())
        l -= 1; r -= 1
        starts[i] = (l, i)
        ends[i] = (r, i)
    starts.sort(); ends.sort()
    cur = set()
    s = e = 0
    for i in range(n):
        if i > 0 and arr[i] != arr[i - 1]:
            for j in cur:
                ans[j][0] = i
                ans[j][1] = i + 1
            cur.clear()
        while s < len(starts) and starts[s][0] == i:
            cur.add(starts[s][1])
            s += 1
        while e < len(ends) and ends[e][0] == i:
            cur.discard(ends[e][1])
            e += 1
    for i in range(q):
        print(*ans[i])
    print()

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## E. Clever Permutation

### Solution 1:  greedy

```py
def main():
    n, k = map(int, input().split())
    ans = [0] * (n + 1)
    start, end = 1, n
    for i in range(k // 2):
        for j in range(2 * i + 1, n + 1, k):
            ans[j] = start
            start += 1
        for j in range(2 * i + 2, n + 1, k):
            ans[j] = end
            end -= 1
    print(*ans[1:])
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## F. Microcycle

### Solution 1: 

```cpp
const int INF = 1e9;
int N, M, timer, src, dst;
vector<vector<pair<int, int>>> adj;
vector<tuple<int, int, int>> edges;
set<int> cedges, on_stack;
vector<bool> vis;
vector<int> ancestor, disc, ans;

struct UnionFind {
    vector<int> parents, size;
    vector<bool> has_cycle;
    void init(int n) {
        parents.resize(n);
        iota(parents.begin(),parents.end(),0);
        size.assign(n,1);
        has_cycle.assign(n, false);
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
            has_cycle[i] = has_cycle[i] || has_cycle[j];
            parents[j]=i;
            return false;
        }
        has_cycle[i] = true;
        return true;
    }
};

void dfs(int u, int p) {
    if (vis[u]) return;
    disc[u] = timer++;
    ancestor[u] = disc[u] + 1;
    vis[u] = true;
    for (auto [v, i] : adj[u]) {
        if (v == p) continue;
        if (vis[v]) {
            ancestor[u] = min(ancestor[u], ancestor[v]);
            continue;
        } 
        dfs(v, u);
        if (ancestor[u] == disc[u]) {
            // cout << "u: " << u << " " << ancestor[u] << " " << disc[u] << endl;
            cedges.erase(i);
        }
    }
}

bool dfs1(int u, int p) {
    // cout << u << " " << p << endl;
    on_stack.insert(u);
    if (u == dst) {
        ans.push_back(u);
        return true;
    }
    for (auto [v, i] : adj[u]) {
        // cout << u << "neighbors: " << v << endl;
        if (v == p) continue;
        if (on_stack.find(v) != on_stack.end()) continue;
        if (dfs1(v, u)) {
            ans.push_back(u);
            return true;
        }
    }
    on_stack.erase(u);
    return false;
}

void solve() {
    cin >> N >> M;
    adj.assign(N, vector<pair<int, int>>());
    edges.resize(M);
    cedges.clear();
    vis.assign(N, false);
    for (int i = 0; i < M; i++) {
        int u, v, w;
        cin >> u >> v >> w;
        u--, v--;
        adj[u].emplace_back(v, i);
        adj[v].emplace_back(u, i);
        edges[i] = {u, v, w};
    }
    UnionFind dsu;
    dsu.init(N);
    for (int i = 0; i < M; i++) {
        int u, v, _;
        tie(u, v, _) = edges[i];
        dsu.union_(u, v);
    }
    for (int i = 0; i < M; i++) {
        int u, v, _;
        tie(u, v, _) = edges[i];
        u = dsu.find(u), v = dsu.find(v);
        if (dsu.has_cycle[u] && dsu.has_cycle[v]) {
            cedges.insert(i);
        }
    }
    disc.assign(N, -1);
    ancestor.assign(N, -1);
    timer = 0;
    for (int u = 0; u < N; u++) {
        if (vis[u]) continue;
        dfs(u, -1);
    }
    ans.clear();
    int weight = INF;
    int start_edge = -1;
    for (int i : cedges) {
        int u, v, w;
        tie(u, v, w) = edges[i];
        if (w < weight) {
            weight = w;
            start_edge = i;
        }
    }
    tie(src, dst, weight) = edges[start_edge];
    dfs1(src, dst);
    cout << weight << " " << ans.size() << endl;
    for (int x : ans) {
        cout << x + 1 << " ";
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

```cpp
struct Edge {
    int u, v, w, index;
};

const int INF = 1e9;
int N, M, timer, src, dst;
vector<vector<pair<int, int>>> adj;
vector<Edge> edges;
set<int> cedges, on_stack;
vector<bool> vis;
vector<int> ancestor, disc, ans;

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
            return false;
        }
        return true;
    }
};


bool dfs(int u, int p) {
    // cout << u << " " << p << endl;
    on_stack.insert(u);
    if (u == dst) {
        ans.push_back(u);
        return true;
    }
    for (auto [v, i] : adj[u]) {
        // cout << u << "neighbors: " << v << endl;
        if (v == p) continue;
        if (on_stack.find(v) != on_stack.end()) continue;
        if (dfs(v, u)) {
            ans.push_back(u);
            return true;
        }
    }
    on_stack.erase(u);
    return false;
}


struct MaxHeapComparator {
    bool operator()(Edge lhs, Edge rhs) {
        // For max heap, we return true if lhs is "less" than rhs.
        return lhs.w < rhs.w;
    }
};

void solve() {
    cin >> N >> M;
    adj.assign(N, vector<pair<int, int>>());
    edges.resize(M);
    cedges.clear();
    vis.assign(N, false);
    for (int i = 0; i < M; i++) {
        int u, v, w;
        cin >> u >> v >> w;
        u--, v--;
        adj[u].emplace_back(v, i);
        adj[v].emplace_back(u, i);
        edges[i] = {u, v, w, i};
    }
    int weight = INF;
    Edge start_edge;
    priority_queue<Edge, vector<Edge>, MaxHeapComparator> max_heap(edges.begin(), edges.end());
    UnionFind dsu;
    dsu.init(N);
    while (!max_heap.empty()) {
        Edge e = max_heap.top();
        max_heap.pop();
        if (dsu.union_(e.u, e.v)) { // cycle detected
            start_edge = e;
            weight = e.w;
        }
    }
    ans.clear();
    int src = start_edge.u, dst = start_edge.v;
    dfs(src, dst);
    cout << weight << " " << ans.size() << endl;
    for (int x : ans) {
        cout << x + 1 << " ";
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

## G. Paint Charges

### Solution 1: 

```py

```

