# USACO 2026

# First Contest, Gold

## Problem 1: COW Traversals    

### Solution 1: 

similar problem reachable pairs january contest 2025, gold

```cpp

```

## Problem 2: Milk Buckets

### Solution 1: bitonic sequence, inversion counting, independence

Easier problem, pyramid array in cses

I believe a way to get the largest value is to create a bitonic array, the array has two components, first part arr[1...i] is descending and second part arr[i...N] is ascending.
The problem reduces to finding the minimum number of adjacent swap operations to create a bitonic array.
For the sequence create two arrays, 
- Array A[i] = the number of elements smaller to the right of ith element, which just means in order to sort this in ascending order these elements to the right would have to be swapped with the ith element.
- Array B[i] = the number of elements smaller to the left of the ith element, which just means in order to sort his in descending order these elements to the left would have to be swapped with the ith element.
The answer is sum(min(A[i], B[i])).
If A[i] < B[i], that means you want to consider including this ith element in the ascending part, which means you have to swap A[i] elements with it.
If B[i] < A[i], that means you want to consider including this ith element in the descending part, which means y ou have to swap B[i] elements with it. 
Basically this picks the optimal part to include the ith element, such that it minimizes the number of swaps. 

This works because the number of swaps of each element is independent, that is whatever I decide for ith element is independent of jth element.  

There is one thing I'm not considering which is elements of equal values, need to consider that carefully.

```cpp

```

## Problem 3: Supervision

### Solution 1: dynamic programming, optimized with lazy segment tree, range multiplication by 2, and point set with the current prefix sum

dynamic programming with the state dp[i][j] = Number of ways to assign campers and coaches in the array prefix [0...i] such that the last assigned coach is at the index j. The transitions are as follows:
- At each step we can assign dp[i][j] = dp[i - 1][j] for all j because the numver of valid assignments will at least stay the same when we consider the ith element.
- if ith element is a coach, we can only set dp[i][i] = sum(dp[i - 1][j]) for all j < i, because we can assign this coach at the ith position and we can do that for all previous valid assignments.
- if the ith element is a camper, we can include this camper when pos[i] - pos[j] <= D in all cases, so the transition is dp[i][j] += dp[i - 1][j]  The reason is if we had x ways to assign those when the jth coach was assigned, then we have two options for each of those x ways, either we include the camper at ith position or we don't, so the number of ways doubles.

Can make the transitions fast by using two pointer technique and lazy segment tree.

```cpp
const int MOD = 1e9 + 7;
int N, D;

struct LazySegmentTree {
    vector<int64> arr;
    vector<int64> lazyTag;
    int size;

    struct Configuration {
        const int64 neutral; // identity element for merge
        const int noop; // identity element for lazy
        function<int64(int64, int64)> merge; // combine two children
        function<int64(int64, int64, int)> apply; // apply lazy tag to node value over length
        function<int64(int64, int64)> compose; // merge two lazy tags
    } config;

    LazySegmentTree(int n, Configuration config) : config(config) { init(n); }

    void init(int n) {
        size = 1;
        while (size < n) size *= 2;
        arr.assign(2 * size, config.neutral);
        lazyTag.assign(2 * size, config.noop);
    }

    void build(const vector<int64>& inputArr) {
        copy(inputArr.begin(), inputArr.end(), arr.begin() + (size - 1));
        for (int i = size - 2; i >= 0; --i) {
            arr[i] = config.merge(arr[2 * i + 1], arr[2 * i + 2]);
        }
    }

    bool is_leaf(int segment_right_bound, int segment_left_bound) {
        return segment_right_bound - segment_left_bound == 1;
    }

    void push(int segment_idx, int segment_left_bound, int segment_right_bound) {
        bool pendingUpdate = lazyTag[segment_idx] != config.noop;
        if (is_leaf(segment_right_bound, segment_left_bound) || !pendingUpdate) return;
        int left_segment_idx = 2 * segment_idx + 1, right_segment_idx = 2 * segment_idx + 2;
        int children_segment_len = (segment_right_bound - segment_left_bound) >> 1;
        lazyTag[left_segment_idx] = config.compose(lazyTag[left_segment_idx], lazyTag[segment_idx]);
        lazyTag[right_segment_idx] = config.compose(lazyTag[right_segment_idx], lazyTag[segment_idx]);
        arr[left_segment_idx] = config.apply(arr[left_segment_idx], lazyTag[segment_idx], children_segment_len);
        arr[right_segment_idx] = config.apply(arr[right_segment_idx], lazyTag[segment_idx], children_segment_len);
        lazyTag[segment_idx] = config.noop;
    }

    void update(int left, int right, int64 val) {
        update(0, 0, size, left, right, val);
    }

    void update(int segment_idx, int segment_left_bound, int segment_right_bound, int left, int right, int64 val) {
        // NO OVERLAP
        if (right <= segment_left_bound || segment_right_bound <= left) return;
        // COMPLETE OVERLAP
        if (left <= segment_left_bound && segment_right_bound <= right) {
            lazyTag[segment_idx] = config.compose(lazyTag[segment_idx], val);
            int segment_len = segment_right_bound - segment_left_bound;
            arr[segment_idx] = config.apply(arr[segment_idx], val, segment_len);
            return;
        }
        // PARTIAL OVERLAP;
        push(segment_idx, segment_left_bound, segment_right_bound);
        int mid_point = (segment_left_bound + segment_right_bound) >> 1;
        int left_segment_idx = 2 * segment_idx + 1, right_segment_idx = 2 * segment_idx + 2;
        update(left_segment_idx, segment_left_bound, mid_point, left, right, val);
        update(right_segment_idx, mid_point, segment_right_bound, left, right, val);
        // pull
        arr[segment_idx] = config.merge(arr[left_segment_idx], arr[right_segment_idx]);
    }

    int64 range_query(int left, int right) {
        return range_query(0, 0, size, left, right);
    }

    int64 range_query(int segment_idx, int segment_left_bound, int segment_right_bound, int left, int right) {
        // NO OVERLAP
        if (right <= segment_left_bound || segment_right_bound <= left) return config.neutral;
        // COMPLETE OVERLAP
        if (left <= segment_left_bound && segment_right_bound <= right) {
            return arr[segment_idx];
        }
        // PARTIAL OVERLAP
        push(segment_idx, segment_left_bound, segment_right_bound);
        int mid_point = (segment_left_bound + segment_right_bound) >> 1;
        int left_segment_idx = 2 * segment_idx + 1, right_segment_idx = 2 * segment_idx + 2;
        int64 left_res = range_query(left_segment_idx, segment_left_bound, mid_point, left, right);
        int64 right_res = range_query(right_segment_idx, mid_point, segment_right_bound, left, right);
        return config.merge(left_res, right_res);
    }

    void point_set(int i, int64 newVal) {
        point_set(0, 0, size, i, newVal);
    }
    void point_set(int node, int nl, int nr, int i, int64 newVal) {
        if (nr - nl == 1) {
            arr[node] = newVal;
            lazyTag[node] = config.noop; // leaf carries no pending work
            return;
        }
        push(node, nl, nr);
        int m = (nl + nr) >> 1;
        if (i < m) point_set(2*node + 1, nl, m, i, newVal);
        else       point_set(2*node + 2, m, nr, i, newVal);
        arr[node] = config.merge(arr[2*node + 1], arr[2*node + 2]); // pull
    }
};

const int NOOP = 1;
LazySegmentTree::Configuration addMulConfiguration{
    0, 
    NOOP,
    [](int64 x, int64 y) {
        return (x + y) % MOD;
    },
    [](int64 nodeVal, int64 val, int len) {
        return nodeVal * val % MOD;
    },
    [](int64 oldTag, int64 newTag) {
        return oldTag * newTag % MOD;
    }
};

void solve() {
    cin >> N >> D;
    vector<int> P;
    LazySegmentTree seg(N + 1, addMulConfiguration);
    seg.point_set(0, 1);
    for (int i = 0, j = 0; i < N; ++i) {
        int p, o;
        cin >> p >> o;
        P.emplace_back(p);
        while (P[i] - P[j] > D) ++j;
        if (o == 0) { // camper
            // take camper
            seg.update(j + 1, i + 1, 2);
        } else { // coach
            seg.point_set(i + 1, seg.range_query(0, i + 1));
        }
    }
    int ans = seg.range_query(0, N + 1);
    ans--;
    if (ans < 0) ans += MOD;
    cout << ans << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

# USACO 2026, Second Contest, Gold

p1:
ternary search to find the minimum in a bitonic array.


p2:
This is the easiest one, the rule to follow is quite simple.
It is a weird bfs on undirected graph, where you have to follow the rule that you can only traverse edges at each level that are the cheapest edge weight, but also they have to be at least as cheap as any edge weight traversed in previous levels.

p3: 
I think it doesn't matter when bessie rests, and I can always just figure out how much she can rest starting from t = 0.
consider bessie and the nearest farmer.

lets say I know for every node, at what times it is occupied by a farmer.

say the distance to the nearest farmer is 1, she can't rest, but is it guaranteed she is never caught. 
It depends on the following is there ever going to be another farmer 1 distance in front of this farmer. 
there is the single path, take the shortest distance to farmer on that path. and store that. 

## Problem 1. Balancing the Barns

### Solution 1: ternary search, binary search, greedy

This might be on the right path, but it is not the correct solution. 

I don't think you can ternary search on how how many decrements you perform on max(A)

And then compute the new min of B given you decremented all those A.

```cpp
const int64 INF = numeric_limits<int64>::max();
int N;
int64 K;
vector<int64> A, B, C;

bool possible(int64 target, int64 rem) {
    int64 cnt = 0;
    for (int i = 0; i < N; ++i) {
        cnt += max(int64(0), target - C[i]);
        if (cnt > rem) return false;
    }
    return cnt <= rem;
}

int64 calc(int64 d) {
    int64 a = *max_element(A.begin(), A.end()) - d;
    int64 cntA = 0;
    C.assign(N, 0);
    for (int i = 0; i < N; i++) {
        int64 cnt = max(int64(0), A[i] - a);
        cntA += cnt;
        if (cntA > K) return INF;
        C[i] = B[i] + cnt;
    }
    int64 rem = K - cntA;
    int64 bmin = *min_element(C.begin(), C.end());
    int64 bmax = *max_element(C.begin(), C.end());
    int64 lo = bmin, hi = bmax + rem + 1;
    while (lo < hi) {
        int64 mi = lo + (hi - lo) / 2;
        if (possible(mi, rem)) {
            lo = mi + 1;
        } else {
            hi = mi;
        }
    }
    lo--;
    return a - lo;
}

void solve() {
    cin >> N >> K;
    A.assign(N, 0);
    B.assign(N, 0);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    for (int i = 0; i < N; i++) {
        cin >> B[i];
    }
    // the number of decrements to A's max element is between 0 and K
    // ternary search for the minimum
    int64 lo = 0, hi = K + 1;
    while (hi - lo > 3) {
        int64 m1 = lo + (hi - lo) / 3;
        int64 m2 = hi - (hi - lo) / 3;
        if (calc(m1) <= calc(m2)) {
            hi = m2 - 1;
        } else {
            lo = m1 + 1;
        }
    }
    int64 ans = INF;
    for (int64 i = lo; i <= hi; i++) {
        ans = min(ans, calc(i));
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

## Problem 2. Lexicographically Smallest Path

### Solution 1: undirected graph, bfs, set

```cpp
const int INF = numeric_limits<int>::max();
int N, M;
vector<vector<pair<int, int>>> adj;

int coefficient(const char& ch) {
    return ch - 'a';
}

void solve() {
    cin >> N >> M;
    adj.assign(N, vector<pair<int, int>>());
    for (int i = 0; i < M; ++i) {
        int u, v;
        char ch;
        cin >> u >> v >> ch;
        u--, v--;
        int w = coefficient(ch);
        adj[u].emplace_back(v, w);
        adj[v].emplace_back(u, w);
    }
    vector<int> dist(N, INF);
    queue<int> q;
    set<int> nxt;
    q.emplace(0);
    int step = 0, cur = INF;
    while (!q.empty()) {
        int sz = q.size();
        int best = INF;
        nxt.clear();
        for (int i = 0; i < sz; ++i) {
            int u = q.front();
            q.pop();
            dist[u] = step;
            for (auto [v, w] : adj[u]) {
                if (w > cur) continue;
                if (w < best) {
                    best = w;
                    nxt.clear();
                }
                if (w == best) nxt.emplace(v);
            }
        }
        for (int v : nxt) {
            if (dist[v] < INF) continue;
            q.emplace(v);
        }
        cur = best;
        step++;
    }
    for (int i = 0; i < N; ++i) {
        cout << (dist[i] < INF ? dist[i] : -1) << (i + 1 < N ? " " : "");
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


## 

### Solution 1: 

```cpp

```

# USACO 2026, Third Contest, Gold

P1: 
Create a sequence that is doubled of the original, so you take S + S and with this new sequence you can represent a fixed size window of length N, and calculate the number of inversions for each of those windows using the fenwick tree, and making sure to update, then you can know the number of swaps or inversions for each cyclic shift in O(1) time if you precompute number of swaps in O(NlogN) time.

P2:
forward and backward dp on the dag to count maximum, and then take dp1[u] + dp2[u] == K, that is it takes all the flowers. then that nodes u will work.
The dag is constructed based on the bfs, that is you travel to next nodes that are shortest distance from the start, and the backward on dp is just reversing the edges of the dag and doing the same thing, but starting from the end nodes.

P3:

## P1: Good Cyclic Shifts

### Solution 1: number of inversions, fenwick tree, fixed size window, cyclic shifts, line sweep events

```cpp
template <typename T>
struct FenwickTree {
    vector<T> nodes;
    T neutral;

    FenwickTree() : neutral(T(0)) {}

    void init(int n, T neutral_val = T(0)) {
        neutral = neutral_val;
        nodes.assign(n + 1, neutral);
    }

    void update(int idx, T val) {
        while (idx < (int)nodes.size()) {
            nodes[idx] += val;
            idx += (idx & -idx);
        }
    }

    T query(int idx) {
        T result = neutral;
        while (idx > 0) {
            result += nodes[idx];
            idx -= (idx & -idx);
        }
        return result;
    }

    T query(int left, int right) {
        return right >= left ? query(right) - query(left - 1) : T(0);
    }
};

int N;
vector<int> A;

void solve() {
    cin >> N;
    A.assign(N + 1, 0);
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
    }
    int64 g = 0;
    int addCnt = 0, subCnt = 0;
    for (int i = 0; i < N; ++i) {
        if (A[i] >= i + 1) addCnt++;
        else subCnt++;
        g += abs(A[i] - i - 1);
    }
    vector<int> addEvent;
    for (int i = 0; i < N; ++i) {
        if (A[i] < i + 1) {
            addEvent.emplace_back(i - A[i]);
        } else if (A[i] > i + 1) {
            addEvent.emplace_back(i + N - A[i]);
        }
    }
    sort(addEvent.begin(), addEvent.end());
    int M = addEvent.size();
    vector<int64> F(N, 0);
    F[0] = g / 2;
    for (int i = 0, j = 0; i + 1 < N; ++i) {
        g += addCnt - subCnt;
        int v = A[i];
        int delta = N - 2 * v;
        g += delta;
        F[i + 1] = g / 2;
        addCnt--;
        subCnt++;
        while (j < M && addEvent[j] == i) {
            addCnt++;
            subCnt--;
            j++;
        }
    }
    FenwickTree<int> seg;
    seg.init(N);
    int64 inversions = 0;
    for (int i = 0; i < N; ++i) {
        inversions += seg.query(A[i] + 1, N);
        seg.update(A[i], 1);
    }
    vector<int> ans;
    for (int i = 0; i < N; ++i) {
        if (inversions <= F[i]) {
            ans.emplace_back((N - i) % N);
        }
        inversions += N - A[i];
        inversions -= A[i] - 1;
    }
    sort(ans.begin(), ans.end());
    cout << ans.size() << endl;
    for (int i = 0; i < ans.size(); ++i) {
        cout << ans[i];
        if (i + 1 < ans.size()) cout << " ";
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

## P2. Picking Flowers

### Solution 1: forward and backward dp on the dag, bfs, reversing edges

```cpp

```

## P3. Random Tree Generation

### Solution 1: 

```cpp

```

