# UTCP Spring 2024 Open Contest

## 

### Solution 1: 

```py

```

## 

### Solution 1: 

```py

```

## I. Record Compression

### Solution 1:  unbounded knapsack problem with O(n* sqrt(n)) with the constraints

```cpp
const int MAXN = 2e5 + 5;
int N, M;
int items[MAXN];
vector<int> values, weights, dp;

void solve() {
    cin >> N >> M;
    memset(items, 0, sizeof(items));
    for (int i = 0; i < N; i++) {
        int v;
        string s;
        cin >> s >> v;
        items[s.size()] = max(items[s.size()], v);
    }
    for (int i = 1; i < MAXN; i++) {
        if (!items[i]) continue;
        weights.push_back(i);
        values.push_back(items[i]);
    }
    int V = values.size();
    dp.assign(M + 1, 0);
    for (int cap = 0; cap <= M; cap++) {
        for (int i = 0; i < V; i++) {
            if (cap < weights[i]) break;
            dp[cap] = max(dp[cap], dp[cap - weights[i]] + values[i]);
        }
    }
    cout << dp.end()[-1] << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## 

### Solution 1: 

```cpp

```

## 

### Solution 1: 

```cpp

```

## 

### Solution 1: 

```cpp

```

# UTPC Contest 09-13-24 Div. 1 (Advanced)

## C. Spooky Hallway

### Solution 1:  count

```cpp
int N;
string S;
int freq[2];

void solve() {
    cin >> N >> S;
    int prv = -1;
    for (int i = 0; i < N; i++) {
        int cur = S[i] - '0';
        if (prv != cur) {
            freq[cur]++;
        }
        prv = cur;
    }
    int ans = min(freq[0], freq[1]);
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

## D. Nightmare on 24th

### Solution 1:  prefix sum, binary search

```cpp
int N, M;
vector<int> A, B, psum;

void solve() {
    cin >> N >> M;
    A.resize(N);
    B.resize(M);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    for (int i = 0; i < M; i++) {
        cin >> B[i];
    }
    psum.resize(N);
    for (int i = 0; i < N; i++) {
        psum[i] = A[i];
        if (i > 0) {
            psum[i] += psum[i - 1];
        }
    }
    for (int x : B) {
        if (x == 0) {
            cout << 0 << endl;
            continue;
        }
        int ans =  lower_bound(psum.begin(), psum.end(), x) - psum.begin();
        if (ans == N) {
            cout << -1 << endl;
        } else {
            cout << ++ans << endl;
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

## E. Candy Eating

### Solution 1:  sort descending order, greedy

1. Pack it horizontally, where X is the width


```cpp
int N, D, X;
vector<int> counts, values;
vector<pair<int, int>> candies; // (value, count)

void solve() {
    cin >> N >> D >> X;
    counts.resize(N);
    values.resize(N);
    candies.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> counts[i];
    }
    for (int i = 0; i < N; i++) {
        cin >> values[i];
    }
    for (int i = 0; i < N; i++) {
        candies[i] = {values[i], counts[i]};
    }
    int ans = 0, day = 0;
    sort(candies.begin(), candies.end(), greater<pair<int, int>>());
    for (auto &[x, c] : candies) {
        int start = day;
        int take = min(c, D - day);
        day += take;
        ans += x * take;
        c -= take;
        if (day == D) {
            day = 0;
            X--;
        }
        if (!X) break;
        take = min(start, c);
        ans += x * take;
        day += take;
    }
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

## F. Haunted House

### Solution 1:  multisource bfs, undirected graph

```cpp
const int INF = 1e18;
int N, M, S, K, G;
vector<vector<int>> adj;
vector<int> exits, ghosts, distg, distp;

void bfs(vector<int>& dist, const vector<int>& starts) {
    queue<int> q;
    for (int start : starts) {
        dist[start] = 0;
        q.push(start);
    }
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (int v : adj[u]) {
            if (dist[v] == INF) {
                dist[v] = dist[u] + 1;
                q.push(v);
            }
        }
    }
}

void solve() {
    cin >> N >> M >> S >> K >> G;
    S--;
    adj.assign(N, vector<int>());
    for (int i = 0; i < M; i++) {
        int u, v;
        cin >> u >> v;
        u--, v--;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    exits.resize(K);
    for (int i = 0; i < K; i++) {
        cin >> exits[i];
        exits[i]--;
    }
    ghosts.resize(G);
    for (int i = 0; i < G; i++) {
        cin >> ghosts[i];
        ghosts[i]--;
    }
    distg.assign(N, INF);
    distp.assign(N, INF);
    bfs(distg, ghosts);
    vector<int> person = {S};
    bfs(distp, person);
    int ans = 0;
    for (int e : exits) {
        if (distp[e] < distg[e]) ans++;
    }
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

## G. Pumpkin Patch

### Solution 1:  bfs, bitmask, state is (r, c, mask, unused corn), grid

```cpp
enum Cell {
    EMPTY,
    PUMPKIN,
    START,
    END,
    JACK,
    CORN
};

struct Location {
    int r, c, mask, cnt;
    Location() {}
    Location(int r, int c, int mask, int cnt) : r(r), c(c), mask(mask), cnt(cnt) {}
};

const string SPOOKY = "SPOOKED!";
const int INF = 1e9;
int R, C, N;
vector<vector<Cell>> grid;
map<pair<int, int>, int> corn;
int dist[100][100][1 << 8][9];

vector<pair<int, int>> neighborhood(int r, int c) {
    return {{r - 1, c}, {r + 1, c}, {r, c - 1}, {r, c + 1}};
}

bool in_bounds(int r, int c) {
    return 0 <= r && r < R && 0 <= c && c < C;
}

void solve() {
    cin >> R >> C;
    grid.assign(R, vector<Cell>(C, EMPTY));
    N = 0;
    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C; c++) {
            for (int mask = 0; mask < (1 << 8); mask++) {
                for (int j = 0; j <= 8; j++) {
                    dist[r][c][mask][j] = INF;
                }
            }
        }
    }
    queue<Location> q;
    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C; c++) {
            char cell;
            cin >> cell;
            if (cell == 'P') {
                grid[r][c] = PUMPKIN;
            } else if (cell == 'S') {
                grid[r][c] = START;
                q.emplace(r, c, 0, 0);
                dist[r][c][0][0] = 0;
            } else if (cell == 'E') {
                grid[r][c] = END;
            } else if (cell == 'J') {
                grid[r][c] = JACK;
            } else if (cell == 'C') {
                grid[r][c] = CORN;
                corn[{r, c}] = N;
                N++;
            }
        }
    }
    // (r, c, mask, unused corn)
    while (!q.empty()) {
        auto [r, c, mask, cnt] = q.front();
        q.pop();
        if (grid[r][c] == END) {
            cout << dist[r][c][mask][cnt] << endl;
            return;
        }
        for (auto [nr, nc] : neighborhood(r, c)) {
            if (!in_bounds(nr, nc) || grid[nr][nc] == PUMPKIN) {
                continue;
            }
            if (grid[nr][nc] == JACK) {
                if (cnt > 0) {
                    if (dist[nr][nc][mask][cnt - 1] == INF) {
                        dist[nr][nc][mask][cnt - 1] = dist[r][c][mask][cnt] + 1;
                        q.emplace(nr, nc, mask, cnt - 1);
                    }
                }
            } else if (grid[nr][nc] == CORN) {
                int cid = corn[{nr, nc}];
                if (!((mask >> cid) & 1)) {
                    if (dist[nr][nc][mask | (1 << cid)][cnt + 1] == INF) {
                        dist[nr][nc][mask | (1 << cid)][cnt + 1] = dist[r][c][mask][cnt] + 1;
                        q.emplace(nr, nc, mask | (1 << cid), cnt + 1);
                    }
                }
            } else {
                if (dist[nr][nc][mask][cnt] == INF) {
                    dist[nr][nc][mask][cnt] = dist[r][c][mask][cnt] + 1;
                    q.emplace(nr, nc, mask, cnt);
                }
            }
        }
    }
    cout << SPOOKY << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## H. Speedway Evacuation

### Solution 1:  binary search, sorting, probability

```cpp
const int INF = 1e9;
int N, Q, threshold;
vector<int> pos;

void solve() {
    cin >> N >> Q;
    threshold = INF;
    for (int i = 0; i < N; i++) {
        int u, v;
        cin >> u;
        v = N - u; 
        if (u > v) swap(u, v);
        pos.push_back(u);
        threshold = min(threshold, v);
    }
    sort(pos.begin(), pos.end());
    while (Q--) {
        int q;
        cin >> q;
        if (q >= threshold) {
            cout << -1 << endl;
        } else {
            int i = upper_bound(pos.begin(), pos.end(), q) - pos.begin();
            cout << i << endl;
        }
    }
}

signed main() {
    solve();
    return 0;
}
```

## I. Trick or Treat

### Solution 1:  dynamic programming, dijkstra, priority queue, interval dp

```cpp
struct State {
    int l, r, pos, v;
    State() {}
    State(int l, int r, int pos, int v) : l(l), r(r), pos(pos), v(v) {}
    bool operator<(const State& other) const {
        return v < other.v;
    }
};

int N, K;
vector<int> A;
priority_queue<State> minheap;

int length(int l, int r) {
    if (l > r) swap(l, r);
    return r - l + 1;
}

void solve() {
    cin >> N >> K;
    K--;
    A.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    minheap.emplace(K, K, K, 0);
    while (!minheap.empty()) {
        auto [l, r, pos, v] = minheap.top();
        minheap.pop();
        if (length(l, r) == N) {
            cout << v << endl;
            return;
        }
        if (r + 1 < N && A[r + 1] > v + length(pos, r)) minheap.emplace(l, r + 1, r + 1, v + length(pos, r));
        if (l - 1 >= 0 && A[l - 1] > v + length(pos, l)) minheap.emplace(l - 1, r, l - 1, v + length(pos, l));
    }
    cout << -1 << endl;
}

signed main() {
    solve();
    return 0;
}
```

## J. Phantom Poker

### Solution 1:  segment tree, combinatorics

point updates
range queries

```cpp
const int M = 1e9 + 7;
int N, Q;
vector<int> deck;

struct SegmentTree {
    int size;
    int neutral = 0;
    vector<vector<int>> nodes;

    void init(int num_nodes) {
        size = 1;
        while (size < num_nodes) size *= 2;
        nodes.assign(size * 2, vector<int>(13, 0));
        for (int i = 0; i < size * 2; i++) {
            nodes[i][1] = 1;
        }
    }

    vector<int> combine(const vector<int>& a, const vector<int>& b) {
        vector<int> res(13, neutral);
        for (int i = 1; i < 13; i++) {
            for (int j = 1; j < 13; j++) {
                int add = a[i] * b[j] % M;
                res[(i * j) % 13] = (res[(i * j) % 13] + add) % M;
            }
        }
        return res;
    }

    void ascend(int segment_idx) {
        while (segment_idx > 0) {
            int left_segment_idx = 2 * segment_idx, right_segment_idx = 2 * segment_idx + 1;
            nodes[segment_idx] = combine(nodes[left_segment_idx], nodes[right_segment_idx]);
            segment_idx >>= 1;
        }
    }
    // this is for assign, for addition change to += val
    void update(int segment_idx, int val) {
        segment_idx += size;
        for (int i = 0; i < 13; i++) {
            nodes[segment_idx][i] = 0;
        }
        nodes[segment_idx][1]++;
        nodes[segment_idx][val]++;
        segment_idx >>= 1;
        ascend(segment_idx);
    }

    int query(int left, int right) {
        left += size, right += size;
        vector<int> res(13, neutral);
        res[1] = 1;
        while (left <= right) {
            if (left & 1) {
                vector<int> res1 = combine(res, nodes[left]);
                swap(res, res1);
                left++;
            }
            if (~right & 1) {
                vector<int> res1 = combine(res, nodes[right]);
                swap(res, res1);
                right--;
            }
            left >>= 1, right >>= 1;
        }
        return res[5];
    }
};

SegmentTree seg;

void solve() {
    cin >> N >> Q;
    deck.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> deck[i];
    }
    seg.init(N);
    for (int i = 0; i < N; i++) {
        seg.update(i, deck[i]);
    }
    while (Q--) {
        int q, l, r;
        cin >> q >> l >> r;
        if (q == 1) {
            seg.update(l - 1, r);
        } else {
            cout << seg.query(l - 1, r - 1) << endl;
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

# UTPC Contest 11-20-24 Div. 1 (Advanced)

## 

### Solution 1: 

```cpp

```

## 

### Solution 1: 

```cpp

```

## E. Crossroads

### Solution 1: 

```cpp

```

## F. Tipsy Chick

### Solution 1:  hamiltonian path, bitmask dynamic programming, undirected graph

1. Visit each node is hamiltonian path
1. form a linear graph will minimize the maximum distance and connect them all in the fewest number of rounds
1. The maximum number of rounds is 2, you can always get it done in 2 rounds.

```cpp
const int INF = 1e18;
int N;
vector<vector<int>> dp;
vector<pair<int, int>> points;
 
int squaredDistance(int x1, int y1, int x2, int y2) {
    return (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
}
 
bool isSet(int mask, int v) {
    return (mask >> v) & 1;
}
 
void solve() {
    cin >> N;
    int endMask = 1 << N;
    dp.assign(endMask, vector<int>(N, INF));
    points.resize(N);
    for (int i = 0; i < N; i++) {
        int x, y;
        cin >> x >> y;
        points[i] = {x, y};
        dp[1 << i][i] = 0;
    }
    for (int mask = 1; mask < endMask; mask++) {
        for (int u = 0; u < N; u++) {
            if (dp[mask][u] == INF) continue;
            for (int v = 0; v < N; v++) {
                if (isSet(mask, v)) continue;
                int nmask = mask | (1 << v);
                auto [x1, y1] = points[u];
                auto [x2, y2] = points[v];
                int dist = max(dp[mask][u], squaredDistance(x1, y1, x2, y2));
                dp[nmask][v] = min(dp[nmask][v], dist);
            }
        }
    }
    int ans = *min_element(dp[endMask - 1].begin(), dp[endMask - 1].end());
    cout << (N > 2 ? 2 : 1) << " " << ans << endl;
}
 
signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## 

### Solution 1: 

```cpp

```

## 

### Solution 1: 

```cpp

```

## H. The Fo Sho

### Solution 1:  disjoint sets, factorials, combinatorics

```cpp
const int MAXN = 2e5 + 5, MOD = 998244353;
int N, M;

int inv(int i, int m) {
  return i <= 1 ? i : m - (int)(m/i) * inv(m % i, m) % m;
}

vector<int> fact, inv_fact;

void factorials(int n, int m) {
    fact.assign(n + 1, 1);
    inv_fact.assign(n + 1, 0);
    for (int i = 2; i <= n; i++) {
        fact[i] = (fact[i - 1] * i) % m;
    }
    inv_fact.end()[-1] = inv(fact.end()[-1], m);
    for (int i = n - 1; i >= 0; i--) {
        inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % m;
    }
}

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
    cin >> N >> M;
    UnionFind dsu(N);
    int countGroups = N;
    int ans = fact[N];
    while (M--) {
        int a, b;
        cin >> a >> b;
        a--; b--;
        int ga = dsu.find(a), gb = dsu.find(b);
        int sa = dsu.size[ga], sb = dsu.size[gb];
        if (ga != gb) {
            ans = ans * inv(countGroups, MOD) % MOD;
            ans = ans * inv(fact[sa], MOD) % MOD;
            ans = ans * inv(fact[sb], MOD) % MOD;
            ans = ans * fact[sa + sb] % MOD;
            countGroups--;
            dsu.same(a, b);
        }
        cout << countGroups << " " << ans << endl;
    }
}

signed main() {
    factorials(MAXN, MOD);
    solve();
    return 0;
}
```

## 

### Solution 1: 

```cpp

```

# UTPC Contest 1-29-25 Div. 1 (Advanced)

## Lion Dancers

### Solution 1:  combinatorics, counting, independent events, factorials

```cpp
const int64 MOD = 1e9 + 7, MAXN = 1e5 + 5;
int N, M;

int64 inv(int i, int64 m) {
  return i <= 1 ? i : m - (m / i) * inv(m % i, m) % m;
}

vector<int64> fact, inv_fact;

void factorials(int n, int64 m) {
    fact.assign(n + 1, 1);
    inv_fact.assign(n + 1, 0);
    for (int i = 2; i <= n; i++) {
        fact[i] = (fact[i - 1] * i) % m;
    }
    inv_fact.end()[-1] = inv(fact.end()[-1], m);
    for (int i = n - 1; i >= 0; i--) {
        inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % m;
    }
}

int64 choose(int n, int r, int64 m) {
    if (n < r) return 0;
    return (fact[n] * inv_fact[r] % m) * inv_fact[n - r] % m;
}

void solve() {
    cin >> N >> M;
    int ans = 1;
    for (int i = 0; i < M; i++) {
        int k;
        cin >> k;
        ans = (ans * choose(k, N, MOD)) % MOD;
    }
    cout << ans << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    factorials(MAXN, MOD);
    solve();
    return 0;
}
```

## Lunar Phases

### Solution 1:  geometry, math, dot product, vectors

1. The key idea is if you take the vector and the perpendicular vector to it, you can determine which quadrant the point that represents the moon is in by using the dot product of the two vectors.

```cpp
int64 dotProduct(int64 x1, int64 y1, int64 x2, int64 y2) {
    return x1 * x2 + y1 * y2;
}

void solve() {
    int64 sx, sy, mx, my;
    cin >> sx >> sy >> mx >> my;
    int64 px = -sy, py = sx;
    int64 dot1 = dotProduct(sx, sy, mx, my);
    int64 dot2 = dotProduct(px, py, mx, my);
    if (dot1 > 0 && dot2 > 0) {
        cout << "Third quarter" << endl;
    } else if (dot1 > 0 && dot2 < 0) {
        cout << "Full moon" << endl;
    } else if (dot1 < 0 && dot2 > 0) {
        cout << "New moon" << endl;
    } else {
        cout << "First quarter" << endl;
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

## Lantern Hopping

### Solution 1:  max heap, greedy

1. The observation is you just need the maximum height over the entire segment, and the required energy to hop everywhere is the how much lower you are from the max height in the entire segment.

```cpp
int N, Q;
vector<int> A;

void solve() {
    cin >> N >> Q;
    A.resize(N);
    priority_queue<pair<int, int>> maxheap;
    for (int i = 0; i < N; i++) {
        cin >> A[i];
        maxheap.emplace(A[i], i);
    }
    while (Q--) {
        int t, i;
        cin >> t >> i;
        --i;
        while (maxheap.top().first != A[maxheap.top().second]) maxheap.pop();
        if (t == 1) {
            int maxHeight = maxheap.top().first;
            int ans = maxHeight - A[i];
            cout << ans << endl;
        } else {
            int h;
            cin >> h;
            A[i] = h;
            maxheap.emplace(h, i);
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

## Dragon Scales

### Solution 1:  line sweep, math

1. LIne sweep from left to right works well if you figure out how to update when starting a new segment, and when ending a segment (everything it contributes must be undone)

```cpp
int N, M;
vector<vector<int>> starts, ends_;
vector<int64> P, L;

void solve() {
    cin >> N >> M;
    starts.assign(M + 1, vector<int>());
    ends_.assign(M + 2, vector<int>());
    P.resize(N);
    L.resize(N);
    for (int i = 0; i < N; i++) {
        int l, r, p;
        cin >> l >> r >> p;
        P[i] = p;
        L[i] = r - l + 1;
        starts[l].emplace_back(i);
        ends_[r + 1].emplace_back(i);
    }
    int64 ans = 0, baseSum = 0;
    for (int i = 1; i <= M; i++) {
        for (int j : starts[i]) baseSum += P[j];
        for (int j : ends_[i]) {
            baseSum -= P[j];
            ans -= L[j] * P[j];
        }
        ans += baseSum;
        cout << ans << " ";
    }
    cout << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Sally's Stroll (Easy Version)

### Solution 1: grid, undirected graph, connected components, bfs, combinatorics

1. precomputation to help with construction of undirected graph is hard to figure out and implement, but you need to mark passagble horizontal and vertical segments in two directions.  This needs to be precomputed, so you know which pair of two movements works to add connect two vertices with an edge.

```cpp
int R, C, KH, KV, sz, Q;
vector<vector<char>> grid;
vector<vector<int>> adj;
vector<bool> vis;

int map2Dto1D(int r, int c) {
    return r * C + c;
}

int64 countPairs(int64 n) {
    return n * (n - 1);
}

void dfs(int u) {
    if (vis[u]) return;
    vis[u] = true;
    sz++;
    for (int v : adj[u]) {
        dfs(v);
    }
}

void solve() {
    cin >> R >> C >> KV >> KH;
    grid.assign(R, vector<char>(C));
    adj.assign(R * C, vector<int>());
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            cin >> grid[i][j];
        }
    }
    cin >> Q;
    vector<vector<vector<bool>>> segments(R, vector<vector<bool>>(C, vector<bool>(4, false)));
    // horizontal segments
    for (int i = 0; i < R; i++) {
        int dist = 0;
        for (int j = 0; j < C; j++) {
            if (grid[i][j] == '*') dist++;
            else dist = 0;
            if (dist >= KH + 1) {
                segments[i][j][2] = true;
                segments[i][j - KH][0] = true;
            }
        }
    }
    // vertical segments
    for (int j = 0; j < C; j++) {
        int dist = 0;
        for (int i = 0; i < R; i++) {
            if (grid[i][j] == '*') dist++;
            else dist = 0;
            if (dist >= KV + 1) {
                segments[i][j][3] = true;
                segments[i - KV][j][1] = true;
            }
        }
    }
    // build graph
    vector<int> dr = {0, 1, 0, -1}, dc = {1, 0, -1, 0};
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            for (int d1 = 0; d1 < 4; d1++) {
                if (!segments[i][j][d1]) continue;
                int r = i + dr[d1] * KV, c = j + dc[d1] * KH;
                for (int d2 = 0; d2 < 4; d2++) {
                    if (!segments[r][c][d2]) continue;
                    int r1 = r + dr[d2] * KV, c1 = c + dc[d2] * KH;
                    int u = map2Dto1D(i, j), v = map2Dto1D(r1, c1);
                    adj[u].emplace_back(v);
                    adj[v].emplace_back(u);
                }
            }
        }
    }
    // explore connected components
    int64 ans = 0;
    vis.assign(R * C, false);
    for (int i = 0; i < R * C; i++) {
        if (vis[i]) continue;
        sz = 0;
        dfs(i);
        ans += countPairs(sz);
    }
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

## Red Envelope

### Solution 1:  reduction to nim game, xor sum, parity

1. You have to realize that if you have an even number of coins in a an envelope that is worthless, and basically it cancels out so the second player wins.  So reduce those to 0
1. If you have odd number of coins, now it gives a nim advantage because if there is just one than first player can pick it and win now. 
1. But the thing is you have frequency of these odd number of coins, and you can reduce it to be that the odd number represents a pile and the frequency is the size of the pile.  And now it is a standard nim gam, and can solve with xor sum, it must be non-zero for first player to win. 

```cpp
int N;
map<int, int> freq;

void solve() {
    cin >> N;
    for (int i = 0, x; i < N; i++) {
        cin >> x;
        freq[x]++;
    }
    int xorSum = 0;
    for (auto [x, f] : freq) {
        if (x & 1) xorSum ^= f;
    }
    if (xorSum > 0) {
        cout << "Ai" << endl;
    } else {
        cout << "Bo" << endl;
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

# ????

## 

### Solution 1: 

```cpp

```

## 

### Solution 1: 

```cpp

```

## 

### Solution 1: 

```cpp

```

## 

### Solution 1: 

```cpp

```

## 

### Solution 1: 

```cpp

```