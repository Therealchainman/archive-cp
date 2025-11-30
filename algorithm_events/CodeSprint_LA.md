# CodeSprint LA

# CodeSprint LA 2024 - Open

## Catbus Planning

### Solution 1:  undirected graph, eulerian paths, dfs, visit every edge exactly once with k entities, eulerian circuits, vertex degrees

```cpp
int N, M, K, bus;
vector<vector<pair<int, int>>> adj;
vector<vector<int>> routes;
vector<int> deg;
vector<bool> vis;

bool dfs(int u) {
    routes[bus].push_back(u + 1);
    for (auto [v, i] : adj[u]) {
        if (vis[i]) continue;
        vis[i] = true;
        dfs(v);
        return true;
    }
    return false;
}

void solve() {
    cin >> N >> M >> K;
    adj.assign(N, vector<pair<int, int>>());
    deg.assign(N, 0);
    for (int i = 0; i < M; i++) {
        int u, v;
        cin >> u >> v;
        u--, v--;
        adj[u].emplace_back(v, i);
        adj[v].emplace_back(u, i);
        deg[u]++;
        deg[v]++;
    }
    int odd_count = 0;
    for (int i = 0; i < N; i++) {
        if (deg[i] & 1) odd_count++;
    }
    if (K > M || K < odd_count / 2) {
        cout << "Impossible" << endl;
        return;
    }
    int count = 0;
    routes.assign(K, vector<int>());
    bus = 0;
    vis.assign(M, false);
    for (int i = 0; i < N; i++) {
        if (bus == K) break;
        if (deg[i] & 1) {
            if (dfs(i)) {
                count += routes[bus].size();
                bus++;
            } else {
                routes[bus].clear();
            }
        }
    }
    for (int i = 0; i < N; i++) {
        if (bus == K) break;
        if (deg[i] % 2 == 0) {
            if (dfs(i)) {
                count += routes[bus].size();
                bus++;
            } else {
                routes[bus].clear();
            }
        }
    }
    if (count < M) {
        cout << "Impossible" << endl;
        return;
    }
    int i = 0;
    while (bus < K) {
        while (i < bus && routes[i].size() <= 2) {
            i++;
        }
        int u = routes[i].end()[-1];
        routes[i].pop_back();
        routes[bus].push_back(routes[i].end()[-1]);
        routes[bus].push_back(u);
        bus++;
    }
    cout << "Possible" << endl;
    for (int i = 0; i < K; i++) {
        for (int u : routes[i]) {
            cout << u << " ";
        }
        cout << endl;
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

## Finding Laputa

### Solution 1: 

not sure, this is tle, but how can I do better. 

```cpp
const int INF = 1e9;
int n, c1, c2;
vector<vector<pair<int, int>>> adj;
vector<int> dist, take;

pair<int, int> dijkstra(int src, int dst) {
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    pq.emplace(0, src);
    take[src] = 0;
    while (!pq.empty()) {
        auto [cost, u] = pq.top();
        pq.pop();
        if (u == dst) return {cost, take[u]};
        if (cost > dist[u]) continue;
        for (auto [v, w] : adj[u]) {
            if (cost + w < dist[v]) {
                c1 = __builtin_popcount(u);
                c2 = __builtin_popcount(v);
                if (c2 > c1 + 1) {
                    take[v] = take[u] + 1;
                } else {
                    take[v] = take[u];
                }
                pq.emplace(cost + w, v);
            }
        }
    }
    return {-1, -1};
}

void solve() {
    cin >> n;
    adj.assign(1 << n, vector<pair<int, int>>());
    int num_edges = n * (1 << (n - 1));
    for (int i = 0; i < num_edges; i++) {
        int u, v, w;
        cin >> u >> v >> w;
        adj[u].emplace_back(v, w);
        adj[v].emplace_back(u, w);
    }
    for (int mask = 0; mask < (1 << n); mask++) {
        int s = mask;
        c1 = __builtin_popcount(mask);
        while (s > 0) {
            s = (s - 1) & mask;
            c2 = __builtin_popcount(s);
            if (c2 + 1 == c1) continue;
            int w = 1 << (c1 - c2);
            adj[mask].emplace_back(s, w);
            adj[s].emplace_back(mask, w);
        }
    }
    dist.assign(1 << n, INF);
    take.assign(1 << n, INF);
    pair<int, int> ans = dijkstra(0, (1 << n) - 1);
    cout << ans.first << endl;
    cout << ans.second << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Kodama Hierarchy

### Solution 1:  2D LIS algorithm

```cpp

```

## Mid Card

### Solution 1: 

```py

```

## Lights, Camera, Airplane

### Solution 1: 

```py

```

## Miyazaki's Masterpiece

### Solution 1:  trie data structure, fixed size window, stack

```cpp
int N, M, L;

struct Node {
    int children[26];
    bool isLeaf;
    void init() {
        memset(children,0,sizeof(children));
        isLeaf = false;
    }
};
struct Trie {
    vector<Node> trie;
    void init() {
        Node root;
        root.init();
        trie.push_back(root);
    }
    void insert(deque<char>& s) {
        int cur = 0;
        for (char &c : s) {
            int i = c-'A';
            if (trie[cur].children[i]==0) {
                Node root;
                root.init();
                trie[cur].children[i] = trie.size();
                trie.push_back(root);
            }
            cur = trie[cur].children[i];
        }
        trie[cur].isLeaf= true;
    }
    bool search(string& s) {
        stack<pair<int, int>> stk, nstk;
        stk.emplace(0, 0);
        for (char &c : s) {
            int i = c-'A';
            while (!stk.empty()) {
                auto [cur, cnt] = stk.top();
                stk.pop();
                if (trie[cur].children[i]) {
                    nstk.emplace(trie[cur].children[i], cnt);
                } else if (cnt < 2) {
                    for (int j = 0; j < 26; j++) {
                        if (trie[cur].children[j]) {
                            nstk.emplace(trie[cur].children[j], cnt + 1);
                        }
                    }
                }
            }
            swap(stk, nstk);

        }
        bool res = false;
        while (!stk.empty()) {
            auto [cur, cnt] = stk.top();
            stk.pop();
            res |= trie[cur].isLeaf;
        }
        return res;
    }
};

void solve() {
    cin >> N >> M >> L;
    string s;
    cin >> s;
    Trie trie;
    trie.init();
    int ans = 0;
    deque<char> buffer;
    for (char ch : s) {
        buffer.push_back(ch);
        if (buffer.size() == L) {
            trie.insert(buffer);
            buffer.pop_front();
        } 
    }
    for (int i = 0; i < M; i++) {
        string pat;
        cin >> pat;
        if (trie.search(pat)) ans++;
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

# CodeSprint LA 2025 - Open

## B - Arcane Secret

### Solution 1: 

```cpp
int N, K;
vector<int> A;

int ceil(int x, int y) {
    return (x + y - 1) / y;
}

int floor(int x, int y) {
    return x / y;
}

void solve() {
    cin >> N >> K;
    A.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    sort(A.begin(), A.end());
    int l = N / K * ceil(K, 2) - 1, r = N - floor(K, 2) - 1;
    while (l > 0 && A[l - 1] == A[l]) l--;
    while (r + 1 < N && A[r + 1] == A[r]) r++;
    int ans = r - l + 1;
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

## D - Chemtech Contagion

### Solution 1: 

```cpp

```

## L - Topside vs Zaun

### Solution 1:  subset sum, knapsack, dp, bitsets, randomization

```cpp
const int INF = 1e9;
int N;
vector<int> A, ssum;
vector<map<pair<int, int>, int>> dp;
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

int dfs(int idx, int sz, int sum) {
    if (abs(sz) > N - idx) return -INF;
    if (abs(sum) > ssum[idx]) return -INF;
    if (idx == N) {
        if (sz == 0 && sum == 0) return 0;
        return -INF;
    }
    if (dp[idx].count({sz, sum})) return dp[idx][{sz, sum}];
    int ans = dfs(idx + 1, sz, sum);
    int teamA = dfs(idx + 1, sz + 1, sum + A[idx]);
    int teamB = dfs(idx + 1, sz - 1, sum - A[idx]);
    if (teamA != -INF) {
        ans = max(ans, teamA + 1);
    }
    if (teamB != -INF) {
        ans = max(ans, teamB + 1);
    }
    return dp[idx][{sz, sum}] = ans;
}

void solve() {
    cin >> N;
    A.resize(N);
    ssum.assign(N + 1, 0);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    shuffle(A.begin(), A.end(), rng);
    for (int i = N - 1; i >= 0; i--) {
        ssum[i] = ssum[i + 1] + A[i];
    }
    dp.assign(N, map<pair<int, int>, int>());
    int ans = dfs(0, 0, 0);
    if (ans == -INF) {
        cout << -1 << endl;
        return;
    }
    cout << ans / 2 << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

```cpp
using ull = unsigned long long;

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int N;
    cin >> N;
    vector<int> a(N);
    long long total = 0;
    for(int i = 0; i < N; i++){
        cin >> a[i];
        total += a[i];
    }

    // randomize to avoid adversarial DP behavior
    mt19937 rng(chrono::high_resolution_clock::now().time_since_epoch().count());
    shuffle(a.begin(), a.end(), rng);

    // We'll keep dp[p] = bit‐vector of all achievable (sum1 - sum2) differences
    // after assigning exactly p people to the two teams (p = c1 + c2).
    // We only need to detect dp[2k][0] (difference == 0) for k from N/2 down to 1.
    int W = (int)total;
    int D = 2*W + 1;              // differences from -W ... +W
    int B = (D + 63) >> 6;        // number of 64-bit words

    // dp[p][b] is the b-th 64-bit block of the bit-vector for p picks
    vector< vector<ull> > dp(N+1, vector<ull>(B, 0ULL));
    auto setBit = [&](vector<ull>& v, int pos){
        v[pos>>6] |= (1ULL << (pos & 63));
    };
    auto testBit = [&](const vector<ull>& v, int pos){
        return (v[pos>>6] >> (pos & 63)) & 1;
    };

    // initialize: with 0 picks, difference=0 is possible
    int OFF = W;
    setBit(dp[0], OFF);

    // temporary buffers for shifts
    vector<ull> tmpL(B), tmpR(B);

    // shift‐left by shift (adds to difference)
    auto shl = [&](const vector<ull>& src, int shift, vector<ull>& dst){
        int w = shift >> 6, b = shift & 63;
        fill(dst.begin(), dst.end(), 0ULL);
        for(int i = 0; i < B; i++){
            int j = i + w;
            if(j < B){
                dst[j] |= src[i] << b;
                if(b && j+1 < B)
                    dst[j+1] |= src[i] >> (64 - b);
            }
        }
    };
    // shift‐right by shift (subtracts from difference)
    auto shr = [&](const vector<ull>& src, int shift, vector<ull>& dst){
        int w = shift >> 6, b = shift & 63;
        fill(dst.begin(), dst.end(), 0ULL);
        for(int i = 0; i < B; i++){
            int j = i + w;
            if(j < B){
                dst[i] |= src[j] >> b;
                if(b && j+1 < B)
                    dst[i] |= src[j+1] << (64 - b);
            }
        }
    };

    // Run the DP: for each person, either assign to team1 (d+=a[i]),
    // assign to team2 (d-=a[i]), or leave unassigned.
    for(int x: a){
        for(int p = N-1; p >= 0; p--){
            // from dp[p] we can go to dp[p+1] by either shl or shr
            shl(dp[p], x, tmpL);
            shr(dp[p], x, tmpR);
            auto &dst = dp[p+1];
            for(int w = 0; w < B; w++){
                dst[w] |= (tmpL[w] | tmpR[w]);
            }
        }
    }

    // Now check for the largest k for which dp[2k][0] is true
    int answer = 0;
    for(int k = N/2; k >= 1; k--){
        if(testBit(dp[2*k], OFF)){
            answer = k;
            break;
        }
    }

    cout << answer << "\n";
    return 0;
}

```