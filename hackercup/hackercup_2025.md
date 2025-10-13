# Meta Hacker Cup 2025

# Practice Round

## Warm up

### Solution 1: sorting, reverse iteration, hash map

```cpp
int N;
vector<int> A, B;

void solve() {
    cin >> N;
    A.assign(N, 0);
    B.assign(N, 0);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    for (int i = 0; i < N; i++) {
        cin >> B[i];
    }
    vector<pair<int, int>> pairs;
    for (int i = 0; i < N; i++) {
        pairs.emplace_back(A[i], i);
    }
    sort(pairs.rbegin(), pairs.rend());
    vector<int> last(N + 1, -1);
    vector<pair<int, int>> ans;
    for (const auto &[value, index] : pairs) {
        int target = B[index];
        if (value > target) {
            cout << -1 << endl;
            return;
        }
        if (value != target && last[target] == -1) {
            cout << -1 << endl;
            return;
        }
        if (value != target) {
            ans.emplace_back(index, last[target]);
        }
        last[value] = index;
    }
    reverse(ans.begin(), ans.end());
    cout << ans.size() << endl;
    for (const auto &[i, j] : ans) {
        cout << i + 1 << " " << j + 1 << endl;
    }
}
```

## Zone in

### Solution 1: multisource bfs on grid, flood fill with bfs

```cpp
int R, C, S;
vector<vector<char>> grid;

bool inBounds(int r, int c) {
    return r >= 0 && r < R && c >= 0 && c < C;
}

int bfs(int r, int c) {
    queue<pair<int, int>> q;
    q.emplace(r, c);
    grid[r][c] = '#';
    int ans = 0;
    while (!q.empty()) {
        auto [r, c] = q.front();
        q.pop();
        ans++;
        for (int dr = -1; dr <= 1; ++dr) {
            for (int dc = -1; dc <= 1; ++dc) {
                if (abs(dr) + abs(dc) != 1) continue;
                int nr = r + dr, nc = c + dc;
                if (!inBounds(nr, nc) || grid[nr][nc] == '#') continue;
                grid[nr][nc] = '#';
                q.emplace(nr, nc);
            }
        }
    }
    return ans;
}

void solve() {
    cin >> R >> C >> S;
    R += 2, C += 2;
    grid.assign(R, vector<char>(C, '#'));
    for (int i = 0; i < R - 2; ++i) {
        for (int j = 0; j < C - 2; ++j) {
            cin >> grid[i + 1][j + 1];
        }
    }
    queue<pair<int, int>> q;
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            if (grid[i][j] == '#') {
                q.emplace(i, j);
            }
        }
    }
    for (int i = 0; i < S; ++i) {
        int sz = q.size();
        while (sz--) {
            auto [r, c] = q.front();
            q.pop();
            for (int dr = -1; dr <= 1; ++dr) {
                for (int dc = -1; dc <= 1; ++dc) {
                    if (abs(dr) + abs(dc) != 1) continue;
                    int nr = r + dr, nc = c + dc;
                    if (!inBounds(nr, nc) || grid[nr][nc] == '#') continue;
                    grid[nr][nc] = '#';
                    q.emplace(nr, nc);
                }
            }
        }
    }
    int ans = 0;
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            if (grid[i][j] == '#') continue;
            ans = max(ans, bfs(i, j));
        }
    }
    cout << ans << endl;
}
```

## Monkey Around

### Solution 1: constructive, modular arithmetic, weakly decreasing array

```cpp
int N;
vector<int> A;

void solve() {
    cin >> N;
    A.assign(N, 0);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    vector<int> seg, rot;
    int prv = N + 1, start = 1, seenOne = true;
    for (int x : A) {
        if (x == 1 && !seenOne) {
            seenOne = true;
        } else if (x == start || x != prv + 1) {
            seg.emplace_back(0);
            rot.emplace_back(x - 1);
            start = x;
            seenOne = x == 1;
        }
        seg.back()++;
        prv = x;
    }
    int M = seg.size();
    for (int i = M - 2; i >= 0; --i) {
        int v = (rot[i] - rot[i + 1]) % seg[i];
        if (v < 0) v += seg[i];
        v += seg[i];
        v %= seg[i];
        rot[i] = v + rot[i + 1];
    }
    int cnt = M + rot[0];
    seg.emplace_back(0); // dummy values
    rot.emplace_back(0);
    cout << cnt << endl;
    for (int i = 0; i < M; ++i) {
        cout << 1 << " " << seg[i] << endl; // place permutation
        int rotations = rot[i] - rot[i + 1];
        while (rotations--) {
            cout << 2 << endl;
        }
    }
}
```

## Plan Out

### Solution 1: Eulerian Circuit, undirected graph, connected components, Hierholzers algorithm

1. Connected component: a maximal set of vertices where every pair is connected by some path.
1. Eulerian component: a connected component whose every vertex has even degree. In an undirected graph, this condition is necessary and sufficient for the existence of an Eulerian circuit inside that component.
1. Use the handshaking lemma to know that with the odd degree vertex all be connected to some dummy node, means the degree of that dummy node will be even.
1. Even lengthed Eulerian circuits (even number of edges) will be balanced.
1. Odd lengthed Eulerian circuits will be imbalanced, where the starting node, will have imbalance of 2.

```cpp
int N, M;
vector<vector<pair<int, int>>> adj;
vector<int> day;
vector<bool> used;

// start node, eulerian circuits, all even degree nodes
void hierholzers(int source) {
    stack<pair<int, int>> stk;
    stk.emplace(source, -1);
    vector<int> edgeTour;
    while (!stk.empty()) {
        auto [u, eid] = stk.top();
        while (!adj[u].empty() && used[adj[u].back().second]) adj[u].pop_back();
        if (adj[u].empty()) {
            edgeTour.emplace_back(eid);
            stk.pop();
        } else {
            // take one neighbor and remove the edge from both sides
            auto [v, i] = adj[u].back();
            adj[u].pop_back();
            if (!used[i]) {
                used[i] = true;
                stk.emplace(v, i);
            }
        }
    }
    for (int i = 0; i < edgeTour.size(); ++i) {
        if (edgeTour[i] < 0) continue;
        day[edgeTour[i]] = i % 2 == 0 ? 1 : 2;
    }
}

void solve() {
    cin >> N >> M;
    adj.assign(N + 1, vector<pair<int, int>>());
    vector<pair<int, int>> edges;
    vector<int> deg(N + 1, 0);
    for (int i = 0; i < M; ++i) {
        int u, v;
        cin >> u >> v;
        adj[u].emplace_back(v, i);
        adj[v].emplace_back(u, i);
        deg[u]++, deg[v]++;
        edges.emplace_back(u, v);
    }
    // for all odd degree vertices point them to the dummy node 0, so that we have all connected components with eulerian circuits
    for (int i = 1; i <= N; ++i) {
        if (deg[i] % 2 == 0) continue;
        adj.emplace_back(vector<pair<int, int>>());
        adj[0].emplace_back(i, M);
        adj[i].emplace_back(0, M);
        M++;
    }
    vector<bool> vis(N + 1, false);
    day.assign(M, 0);
    used.assign(M, false);
    for (int i = 0; i <= N; ++i) {
        if (vis[i]) continue;
        hierholzers(i);
    }
    string assigned;
    vector<int64> d1(N + 1, 0), d2(N + 1, 0);
    for (int i = 0; i < edges.size(); ++i) {
        auto [u, v] = edges[i];
        if (day[i] == 1) {
            d1[u]++;
            d1[v]++;
            assigned += '1';
        } else {
            d2[u]++;
            d2[v]++;
            assigned += '2';
        }
    }
    int64 ans = 0;
    for (int i = 1; i <= N; ++i) {
        ans += d1[i] * d1[i] + d2[i] * d2[i];
    }
    cout << ans << " " << assigned << endl;
}
```

## Pay Off

### Solution 1:

```cpp

```

# Round 1

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

# Round 2

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

# Round 3

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
