


```cpp
int N;

struct Rectangle {
    int r1, c1, r2, c2;
    char c;
};

struct Piece {
    int R, C, u, d;
    char c;
};

vector<Rectangle> puzzle;
vector<Piece> pieces;

void solve() {
    cin >> N;
    for (int i = 0; i < N; i++) {
        Piece p;
        cin >> p.c >> p.R >> p.C >> p.u >> p.d;
        pieces.push_back(p);
    }
    int idx, u, d;
    cin >> idx;
    idx--;
    Rectangle rec;
    rec.r1 = 0; rec.c1 = 0; rec.r2 = pieces[idx].R; rec.c2 = pieces[idx].C; 
    u = pieces[idx].u; d = pieces[idx].d;
    rec.c = pieces[idx].c;
    puzzle.push_back(rec);
    int R = pieces[idx].R, C = pieces[idx].C;
    for (int i = 0; i < N - 1; i++) {
        int idx;
        cin >> idx;
        idx--;
        Rectangle rec;
        if (u == 0) {
            rec.c1 = puzzle[i].c1 + d - 1; 
            rec.c2 = rec.c1 + pieces[idx].C;
            rec.r1 = puzzle[i].r2;
            rec.r2 = rec.r1 + pieces[idx].R;
        } else {
            rec.r1 = puzzle[i].r2 - d;
            rec.r2 = rec.r1 + pieces[idx].R;
            rec.c1 = puzzle[i].c2;
            rec.c2 = rec.c1 + pieces[idx].C;
        }
        rec.c = pieces[idx].c;
        R = max(R, rec.r2);
        C = max(C, rec.c2);
        u = pieces[idx].u; d = pieces[idx].d;
        puzzle.push_back(rec);
    }
    vector<vector<char>> grid(R, vector<char>(C, '.'));
    for (Rectangle rec : puzzle) {
        for (int r = rec.r1; r < rec.r2; r++) {
            for (int c = rec.c1; c < rec.c2; c++) {
                grid[R - r - 1][c] = rec.c;
            }
        }
    }
    cout << R << " " << C << endl;
    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C; c++) {
            cout << grid[r][c];
        }
        cout << endl;
    }
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    while (T--) {
        solve();
    }
    return 0;
}
```

```cpp
const int LOG = 16, MAXN = 33'000;
int N;
vector<vector<int>> adj;
int blocked[MAXN], A[MAXN], B[MAXN];
vector<pair<int, int>> ans;

void bfs(int src) {
    queue<int> q;
    q.push(src);
    map<int, int> backtrack;
    unordered_set<int> vis;
    vis.insert(src);
    int u = src;
    while (!q.empty()) {
        u = q.front();
        q.pop();
        if (B[u]) break;
        for (auto v : adj[u]) {
            if (vis.count(v)) continue;
            if (blocked[v] || A[v]) continue;
            backtrack[v] = u;
            q.push(v);
            vis.insert(v);
        }
    }
    blocked[u] = 1;
    vector<pair<int, int>> path;
    while (u != src) {
        path.push_back({backtrack[u], u});
        u = backtrack[u];
    }
    ans.insert(ans.end(), path.rbegin(), path.rend());
}

void solve() {
    cin >> N;
    memset(A, 0, sizeof(A));
    memset(B, 0, sizeof(B));
    memset(blocked, 0, sizeof(blocked));
    for (int i = 0; i < N; i++) {
        int x;
        cin >> x;
        A[x] = 1;
    }
    for (int i = 0; i < N; i++) {
        int x;
        cin >> x;
        B[x] = 1;
    }
    for (int i = 0; i < MAXN; i++) {
        if (A[i] && B[i]) {
            blocked[i] = 1;
            A[i] = 0;
        }
    }
    for (int i = 0; i < MAXN; i++) { // 33000
        if (A[i]) {
            bfs(i); // 20
        }
    }
    cout << ans.size() << endl;
    for (auto &[x, y] : ans) {
        cout << x << " " << y << endl;
    }
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    adj.assign(MAXN, vector<int>());
    for (int i = 0; i < MAXN; i++) {
        for (int j = 0; j < LOG; j++) {
            int v = i ^ (1 << j);
            if (v < MAXN) {
                adj[i].push_back(v);
            }
        }
    }
    while (T--) {
        solve();
    }
    return 0;
}
```

```cpp

```