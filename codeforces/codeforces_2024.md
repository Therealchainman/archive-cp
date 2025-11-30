# Problems solved in 2024

## The Two Routes

### Solution 1:  bfs, adjacency matrix, unweighted undirected graph

This is a fun little problem, it actually just requires you to use simple bfs. Either the bus or train will have an edge from 1 to N, so just let that one travel that edge, and have the other take shortest path on unweighted graph from 1 to N. 


```cpp
const int MAXN = 405;
int N, M;
int adj[MAXN][MAXN];
bool vis[MAXN];

void solve() {
    cin >> N >> M;
    memset(adj, 0, sizeof(adj));
    memset(vis, 0, sizeof(vis));
    for (int i = 0; i < M; i++) {
        int u, v;
        cin >> u >> v;
        u--; v--;
        adj[u][v] = adj[v][u] = 1;
    }
    // swap everything 
    if (adj[0][N - 1]) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                adj[i][j] ^= 1;
            }
        }
    }
    int ans = 0;
    queue<int> q;
    q.push(0);
    while (!q.empty()) {
        int sz = q.size();
        while (sz--) {
            int u = q.front();
            q.pop();
            if (u == N - 1) {
                cout << ans << endl;
                return;
            }
            for (int v = 0; v < N; v++) {
                if (!adj[u][v] || vis[v]) continue;
                vis[v] = true;
                q.push(v);
            }
        }
        ans++;
    }
    cout << -1 << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```