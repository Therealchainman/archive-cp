


# P2: Festival

### Solution: 2 segment trees

```c++

```

### Solution: 2 multisets (binary search trees)

```c++

```

# P3: Star Trappers

### Solution: geometry


```c++

```


# P4: Graph Travel

### Solution: DFS + backtracking that passes test case #1 only
This solution works for passing the first test case only.  Need to optimize or another algorithm for the second test case. 

```c++
bool vis[15];
int activate[15];
int cnt;
int dfs(int u, vector<vector<int>>& graph, vector<vector<int>>& rooms, int& magic, int K) {
    int n = rooms.size(); // number of rooms
    if (magic>K) {return 0;}
    if (magic==K) {
        return 1;
    }
    for (int v : graph[u]) {
        if (vis[v]) continue;
        activate[v]++;
    }
    int cnt = 0;
    for (int v = 0;v<n;v++) {
        if (activate[v]==0 || vis[v] || magic<rooms[v][0] || magic>rooms[v][1]) continue;
        vis[v] = true;
        activate[v]--;
        magic+=rooms[v][2];
        cnt+=dfs(v, graph, rooms, magic, K);
        magic-=rooms[v][2];
        vis[v] = false;
        activate[v]++;
    }
    for (int v : graph[u]) {
        if (vis[v]) continue;
        activate[v]--;
    }
    return cnt;
}
int main() {
    int T, N, M, K, l, r, a,x,y;
    freopen("input.txt", "r", stdin);
    cin >> T;
    for (int t = 1;t<=T;t++) {
        cin>>N>>M>>K;
        vector<vector<int>> rooms, graph(N);
        for (int i = 0;i<N;i++) {
            cin>>l>>r>>a;
            rooms.push_back({l, r, a});
        }
        while (M--) {
            cin>>x>>y;
            graph[x].push_back(y);
            graph[y].push_back(x);
        }

        cnt = 0;
        for (int i = 0;i<N;i++) {
            memset(vis, false, sizeof(vis));
            memset(activate, false, sizeof(activate));
            int magicPoints = rooms[i][2];
            vis[i] = true;
            cnt+=dfs(i, graph, rooms, magicPoints, K);
        }
        printf("Case #%d: %d\n", t, cnt);
    }
}
```

### Solution: DFS + backtracking + dynamic programming

```c++

```