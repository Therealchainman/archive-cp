# Bipartite 

A bipartite graph is any graph which contains no odd length cycles and is two colorable. 

## Two coloring algorithm to determine if graph is bipartite

This one contains a remant of bridge_edge, might need to remove that and so on. 

```cpp
bool bipartite(int source) {
    stack<int> stk;
    stk.push(source);
    colors[source] = 1;
    bool ans = true;
    while (!stk.empty()) {
        int u = stk.top();
        stk.pop();
        for (int v : adj[u]) {
            if (colors[v] == 0) { // unvisited
                colors[v] = 3 - colors[u]; // needs to be different color of two possible values 1 and 2
                stk.push(v);
            } else if (colors[u] == colors[v]) {
                ans = false;
            }
        }
    }
    return ans;
}
```

## Maximum Bipartite Matching

Choose as many pairings as possible
without reusing any vertex.

Maximum bipartite matching means selecting the largest possible number of left-right edges such that no vertex is selected more than once.

### Kuhn's algorithm

O(V*E) time complexity, where V is the number of vertices and E is the number of edges in the bipartite graph. This can be really bade for a dense graph.

```cpp
vector<bool> visited;
vector<int> match;
vector<vector<int>> graph;
bool dfs(int u) {
    if (visited[u]) return false;
    visited[u] = true;
    for (int& v : graph[u]) {
        if (match[v] == -1 || dfs(match[v])) {
            match[v] = u;
            return true;
        }
    }
    return false;
}
int main() {
    int n, m, k, boy, girl;
    } directed graph from boy nodes to girl nodes
    for (int i = 0;i<k;i++) {
        cin>>boy>>girl;
        graph[boy].push_back(girl);
    }
    match.assign(m+1,-1);
    for (int u=1;u<=n;u++) {
        visited.assign(n+1, false);
        dfs(u);
    }
    int cnt = accumulate(match.begin(), match.end(), 0, [](const auto& a, const auto& b) {
        return a + (b!=-1);
    });
    cout<<cnt<<endl;
    for (int i = 1;i<=m;i++) {
        if (match[i]!=-1) {
            printf("%d %d\n", match[i], i);
        }
    }
}
```

### Hopcroft-Karp algorithms

The time complexity is O(E*sqrt(V)), which is much better than Kuhn's algorithm for dense graphs.

[Hopcroft-Karp Algorithm](hopcroft_karp_algorithm.md)

```cpp
int N, M;
vector<vector<int>> adj;
vector<int> matchL, matchR, dist;

bool bfs() {
    queue<int> q;
    for (int l = 0; l < N; l++) {
        if (matchL[l] == -1) {
            dist[l] = 0;
            q.emplace(l);
        } else {
            dist[l] = -1;
        }
    }
    bool foundAugmentingPath = false;
    while (!q.empty()) {
        int l = q.front();
        q.pop();
        for (int r : adj[l]) {
            int nextL = matchR[r];

            if (nextL == -1) {
                foundAugmentingPath = true;
            } else if (dist[nextL] == -1) {
                dist[nextL] = dist[l] + 1;
                q.emplace(nextL);
            }
        }
    }
    return foundAugmentingPath;
}

bool dfs(int l) {
    for (int r : adj[l]) {
        int nextL = matchR[r];
        if (nextL == -1 || (dist[nextL] == dist[l] + 1 && dfs(nextL))) {
            matchL[l] = r;
            matchR[r] = l;
            return true;
        }
    }
    dist[l] = -1;
    return false;
}

void solve() {
    cin >> N >> M;
    adj.assign(N, vector<int>());
    matchL.assign(N, -1);
    matchR.assign(N, -1);
    dist.assign(N, -1);
    for (int i = 0; i < M; ++i) {
        int u, v;
        cin >> u >> v;
        --u, --v;
        adj[u].emplace_back(v);
        adj[v].emplace_back(u);
    }
    int matching = 0;
    while (bfs()) {
        for (int l = 0; l < N; l++) {
            if (matchL[l] == -1 && dfs(l)) {
                matching++;
            }
        }
    }
    int maxIndependentSetSize = 2 * N - matching;
    int64 ans = 1013LL * maxIndependentSetSize;
    cout << ans << endl;
}
```

## Maximum Bipartite Matching on tree in linear time with DP on tree.

```cpp

```

### Solving the problem of Minimum Disjoint Vertex path cover in a DAG

A path cover is made by choosing some edges from the DAG so that the chosen edges form paths.

For that to be valid:

Each vertex can have at most 1 chosen outgoing edge.
Each vertex can have at most 1 chosen incoming edge.

left copy  = outgoing side
right copy = incoming side

answer = N - maximum matching result

