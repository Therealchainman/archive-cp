# Graph Algorithms



## School Dance

### Solution: Maximum Bipartite Matching with Kuhn's algorithm


```c++
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
    cin>>n>>m>>k;
    graph.resize(n+1); // directed graph from boy nodes to girl nodes
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