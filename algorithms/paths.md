# Paths

• An Eulerian path is a path that goes through each edge exactly once.
• A Hamiltonian path is a path that visits each node exactly once.

The existence of Eulerian paths and circuits depends on the degrees of the nodes.
First, an undirected graph has an Eulerian path exactly when all the edges
belong to the same connected component and
• the degree of each node is even or
• the degree of exactly two nodes is odd, and the degree of all other nodes is
even.
In the first case, each Eulerian path is also an Eulerian circuit. In the second
case, the odd-degree nodes are the starting and ending nodes of an Eulerian path
which is not an Eulerian circuit.

## Eulerian Circuit in Directed Graphs

For an Eulerian circuit in each node belongs to the same strongly connected component 
and the indegree equals the outdegree for each node. 

# HAMILTONIANS

No efficient method is known for testing if a graph contains a Hamiltonian path,
and the problem is NP-hard. Still, in some special cases, we can be certain that a
graph contains a Hamiltonian path.
A simple observation is that if the graph is complete, i.e., there is an edge
between all pairs of nodes, it also contains a Hamiltonian path. Also stronger
results have been achieved:
• Dirac’s theorem: If the degree of each node is at least n/2, the graph
contains a Hamiltonian path.
• Ore’s theorem: If the sum of degrees of each non-adjacent pair of nodes is
at least n, the graph contains a Hamiltonian path.

A more efficient solution is based on dynamic programming (see Chapter
10.5). The idea is to calculate values of a function possible(S, x), where S is a
subset of nodes and x is one of the nodes. The function indicates whether there is
a Hamiltonian path that visits the nodes of S and ends at node x. It is possible to
implement this solution in O(n*2^n) time

This finds a hamiltonian path that happens to contain all the nodes, so it is a line graph.  But yeah it uses dp bitmask algorithm.
This is trying to find the minimum maximum distance between any two nodes in the graph.  or in other words minimize the maximum edge weight between any 
two nodes connected in the hamiltonian path.

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
```

# EULERIAN CIRCUITS

## EULERIAN CIRCUITS IN UNDIRECTED GRAPH USING HIERHOLZER'S ALGORITHM

adj list is a python list with set of neighbor nodes.  That way can remove them from adjency list to to not reprocess edges.

```py
def eulerian_circuit(adj_list, degrees):
    # start node is 1 in this instance
    n = len(degrees)
    start_node = 1
    stack = [start_node]
    vis = [0] * (n + 1)
    vis[start_node] = 1
    while stack:
        node = stack.pop()
        for nei in adj_list[node]:
            if vis[nei]: continue
            vis[nei] = 1
            stack.append(nei)
    for i in range(n):
        if (degrees[i] & 1) or (degrees[i] > 0 and not vis[i]): return False
    return True

def hierholzers_undirected(adj_list):
    start_node = 1
    stack = [start_node]
    circuit = []
    while stack:
        node = stack[-1]
        if len(adj_list[node]) == 0:
            circuit.append(stack.pop())
        else:
            nei = adj_list[node].pop()
            adj_list[nei].remove(node)
            stack.append(nei)
    return circuit
```

## Eulerian Path in Directed Graphs

In a directed graph, we focus on indegrees and outdegrees of the nodes. A
directed graph contains an Eulerian path exactly when all the edges belong to
the same connected component and
• in each node, the indegree equals the outdegree, or
• in one node, the indegree is one larger than the outdegree, in another node,
the outdegree is one larger than the indegree, and in all other nodes, the
indegree equals the outdegree.

### Function checking if Eulerian Path exists

Just to know this checks if there is an Eulerian path from the
specified start node to the end node. 

```py
def is_eulerian_path(n, adj_list, indegrees, outdegrees):
    # start node is 1 in this instance
    start_node = 1
    end_node = n
    stack = [start_node]
    vis = [0] * (n + 1)
    vis[start_node] = 1
    while stack:
        node = stack.pop()
        for nei in adj_list[node]:
            if vis[nei]: continue
            vis[nei] = 1
            stack.append(nei)
    if outdegrees[start_node] - indegrees[start_node] != 1 or indegrees[end_node] - outdegrees[end_node] != 1: return False
    for i in range(1, n + 1):
        if ((outdegrees[i] > 0 or indegrees[i] > 0) and not vis[i]): return False
        if (indegrees[i] != outdegrees[i] and i not in (start_node, end_node)): return False
    return True
```

### Finding the Eulerian Path in using Hierholzer's Algorithm

This assumes you know the start and end node.  A trick is you can find the start and end node based on the following logic.
if outdegree > indegree that will be the start node
if outdegree < indegree that will be the end node

```py
def hierholzers_directed(n, adj_list):
    start_node = 1
    end_node = n
    stack = [start_node]
    euler_path = []
    while stack:
        node = stack[-1]
        if len(adj_list[node]) == 0:
            euler_path.append(stack.pop())
        else:
            nei = adj_list[node].pop()
            stack.append(nei)
    return euler_path[::-1]
```

This is a specific implementation that uses unordered_map because the node values are all over the place.
It needs coordinate compression. 

```cpp
class Solution {
private:
    unordered_set<int> nodes;
    unordered_map<int, int> outdeg, indeg;
    unordered_map<int, vector<int>> adj;
    vector<vector<int>> eulerPath;
    void dfs(int u) {
        while (outdeg[u]) {
            outdeg[u]--;
            int v = adj[u][outdeg[u]];
            dfs(v);
            eulerPath.push_back({u, v});
        }
    }
public:
    vector<vector<int>> validArrangement(vector<vector<int>>& pairs) {
        for (const auto &edge : pairs) {
            int u = edge[0], v = edge[1];
            outdeg[u]++;
            indeg[v]++;
            adj[u].emplace_back(v);
            nodes.insert(u);
            nodes.insert(v);
        }
        int s = pairs[0][0];
        for (int u : nodes) {
            if (outdeg[u] - indeg[u] == 1) {
                s = u;
                break;
            }
        }
        dfs(s);
        reverse(eulerPath.begin(), eulerPath.end());
        return eulerPath;
    }
};
```