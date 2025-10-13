# EULERIAN

## Eulerian Paths

### Hierholzer's Algorithm

This is example, essentially it just consists of visiting all nodes and once all nodes visited it gets added to the stack.  In this specific example it required sorting of edges as well, but that is not necessary for Hierholzer's algorithm.  It's just the dfs that is essential and the stack.

```py
tickets.sort(key = lambda x: x[-1], reverse = True)
adj_list = defaultdict(list)
for u, v in tickets:
    adj_list[u].append(v)
stack = []
def dfs(u):
    while adj_list[u]:
        dfs(adj_list[u].pop())
    stack.append(u)
dfs("JFK")
return stack[::-1]
```

# EULERIAN CIRCUITS

## EULERIAN CIRCUITS IN UNDIRECTED GRAPH USING HIERHOLZER'S ALGORITHM

Traverses every edge edge of a graph exactly once.
if all nodes have even degree it has an eulerian circuit.
if you have two nodes with odd degree it has an eulerian path, so it will start and end at those nodes.

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

### Hierholzers from start node and finding the edges in the circuit

1. Adjacency stores both neighbor and edge id. This lets you mark edges used once globally with used[edge_id].
1. Lazy pruning of used edges. You pop from the back of adj[u] while the last edge is already used, which keeps each edge touched O(1) times.
1. Edge-centric tour. You record the exact edge ids in edgeTour, then label edges directly by id. This prevents the classic bug of assigning by tour inex.

1. Simple alternating labeling. After the tour, you set day[edge_id] to 1,2,1,2,... along the Eulerian order, which is a clean post-processing pass talored to your scoring.

1. Sentinel aware. You push -1 as the initial eid and skip it during labeling.
1. Linear time and memory. Overall O(E) time and O(E) memory once degrees are even per component.

```cpp
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
