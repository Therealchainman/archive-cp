# Functional Graph

Graph where each vertex has outdegree = 1
A functional graph is a directed graph

## Detection of cycle

The following implementation can detect cycles in functional graphs.  And also backtrack and recover the list of nodes in the order they were visited in the cycle.  So you could compute anything related to the cycle.  You know the entire cycle.  And you can backtrack through the rest of the nodes as well for each weakly connected component. 

```py
class Solution:
    def countVisitedNodes(self, edges: List[int]) -> List[int]:
        n = len(edges)
        ans, vis = [0] * n, [0] * n
        def search(u):
            parent = {u: None}
            is_cycle = False
            while True:
                vis[u] = 1
                v = edges[u]
                if v in parent: 
                    is_cycle = True
                    break
                if vis[v]: break
                parent[v] = u
                u = v
            if is_cycle:
                crit_point = parent[edges[u]]
                cycle_path = []
                while u != crit_point:
                    cycle_path.append(u)
                    u = parent[u]
                len_ = len(cycle_path)
                for val in cycle_path:
                    ans[val] = len_
            while u is not None:
                ans[u] = ans[edges[u]] + 1
                u = parent[u]
        for i in range(n):
            if vis[i]: continue
            search(i)
        return ans
```

This is a more concrete example, where it is purely computing the cycles and marking all the nodes that are part of a cycle.

```py
    def search(u):
        parent = {u: None}
        is_cycle = False
        while True:
            vis[u] = 1
            v = edges[u]
            if v in parent: 
                is_cycle = True
                break
            if vis[v]: break
            parent[v] = u
            u = v
        if is_cycle:
            crit_point = parent[edges[u]]
            cnt = 0
            while u != crit_point:
                cycle[u] = 1
                cnt += 1
                u = parent[u]
            return cnt
        return 0
```

## Condensation graph

Find the cycles, which are strongly connected components and condense them into a single node, by marking the nodes in cycles.

```cpp
int N, cnt;
vector<int> out, inCycle;
vector<vector<int>> cycleNodes;
vector<bool> vis;

void search(int u) {
    map<int, int> par;
    par[u] = -1;
    bool isCycle = false;
    while (!vis[u]) {
        vis[u] = true;
        int v = out[u];
        if (par.count(v)) {
            isCycle = true;
            break;
        }
        if (vis[v]) break;
        par[v] = u;
        u = v;
    }
    if (isCycle) {
        int critPoint = par[out[u]];
        vector<int> cycle;
        while (u != critPoint) {
            cycle.emplace_back(u);
            u = par[u];
        }
        cnt++;
        cycleNodes.emplace_back(cycle);
    }
}
```

This implementation is good for finding cycles, and taking all nodes in a cycle and assigning them all to a single representative node, and updating the indegrees for that representative node.  Basically this is creating a condensation graph and condensing the strongly connected component.

```cpp

void search(int u) {
    map<int, int> par;
    par[u] = -1;
    bool isCycle = false;
    while (!vis[u]) {
        vis[u] = true;
        int v = child[u];
        if (par.count(v)) {
            isCycle = true;
            break;
        }
        if (vis[v]) break;
        par[v] = u;
        u = v;
    }
    if (isCycle) {
        int critPoint = par[child[u]];
        int cycleNode = u;
        int indegree = 0;
        while (u != critPoint) {
            node[u] = cycleNode;
            indegree += indegrees[u] - 1;
            indegrees[u] = 0;
            u = par[u];
        }
        indegrees[cycleNode] = indegree;
        inCycle[cycleNode] = true;
    }
}
```