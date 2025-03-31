# Functional Graph

Graph where each vertex has outdegree = 1
A functional graph is a directed graph

## Condensation graph

Find the cycles, which are strongly connected components and condense them into a single node, by marking the nodes in cycles.

I think there is something wrong with this algorithm it won't detect connected components that is one cycle, or loop. 


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

I think there is something wrong with this algorithm it won't detect connected components that is one cycle, or loop. 

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