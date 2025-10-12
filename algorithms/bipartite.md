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