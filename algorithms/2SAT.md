# 2SAT Problem

Check notes with visualizations for examination of 2SAT Problem

General algorithm

1. Construct a directed implication graph
1. Variables can be numbered from 1 to N and their complements can be numbered from N + 1 to 2N. So i + N is i's complement. 
1. Tarjan's algorithm for finding strongly connected components (SCC)
1. This is all that is needed for the 2-SAT solver, Tarjan's generates the SCC in reverse topological order, and you can use that to determine if this problem is satisfiable.
1. You can also find one possible solution using this greedy algorithm wit the reverse topological order


CNF (conjunctive normal form)
conjunction of multiple clauses, where each clause is a disjunction of literals.

literal: boolean variable or its negation

disjunction: logical or operation between boolean expressions

2SAT, every clause has exactly two literals

## Tarjan's algorithm for SCC

```cpp
int N, M, timer, scc_count;
vector<vector<int>> adj;
vector<int> disc, low, comp;
stack<int> stk;
vector<bool> on_stack;

void dfs(int u) {
    disc[u] = low[u] = ++timer;
    stk.push(u);
    on_stack[u] = true;
    for (int v : adj[u]) {
        if (!disc[v]) dfs(v);
        if (on_stack[v]) low[u] = min(low[u], low[v]);
    }
    if (disc[u] == low[u]) { // found scc
        scc_count++;
        while (!stk.empty()) {
            int v = stk.top();
            stk.pop();
            on_stack[v] = false;
            low[v] = low[u];
            comp[v] = scc_count;
            if (v == u) break;
        }
    }
}
```

```cpp
adj.assign(2 * N, vector<int>());
// CONSTRUCT IMPLICATION GRAPH HERE
disc.assign(2 * N, 0);
low.assign(2 * N, 0);
comp.assign(2 * N, -1);
on_stack.assign(2 * N, false);
scc_count = -1;
timer = 0;
for (int i = 0; i < 2 * N; i++) {
    if (!disc[i]) dfs(i);
}
for (int i = 0; i < N; i++) {
    if (comp[i] == comp[i + N]) {
        cout << "IMPOSSIBLE" << endl;
        return 0;
    }
}
vector<int> ans(N, 0);
for (int i = 0; i < N; i++) {
    ans[i] = comp[i] < comp[i + N];
}
```