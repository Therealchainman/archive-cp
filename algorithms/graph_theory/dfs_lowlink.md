# DFS Lowlink or Tarjan's algorithm

What are some example problems to practice:
Forbidden Cities (queries on removing a “forbidden” vertex, classic 2-vertex-connected / block-cut reasoning).
Critical Cities (articulation-point flavored).
Strongly Connected Edges (bridge check + orient edges, very “Tarjan bridge” adjacent).
Disruption (bridge-tree style thinking in a tree with extra edges; often taught alongside 2ECC ideas).
Push a Box (grid graph + biconnected components / articulation logic).
ABC075 C – Bridge (intro: find all bridges; lowlink 101).
ABC120 D – Decayed Bridges (bridge concept, solved via offline DSU; good complement to lowlink).
ARC143 D – Bridges (harder; explicitly about minimizing bridges).
118E – Bertown Roads (orient edges iff no bridges; classic lowlink bridge check).
1000E – We Need More Bosses (bridge-tree compression + diameter).
732F – Tourist Reform (build an ordering / rooting using bridges + DFS structure).
652E – Pursuit For Artifacts (uses 2-edge-connected components / bridges reasoning).
962F – (contest 962) F “edges that belong to exactly one simple cycle” (lowlink / bridge-ish + cycle structure).
173D – Deputies (constraints about “cities connected by a bridge”; bridge decomposition).

## Undirected graphs

Consider the concepts of "DFS tree edges" and "back edges".

### low links and pre

pre[u] is just the preorder traversal order for each node in the DFS.
low[u] captures how far "up" the DFS tree you can reach from the subtree of v without using the parent edge upward.

### Cut vertex or articulation points

Cut vertex are vertices, where if they are removed it increases the number of connected components.  

They can be detected using this algorithm. Let's explore the idea and why they work. 

Say you are at node u in the DFS traversal. 

You will explore some children nodes, say one of them is called v. 
If v is not a back edge then continue. 
Perform the dfs(v, u) 
and after it returns you check the following is the low[v] >= pre[u].  If this is the case that means that node u is a cut vertex, because what this means is that there is no alternative route around node u from within v's subtree, so if you were to remove node u, that means v's subtree would become disconnected from other part of the graph.  Or in another way it means that v's subtree must go through vertex u to get above it. 

### 2-vertex connected components

Just need to add a stack to keep track of the edges, and whenever you detect a cut vertex, you pop from the stack and build up the edge set that makes up the 2-vertex connected component. 

You will at the end construct multiple disjoint sets of all the edges of the graph, and these represent the 2-vertex connected components.

### Cut edge or bridge

Cut edge are edges that when they are removed it increases the number of connected components by exactly `one`.

These are detected slightly different, now you require that low[v] > pre[u], 
Intuition: v’s subtree cannot reach u or above via a back-edge, so that edge is the only connection upward.

### 2-edge connected components

Run DFS, compute pre, low, and mark all bridges using low[v] > pre[u].

Build components by doing a DFS/BFS on the original graph but never crossing a bridge. Each traversal gives one 2-edge connected component. 

## Directed graphs

lowlink never passes the root

low[u] = pre[u] means a root of a strongly connected component.

Use a stack to find all nodes in each SCC.

```cpp
int N, M, numScc, cnt;
vector<vector<int>> adj;
vector<int> pre, comp, low;
stack<int> stk;

void dfs(int u) {
    if (pre[u] != -1) return;
    pre[u] = cnt;
    low[u] = cnt++;
    stk.emplace(u);
    for (int v : adj[u]) {
        dfs(v);
        low[u] = min(low[u], low[v]);
    }
    if (pre[u] == low[u]) {
        numScc++;
        while (true) {
            int v = stk.top();
            stk.pop();
            comp[v] = numScc;
            low[v] = N;
            if (u == v) break;
        }
    }
}
```

And it can be used to construct a DAG, where you condense all the strongly connected components into a single node in this graph. 
Then you can perform dp on it by the topological sorting. 

```cpp
    vector<int64> coins(numScc, 0), dp(numScc, 0);
    for (int i = 0; i < N; ++i) {
        coins[comp[i]] += A[i];
    }
    vector<vector<int>> dag(numScc, vector<int>());
    for (int u = 0; u < N; ++u) {
        int cu = comp[u];
        for (int v : adj[u]) {
            int cv = comp[v];
            if (cu == cv) continue;
            dag[cu].emplace_back(cv);
        }
    }
    vector<int> ind(numScc, 0);
    for (int i = 0; i < numScc; ++i) {
        sort(dag[i].begin(), dag[i].end());
        dag[i].erase(unique(dag[i].begin(), dag[i].end()), dag[i].end());
        for (int v : dag[i]) {
            ind[v]++;
        }
    }
    stack<int> st;
    for (int i = 0; i < numScc; ++i) {
        if (ind[i] == 0) {
            dp[i] = coins[i];
            st.emplace(i);
        }
    }
    while (!st.empty()) {
        int u = st.top();
        st.pop();
        for (int v : dag[u]) {
            dp[v] = max(dp[v], dp[u] + coins[v]);
            if (--ind[v] == 0) {
                st.emplace(v);
            }
        }
    }
```

1684 – Giant Pizza

1685 – New Flight Routes 

USACO (directed SCC / condensation)

USACO Training – “schlnet” (School Network)

USACO Jan 2015 Gold – Grass Cownoisseur 

USACO Feb 2022 Silver – Redistributing Gifts

Typical90: 021 – Come Back in One Piece (counts mutually reachable pairs, SCC sizes)

ABC245 F – Endless Walk

ABC296 E – Transition Game 

ABC357 E – Reachability in Functional Graph 
