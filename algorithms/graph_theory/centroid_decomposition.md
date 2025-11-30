# Centroid Decomposition

Centroid decomposition is a recursive technique that splits a tree into smaller pieces by repeatedly removing “centroid” nodes—nodes whose removal leaves every connected component of size ≤½ the original.  The result is a new “centroid-tree” which has height O(log n).

---

## Data Structures

- `tree`: adjacency list of the original tree, size n+1  
- `cd`: adjacency list of the centroid-tree, size n+1  
- `sub[v]`: size of the (active) subtree rooted at v  
- `del[v]`: boolean flag marking whether v has been “deleted” in the decomposition 

Important to note that this you need the nodes to be 1-indexed, cause it makes a dummy node labeled as 0 to be the root of the centroid tree.

### calcSums
Compute the size of each (remaining) subtree rooted at node, storing it in sub[node]. This size count excludes any nodes already “removed” (marked in del[]).

### findCentroid

Given a connected component of (un‐deleted) nodes of total size sz, find its centroid: a node which, if chosen as root and “removed,” leaves no child‐subtree of size ≥ sz/2.

### centroidDecomposition

Recursively assemble the centroid‐decomposition tree. Each time you pick a centroid of some remaining component, you:
1. Record it as a child of its parent‐centroid in the cd adjacency list.
1. Mark it deleted in del[], so subsequent centroids ignore it.
1. For each of the resulting smaller components (each neighbor subtree under that centroid), recompute sizes, find that component’s centroid, and recurse.

```cpp
vector<vector<int>> adj, cd;
vector<int> sub;
vector<bool> vis;
int cdRoot = -1;

void calcSums(int u, int p = -1) {
    sub[u] = 0;
    if (vis[u]) return;
    for (int v : adj[u]) {
        if (v == p) continue;
        calcSums(v, u);
        sub[u] += sub[v];
    }
    sub[u]++;
}
 
int findCentroid(int u, int sz, int p = -1) {
    for (int v : adj[u]) {
        if (v == p || vis[v]) continue;
        if (sub[v] * 2 >= sz) return findCentroid(v, sz, u);
    }
    return u;
}
 
void centroidDecomposition(int u, int p) {
    cd[p].emplace_back(u);
    vis[u] = true;
    for (int v : adj[u]) {
        if (vis[v]) continue;
        calcSums(v, u);
        int cent = findCentroid(v, sub[v], u);
        centroidDecomposition(cent, u);
    }
}

calcSums(1);
cdRoot = findCentroid(1, n);
centroidDecomposition(cdRoot, 0);
```