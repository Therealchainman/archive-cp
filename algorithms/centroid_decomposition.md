# Centroid Decomposition

Centroid decomposition is a recursive technique that splits a tree into smaller pieces by repeatedly removing “centroid” nodes—nodes whose removal leaves every connected component of size ≤½ the original.  The result is a new “centroid-tree” which has height O(log n).

---

## Data Structures

- `tree`: adjacency list of the original tree, size n+1  
- `cd`: adjacency list of the centroid-tree, size n+1  
- `sub[v]`: size of the (active) subtree rooted at v  
- `del[v]`: boolean flag marking whether v has been “deleted” in the decomposition 

```cpp
vector<vector<int>> tree, cd;
vector<int> sub, val;
vector<bool> del;
int n, cdRoot = -1;

void calc_sums(int node, int par = 0) {
    sub[node] = 0;
    if (del[node]) return;
    for (auto next: tree[node]) {
        if (next == par) continue;
        calc_sums(next, node);
        sub[node] += sub[next];
    }
    sub[node]++;
}
 
int find_centroid(int node, int sz, int par = 0) {
    for (auto next: tree[node]) {
        if (next == par || del[next]) continue;
        if (sub[next] * 2 >= sz) return find_centroid(next, sz, node);
    }
    return node;
}
 
void build(int node, int prev) {
    cd[prev].push_back(node);
    del[node] = true;
    for (auto next: tree[node]) {
        if (del[next]) continue;
        calc_sums(next, node);
        int cent = find_centroid(next, sub[next], node);
        build(cent, node);
    }
}

 calc_sums(1);
cdRoot = find_centroid(1, n);
build(cdRoot, 0);
```