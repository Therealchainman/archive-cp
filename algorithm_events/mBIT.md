# Montgomery Blair Informatics Tournament

# Montgomery Blair Informatics Tournament 2025

## Azhdaha's Adventure

### Solution 1: 

The algorithm solves a tree problem involving two special nodes, **U** and **V**, and tries to compute the number of nodes that are decisively closer to either U or V when you consider not just the path between them, but also the branches hanging off that path.

---

## Key Insight

- In a tree, there is **exactly one simple path** between any two nodes.
- Imagine walking along the path from U to V — this forms a **backbone path**.
- The rest of the tree forms **branches hanging off** the nodes on this path.
- The challenge: determine which branches are **controlled by U**, **controlled by V**, or **contested** (neither dominates).

---

## Core Algorithmic Ideas

### 1. **Decompose the Tree into Path + Branches**
- **Find the unique path** from U to V.
- For each node on the path, compute:
  - The **size** of its off-path subtree.
  - The **depth** of the deepest leaf in that subtree.
- This gives you a line (U–V path) with vertical “branch” trees.

### 2. **Balance the Competition Between U and V**
- Think of U and V as **racing to reach the deepest leaf** in each branch.
- For a node `i` on the path:
  - U reaches it in `i` steps; then down `height[i]`.
  - V reaches it in `M−1−i` steps (from the other end); then down `height[i]`.
- This leads to the idea of a **"contested zone"** in the middle of the path.

### 3. **Use Two-Pointer Sliding Window to Find the Contested Interval**
- Start with the full path `[0..M−1]`.
- Shrink in from both ends using two pointers (`l` and `r`) based on U/V's reachability.
- Stop when neither side clearly dominates any node in `[l..r]`.
- Nodes **outside** `[l..r]` are clearly owned by U (left) or V (right).

### 4. **Sum the Sizes of Clearly Owned Branches**
- For all path nodes `i` **outside** the contested interval:
  - Add `sz[path[i]]` to the answer (size of that branch).
- This gives you the total number of "safe" off-path nodes.

---

## Problem-Solving Patterns Used

### Tree Decomposition
Break a tree into a central path and separate subtrees.

### Contested Zone via Sliding Window
Use a two-pointer method to simulate balance/competition between two endpoints.

### Depth + Reach Comparison
Model reachability using `distance + depth` to determine who gets somewhere first.

### Selective DFS
Run DFS on only the off-path parts by excluding known nodes (on the main path).


```cpp
int N, U, V;
vector<int> parent, height, sz;
vector<vector<int>> adj;
vector<bool> inPath;

void dfs(int u, int p = -1) {
    parent[u] = p;
    for (int v : adj[u]) {
        if (v == p) continue;
        dfs(v, u);
    }
}

void dfs1(int u, int p = -1) {
    sz[u] = 1;
    height[u] = 0;
    for (int v : adj[u]) {
        if (v == p) continue;
        if (inPath[v]) continue;
        dfs1(v, u);
        sz[u] += sz[v];
        height[u] = max(height[u], height[v] + 1);
    }
}

pair<int, int> calc(const vector<int>& heights) {
    int M = heights.size();
    int l = 0, r = M - 1, nl = heights[r], nr = M - 1 - heights[l];
    while (l < r && (l < nl || r > nr)) {
        if (l < nl) {
            l++;
            nr = min(nr, M - 1 - heights[l] + l);
        }
        if (r > nr) {
            r--;
            nl = max(nl, heights[r] - (M - 1 - r));
        }
    }
    return {l, r};
}

void solve() {
    cin >> N >> U >> V;
    U--, V--;
    adj.assign(N, vector<int>());
    parent.assign(N, -1);
    height.assign(N, 0);
    sz.assign(N, 0);
    for (int i = 1; i < N; i++) {
        int u, v;
        cin >> u >> v;
        u--, v--;
        adj[u].emplace_back(v);
        adj[v].emplace_back(u);
    }
    dfs(U);
    vector<int> path;
    inPath.assign(N, false);
    while (V != -1) {
        inPath[V] = true;
        path.emplace_back(V);
        V = parent[V];
    }
    reverse(path.begin(), path.end());
    int M = path.size();
    vector<int> heights;
    for (int x : path) { // treat each node in path as root, get a forest
        dfs1(x);
        heights.emplace_back(height[x]);
    }
    pair<int, int> interval = calc(heights);
    int ans = 0;
    for (int i = 0; i < M; i++) {
        if (i <= interval.first || i >= interval.second) ans += sz[path[i]];
    }
    cout << ans << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    int T;
    cin >> T;
    while (T--) {
        solve();
    }
    return 0;
}
```

## Xochilipilli's Dance

### Solution 1: 

Is it ever part of optimal strategy to hold a position for x seconds, and then begin the process to switch to another position? 

Or is it always just the case you want to hold a position, and begin the switching after that dance move.

there are a few intervals which you will skip, if they sum to less than K, because you get no points for them, if you chose to get the points for the position before this segment of intervals. 

I don't think you will ever change positions in the middle of an interval, when you are at the correct position for that interval and gaining those points. 

making an assumption that you hold every position that you can. 

two decisions you have.  

hold the position
change position


dynamic programming where you consider the current position, and consider the k time to move into that position would work.
Just need a little correction delta for when moving from k back takes from just a part of a prior interval.  
And also need to consider the decision to hold position.  But basically I just can break into sub problem because
I just want to maximize the time matched up to the ith interval. 

But the problem might be on the correction delta, cause what if for some ith interval the best option at the time seems to 
be to not take it.  So it is not counted in the dp score. 

I think I just need to keep track of that.  

```cpp

```

## Nian's Fear

### Solution 1: 

You launch the banners first, then figure out how to move them so you can launch the most firecrackers. 

Oh it is implied the banners are just there, you can only slide them around to help with launching firecrackers. 

So yeah when you fire the tallest firecrackers, you now free up lower row numbers and potentially more banners can slide more now, and allow you to fire other firecrackers.

I think you always move the banners to either extreme, that is all the way to the left or all the way to the right. 

So then it is recursive, you keep going into these intervals.  

There is only small part that is available on the left or right, and if you can remove firecracker, that is awesome, if you can't you need to keep that firecracker, and it will act as a wall for banners later. 

To me the left and right end intervals just continue to decrease over time, basically.

Cause then you have a banner that you might not be free to move very far, yo uneed to be able to query to the find the nearest firecracker that is greater than or equal in height. 

And then you can only move it that far. 

you want to query to the left and the right of a banner, 

if one to the right, shift it all the way over
if one to the left shift all the way over. 

```cpp

```
