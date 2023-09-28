# HEAVY LIGHT DECOMPOSITION

This algorithm is excellent for finding min/max path queries on a tree and does it in log squared time complexity.  It also allows updates to node's values on the tree.  So this works for dynamic trees which will have queries and updates.

## implementation for max

Needs a rooted tree that is undirected.  Uses heavy light decomposition to compute the max. 

So decomposes the tree into heavy paths, which are disjoint by the light edges. The segment tree is used to query the heavy paths.  And you can don't need to find the lca if you store the head node of each heavy path and use the depth to determine which one to move upwards from. 

```py
import math

class SegmentTree:
    def __init__(self, n: int, neutral: int, func, initial_arr):
        self.func = func
        self.neutral = neutral
        self.size = 1
        self.n = n
        while self.size<n:
            self.size*=2
        self.nodes = [neutral for _ in range(self.size*2)] 
        self.build(initial_arr)

    def build(self, initial_arr: List[int]) -> None:
        for i, segment_idx in enumerate(range(self.n)):
            segment_idx += self.size - 1
            val = initial_arr[i]
            self.nodes[segment_idx] = val
            self.ascend(segment_idx)

    def ascend(self, segment_idx: int) -> None:
        while segment_idx > 0:
            segment_idx -= 1
            segment_idx >>= 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.nodes[segment_idx] = self.func(self.nodes[left_segment_idx], self.nodes[right_segment_idx])
        
    def update(self, segment_idx: int, val: int) -> None:
        segment_idx += self.size - 1
        self.nodes[segment_idx] = val
        self.ascend(segment_idx)
            
    def query(self, left: int, right: int) -> int:
        stack = [(0, self.size, 0)]
        result = self.neutral
        while stack:
            # BOUNDS FOR CURRENT INTERVAL and idx for tree
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # COMPLETE OVERLAP
            if segment_left_bound >= left and segment_right_bound <= right:
                result = self.func(result, self.nodes[segment_idx])
                continue
            # PARTIAL OVERLAP
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
        return result
    
    def __repr__(self) -> str:
        return f"nodes array: {self.nodes}, next array: {self.nodes}"

def main():
    n, q = map(int, input().split())
    values = list(map(int, input().split()))
    adj_list = [[] for _ in range(n)]
    for _ in range(n -1):
        u, v = map(int, input().split())
        u -= 1
        v -= 1
        adj_list[u].append(v)
        adj_list[v].append(u)
    parent, depth, head, size, index_map = [-1] * n, [0] * n, [-1] * n, [0] * n, [0] * n
    counter = 0
    heavy = list(range(n))
    def dfs(u):
        size[u] = 1
        heavy_size = 0
        for v in adj_list[u]:
            if v == parent[u]: continue
            parent[v] = u
            depth[v] = depth[u] + 1
            sz = dfs(v)
            size[u] += sz
            if sz > heavy_size:
                heavy_size = sz
                heavy[u] = v
        return size[u]
    dfs(0) # set node 0 as root
    def decompose(u, h):
        nonlocal counter
        index_map[u] = counter
        counter += 1
        head[u] = h
        for v in adj_list[u]:
            if v != heavy[u]: continue
            decompose(v, h)
        for v in adj_list[u]:
            if v == heavy[u] or v == parent[u]: continue
            decompose(v, v)
    decompose(0, 0)
    vals = [0] * n
    for i, v in enumerate(values):
        vals[index_map[i]] = v
    segment_tree = SegmentTree(n, -math.inf, max, vals) # need to compute the initial array, should contain heavy paths
    def query(u, v):
        res = 0
        while True:
            if depth[u] > depth[v]:
                u, v = v, u
            x, y = head[u], head[v]
            if x == y:
                left, right = index_map[u], index_map[v]
                res = max(res, segment_tree.query(left, right + 1))
                break
            elif depth[x] > depth[y]:
                left, right = index_map[x], index_map[u]
                res = max(res, segment_tree.query(left, right + 1))
                u = parent[x]
            else:
                left, right = index_map[y], index_map[v]
                res = max(res, segment_tree.query(left, right + 1))
                v = parent[y]
        return res
    result = []
    for _ in range(q):
        t, u, v = map(int, input().split())
        u -= 1
        if t == 1:
            segment_tree.update(index_map[u], v)
        else:
            v -= 1
            res = query(u, v)
            result.append(res)
    print(*result)
```

## cpp implementation that is fast

This segment tree is inclusive that is [L, R] for queries. 

don't forget to change the function to min or max depending on problem

also update the neutral value for each one

```cpp
vector<int> parent, depth, head, sz, index_map, heavy;
int counter, neutral;

int func(int x, int y) {
    return min(x, y);
}

struct SegmentTree {
    int size;
    vector<int> nodes;

    void init(int num_nodes) {
        size = 1;
        while (size < num_nodes) size *= 2;
        nodes.assign(size * 2, neutral);
    }

    void ascend(int segment_idx) {
        while (segment_idx > 0) {
            int left_segment_idx = 2 * segment_idx, right_segment_idx = 2 * segment_idx + 1;
            nodes[segment_idx] = func(nodes[left_segment_idx], nodes[right_segment_idx]);
            segment_idx >>= 1;
        }
    }

    void update(int segment_idx, int val) {
        segment_idx += size;
        nodes[segment_idx] = val;
        segment_idx >>= 1;
        ascend(segment_idx);
    }

    int query(int left, int right) {
        left += size, right += size;
        int res = neutral;
        while (left <= right) {
            if (left & 1) {
                res = func(res, nodes[left]);
                left++;
            }
            if (~right & 1) {
                res = func(res, nodes[right]);
                right--;
            }
            left >>= 1, right >>= 1;
        }
        return res;
    }
};

SegmentTree seg;

int heavy_dfs(int u) {
    sz[u] = 1;
    int heavy_size = 0;
    for (int v : adj_list[u]) {
        if (v == parent[u]) continue;
        parent[v] = u;
        depth[v] = depth[u] + 1;
        int s = heavy_dfs(v);
        sz[u] += s;
        if (s > heavy_size) {
            heavy_size = s;
            heavy[u] = v;
        }
    }
    return sz[u];
}

void decompose(int u, int h) {
    index_map[u] = counter++;
    seg.update(index_map[u], dist[u]);
    head[u] = h;
    for (int v : adj_list[u]) {
        if (v == heavy[u]) {
            decompose(v, h);
        }
    }
    for (int v : adj_list[u]) {
        if (v == heavy[u] || v == parent[u]) continue;
        decompose(v, v);
    }
}

int query(int u, int v) {
    int res = neutral;
    while (true) {
        if (depth[u] > depth[v]) {
            swap(u, v);
        }
        int x = head[u];
        int y = head[v];
        if (x == y) {
            int left = index_map[u];
            int right = index_map[v];
            res = func(res, seg.query(left, right));
            break;
        } else if (depth[x] > depth[y]) {
            int left = index_map[x];
            int right = index_map[u];
            res = func(res, seg.query(left, right));
            u = parent[x];
        } else {
            int left = index_map[y];
            int right = index_map[v];
            res = func(res, seg.query(left, right));
            v = parent[y];
        }
    }
    return res;
}

int32_t main() {
    int n = read(), q = read();
    for (int i = 0; i < n; i++) {
        values[i] = read();
    }
    for (int i = 0; i < n - 1; ++i) {
        int u = read(), v = read();
        u--;
        v--;
        adj_list[u].push_back(v);
        adj_list[v].push_back(u);
    }
    counter = 0;
    neutral = LLONG_MAX; // for min queries
    parent.assign(n, -1);
    depth.assign(n, 0);
    heavy.assign(n, -1);
    head.assign(n, 0);
    sz.assign(n, 0);
    index_map.assign(n, 0);
    for (int i = 0; i < n; ++i) heavy[i] = i;
    seg.init(n);
    heavy_dfs(0);
    decompose(0, 0);
    vector<int> result;
    for (int i = 0; i < q; ++i) {
        int t = read(), u = read(), v = read();
        u--;
        if (t == 1) {
            // Update operation
            // Call the update function for the segment tree
            seg.update(index_map[u], v);
        } else {
            v -= 1;
            // Query operation
            int res = query(u, v);
            result.push_back(res);
        }
    }
    for (int res : result) {
        cout << res << " ";
    }
    cout << endl;
    return 0;
}
```