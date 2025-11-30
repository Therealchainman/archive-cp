# Global Minimum Cut

Algorithms that solve the global minimum cut problem

## Stoer Wagner in python

This is Stower Wagner algorithm in python.  It probably has O(v^3) time and could be improved upon.  It is tested to work on a test set.  In current form it returns the size of the partition and can use that to get the size of both partitions.  Can also change it to return the nodes if that is needed instead. 

nodes should be a set of the nodes
adj should be a dict of set of the neighbors with edge weights

stoer wagner requires that the edge weights be non-negative.
Works for undirected graphs with edge weights.

If you want to use it for unweighted graphs, just set all the edge weights to 1.

```py
def stoer_wagner(nodes, adj, edges):
    n = len(nodes)
    min_cut = math.inf
    connected_components = {node: [node] for node in nodes}
    partition_size = None
    for i in range(n - 1):
        # not necessarily random, and doesn't have to be
        u = nodes.pop()
        nodes.add(u)
        seen = set()
        weights = Counter()
        max_heap = []
        while len(seen) < n - i - 1:
            while max_heap and weights[max_heap[0][1]] != abs(max_heap[0][0]):
                heapq.heappop(max_heap)
            if max_heap:
                u = heapq.heappop(max_heap)[1]
                weights[u] = 0
            seen.add(u)
            for v in adj[u]:
                if v in seen: continue
                w = edges[(u, v)]
                weights[v] += w
                heapq.heappush(max_heap, (-weights[v], v))
        while max_heap and weights[max_heap[0][1]] != abs(max_heap[0][0]):
            heapq.heappop(max_heap)
        assert len(max_heap) > 0, "max heap is empty"
        wei, v = heapq.heappop(max_heap)
        wei = abs(wei)
        if wei < min_cut:
            min_cut = wei
            partition_size = len(connected_components[v])
        # contract v into u, merge into one connected component
        connected_components[u].extend(connected_components[v])
        # loop through children of v as w
        neighbors = adj[v]
        for w in neighbors:
            adj[w].remove(v)
            if w == u: continue
            if (u, w) not in edges:
                adj[u].add(w)
                adj[w].add(u)
            edges[(u, w)] += edges[(v, w)]
            edges[(w, u)] += edges[(w, v)]
        adj[v].clear()
        nodes.remove(v)
    return min_cut, partition_size
```

## Stoer Wagner in C++

This is an implementation of stoer wagner in C++, it is unfortunately untested because the online judge was down.  Could still have a few bugs.  

For one it doesn't check if the graph is already disconnected, which might be necessary depending on the data input

```cpp
int N, M;

int stoerWagner(vector<vector<int>>& adj, map<pair<int, int>, int>& edges) {
    int min_cut = INT_MAX;
    vector<int> vis(N, 0);
    for (int i = 0; i < N - 1; i++) {
        set<int> seen;
        priority_queue<pair<int, int>, vector<pair<int, int>>> maxHeap;
        map<int, int> weights;
        int u = 0;
        while (vis[u]) u++;
        int w, _;
        while (seen.size() < N - i - 1) {
            while (!maxHeap.empty() && weights[maxHeap.top().second] != maxHeap.top().first) maxHeap.pop();
            if (!maxHeap.empty()) {
                tie(_, u) = maxHeap.top();
                maxHeap.pop();
            }
            seen.insert(u);
            for (int v : adj[u]) {
                if (seen.find(v) != seen.end() || vis[v]) continue;
                if (weights.find(v) != weights.end()) {
                    weights[v] += edges[{u, v}];
                } else {
                    weights[v] = edges[{u, v}];
                }
                maxHeap.emplace(weights[v], v);
            }
        }
        while (!maxHeap.empty() && weights[maxHeap.top().second] != maxHeap.top().first) maxHeap.pop();
        int v, cv;
        tie(cv, v) = maxHeap.top();
        min_cut = min(min_cut, cv);
        // merge v into u
        for (int w : adj[v]) {
            if (w == u) continue;
            if (edges.find({u, w}) == edges.end()) {
                adj[u].push_back(w);
                adj[w].push_back(u);
                edges[{u, w}] = 0;
                edges[{w, u}] = 0;
            }
            edges[{u, w}] += edges[{v, w}];
            edges[{w, u}] += edges[{v, w}];
        }
        vis[v] = 1;
    }
    return min_cut;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    while (cin >> N) {
        cin >> M;
        vector<vector<int>> adj(N);
        map<pair<int, int>, int> edges;
        for (int i = 0; i < M; i++) {
            int u, v, w;
            cin >> u >> v >> w;
            adj[u].push_back(v);
            adj[v].push_back(u);
            edges[{u, v}] = w;
            edges[{v, u}] = w;
        }
        int cutValues = stoerWagner(adj, edges);
        cout << cutValues << endl;
    }
    return 0;
}
```

## Minimum cut Max flow theorem

```py

```

## fast networkx method using minimum cut

If s and t are in the same connected component in the partition from the global minimum cut it will not be the global minimum cut.  But when s and t belong to both sides then it finds the minimum cut

This works when you already know the global minimum cut and are just looking for the partition of nodes.

This method is really only possibly fast if you already know the global minimum cut and just want to find the partition of nodes.

```py
    nodes = set()
    edges = []
    for line in data:
        u, nei_nodes = line.split(": ")
        nodes.add(u)
        for v in nei_nodes.split():
            edges.append((u, v))
    nodes = list(nodes)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges, capacity = 1)
    s = nodes[0]
    for t in nodes[1:]:
        cut_value, partitions = nx.minimum_cut(G, s, t)
        if cut_value == 3: 
            print("part 1:", math.prod(len(p) for p in partitions))
            break
```

Finding the global minimum cut with this method, this is way slower than stoer wagner

```py
with open("big.txt", "r") as f:
    data = f.read().splitlines()
    nodes = set()
    edges = []
    for line in data:
        u, nei_nodes = line.split(": ")
        nodes.add(u)
        for v in nei_nodes.split():
            edges.append((u, v))
    nodes = list(nodes)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges, capacity = 1)
    u = nodes[0]
    min_cut = math.inf
    for v in nodes[1:]:
        cut_value, partitions = nx.minimum_cut(G, u, v)
        if cut_value < min_cut:
            min_cut = cut_value
```

Can also use stoer wagner in networkx

```py
    nodes = set()
    edges = []
    for line in data:
        u, nei_nodes = line.split(": ")
        nodes.add(u)
        for v in nei_nodes.split():
            edges.append((u, v))
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges, weight = 1)
    cut_value, partition = nx.stoer_wagner(G)
    print("part 1:", math.prod(map(len, partition)))
```