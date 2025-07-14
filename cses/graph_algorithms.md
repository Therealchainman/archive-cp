# Graph Algorithms

## USED IN SUBMISSIONS

```py
import os,sys
from io import BytesIO, IOBase
from typing import *
sys.setrecursionlimit(1_000_000)
# only use pypyjit when needed, it usese more memory, but speeds up recursion in pypy
import pypyjit
pypyjit.set_param('max_unroll_recursion=-1')

# Fast IO Region
BUFSIZE = 8192
class FastIO(IOBase):
    newlines = 0
    def __init__(self, file):
        self._fd = file.fileno()
        self.buffer = BytesIO()
        self.writable = "x" in file.mode or "r" not in file.mode
        self.write = self.buffer.write if self.writable else None
    def read(self):
        while True:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            if not b:
                break
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines = 0
        return self.buffer.read()
    def readline(self):
        while self.newlines == 0:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            self.newlines = b.count(b"\n") + (not b)
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines -= 1
        return self.buffer.readline()
    def flush(self):
        if self.writable:
            os.write(self._fd, self.buffer.getvalue())
            self.buffer.truncate(0), self.buffer.seek(0)
class IOWrapper(IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        self.write = lambda s: self.buffer.write(s.encode("ascii"))
        self.read = lambda: self.buffer.read().decode("ascii")
        self.readline = lambda: self.buffer.readline().decode("ascii")
sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)
input = lambda: sys.stdin.readline().rstrip("\r\n")
```

## Planets and Kingdoms

### Solution 1:  Counting and finding strongly connected components + tarjan's algorithm + dfs

```py
sys.setrecursionlimit(1_000_000)

def main():
    n, m = map(int, input().split())
    adj_list = [[] for _ in range(n + 1)]
    for _ in range(m):
        a, b = map(int, input().split())
        adj_list[a].append(b)
    time = num_scc = 0
    scc_ids = [0]*(n + 1)
    disc, low, on_stack = [0]*(n + 1), [0]*(n + 1), [0]*(n + 1)
    stack = []
    def dfs(node):
        nonlocal time, num_scc
        time += 1
        disc[node] = time
        low[node] = disc[node]
        on_stack[node] = 1
        stack.append(node)
        for nei in adj_list[node]:
            if not disc[nei]: dfs(nei)
            if on_stack[nei]: low[node] = min(low[node], low[nei])
        # found scc
        if disc[node] == low[node]:
            num_scc += 1
            while stack:
                snode = stack.pop()
                on_stack[snode] = 0
                low[snode] = low[node]
                scc_ids[snode] = num_scc
                if snode == node: break
    for i in range(1, n + 1):
        if disc[i]: continue
        dfs(i)
    print(num_scc)
    print(*scc_ids[1:])

if __name__ == '__main__':
    main()
```

## Coin Collector

### Solution 1:  Create component/condensation graph from strongly connected components + dynamic programming on the dag/trees

You can collect all the coins in any strongly connected component because every node is reachable from every other node.  Then you can create a component graph where each component is a node and there is an edge between two components if there is an edge between two nodes in the original graph.  Then you can do a topological sort on the component graph and do dynamic programming on the component graph to find the maximum number of coins you can collect.

condensation graph is a contraction of each scc into a single vertex, and creates a dag.

```py
def main():
    n, m = map(int, input().split())
    coins = [0] + list(map(int, input().split()))
    # PHASE 0: CONSTRUCT ADJACENCY LIST REPRESENTATION OF GRAPH
    adj_list = [[] for _ in range(n + 1)]
    for _ in range(m):
        a, b = map(int, input().split())
        adj_list[a].append(b)
    # PHASE 1: FIND STRONGLY CONNECTED COMPONENTS
    time = num_scc = 0
    scc_ids = [0]*(n + 1)
    disc, low, on_stack = [0]*(n + 1), [0]*(n + 1), [0]*(n + 1)
    stack = []
    def dfs(node):
        nonlocal time, num_scc
        time += 1
        disc[node] = time
        low[node] = disc[node]
        on_stack[node] = 1
        stack.append(node)
        for nei in adj_list[node]:
            if not disc[nei]: dfs(nei)
            if on_stack[nei]: low[node] = min(low[node], low[nei])
        # found scc
        if disc[node] == low[node]:
            num_scc += 1
            while stack:
                snode = stack.pop()
                on_stack[snode] = 0
                low[snode] = low[node]
                scc_ids[snode] = num_scc
                if snode == node: break
    for i in range(1, n + 1):
        if disc[i]: continue
        dfs(i)
    # PHASE 2: CONSTRUCT CONDENSATION GRAPH
    scc_adj_list = [[] for _ in range(num_scc + 1)]
    indegrees = [0]*(num_scc + 1)
    # condensing the values of the coins into it's scc
    val_scc = [0]*(num_scc + 1)
    for i in range(1, n + 1):
        val_scc[scc_ids[i]] += coins[i]
        for nei in adj_list[i]:
            if scc_ids[i] != scc_ids[nei]:
                indegrees[scc_ids[nei]] += 1
                scc_adj_list[scc_ids[i]].append(scc_ids[nei])
    # PHASE 3: DO TOPOLOGICAL SORT ON CONDENSATION GRAPH WITH MEMOIZATION FOR MOST COINS COLLECTED IN EACH NODE IN CONDENSATION GRAPH
    stack = []
    memo = [0]*(num_scc + 1)
    for i in range(1, num_scc + 1):
        if indegrees[i] == 0:
            stack.append(i)
            memo[i] = val_scc[i]
    while stack:
        node = stack.pop()
        for nei in scc_adj_list[node]:
            indegrees[nei] -= 1
            memo[nei] = max(memo[nei], memo[node] + val_scc[nei])
            if indegrees[nei] == 0: stack.append(nei)
    print(max(memo))
        
if __name__ == '__main__':
    main() 
```

## Download Speed

### Solution 1: dinics algorithm 

```cpp
#include <bits/stdc++.h>
using namespace std;
 
struct FlowEdge {
    int src, dst;
    long long cap, flow = 0;
    void init(int u, int v, long long c) {
        src = u;
        dst = v;
        cap = c;
    }
};
 
struct Dinic {
    const long long flow_inf = 1e18;
    vector<FlowEdge> edges;
    vector<vector<int>> adj_list;
    int size, num_edges = 0;
    int source, sink;
    vector<int> level, ptr;
    queue<int> q;
 
    void init(int sz, int src, int snk) {
        size = sz;
        source = src;
        sink = snk;
        adj_list.resize(size);
        level.resize(size);
        ptr.resize(size);
    }
 
    void add_edge(int src, int dst, long long cap) {
        FlowEdge fedge;
        fedge.init(src, dst, cap);
        edges.emplace_back(fedge);
        FlowEdge residual_fedge;
        residual_fedge.init(dst, src, 0);
        edges.emplace_back(residual_fedge);
        adj_list[src].push_back(num_edges);
        adj_list[dst].push_back(num_edges + 1);
        num_edges += 2;
    }
 
    bool bfs() {
        while (!q.empty()) {
            int node = q.front();
            q.pop();
            for (int index: adj_list[node]) {
                if (edges[index].cap - edges[index].flow < 1) continue;
                if (level[edges[index].dst] != -1) continue;
                level[edges[index].dst] = level[node] + 1;
                q.push(edges[index].dst);
            }
        }
        return level[sink] != -1;
    }
 
    long long dfs(int node, long long pushed) {
        if (pushed == 0) return 0;
        if (node == sink) return pushed;
        for (int& cid = ptr[node]; cid < (int)adj_list[node].size(); cid++ ) {
            int id = adj_list[node][cid];
            int nei = edges[id].dst;
            if (level[node] + 1 != level[nei] || edges[id].cap - edges[id].flow < 1) continue;
            long long flow = dfs(nei, min(pushed, edges[id].cap - edges[id].flow));
            if (flow == 0) continue;
            edges[id].flow += flow;
            edges[id^1].flow -= flow;
            return flow;
        }
        return 0;
    }
 
    long long main() {
        long long maxflow = 0;
        while (true) {
            fill(level.begin(), level.end(), -1);
            level[source] = 0;
            q.push(source);
            if (!bfs()) break;
            fill(ptr.begin(), ptr.end(), 0);
            while (long long pushed = dfs(source, flow_inf)) {
                maxflow += pushed;
            }
        }
        return maxflow;
    }
};
int main() {
    Dinic dinic;
    int n, m;
    cin >> n >> m;
    dinic.init(n, 0, n - 1);
    for (int i = 0; i < m; i++) {
        int src, dst, cap;
        cin >> src >> dst >> cap;
        dinic.add_edge(src - 1, dst - 1, cap);
    }
    long long res = dinic.main();
    cout << res << endl;
    return 0;
}
```

### Solution 2: ford fulkerson dfs algorithm

```py
class MaxFlow:
    def __init__(self, n: int, edges: List[Tuple[int, int, int]]):
        self.size = n
        self.edges = edges
 
    def build(self, n: int, edges: List[Tuple[int, int, int]]) -> None:
        self.adj_list = {}
        for u, v, cap in edges:
            if u not in self.adj_list:
                self.adj_list[u] = Counter()
            self.adj_list[u][v] += cap
            if v not in self.adj_list:
                self.adj_list[v] = Counter()
 
    def main(self, source: int, sink: int) -> int:
        self.build(self.size, self.edges)
        maxflow = 0
        while True:
            self.reset()
            cur_flow = self.dfs(source, sink, math.inf)
            if cur_flow == 0:
                break
            maxflow += cur_flow
        return maxflow
 
    def reset(self) -> None:
        self.vis = [False] * self.size
 
    def dfs(self, node: int, sink: int, flow: int) -> int:
        if node == sink:
            return flow
        self.vis[node] = True
        cap = self.adj_list[node]
        for nei, cap in cap.items():
            if not self.vis[nei] and cap > 0:
                cur_flow = self.dfs(nei, sink, min(flow, cap))
                if cur_flow > 0:
                    self.adj_list[node][nei] -= cur_flow
                    self.adj_list[nei][node] += cur_flow
                    return cur_flow
        return 0
 
def main():
    n, m = map(int, input().split())
    edges = [None] * m
    mx = 0
    for i in range(m):
        u, v, cap = map(int, input().split())
        edges[i] = (u - 1, v - 1, cap)
        mx = max(mx, cap)
    source, sink = 0, n - 1
    maxflow = MaxFlow(n, edges)
    return maxflow.main(source, sink)
 
if __name__ == '__main__':
    print(main())
```

## Distinct Routes

### Solution 1: dinics algorithm + general path cover + edge disjoint paths

```py
class FordFulkersonMaxFlow:
    """
    Ford-Fulkerson algorithm 
    - pluggable augmenting path finding algorithms
    - residual graph
    - bottleneck capacity
    """
    def __init__(self, n: int, edges: List[Tuple[int, int, int]]):
        self.size = n
        self.edges = edges
        self.cap = defaultdict(Counter)
        self.flow = defaultdict(Counter)
        self.adj_list = [[] for _ in range(self.size)]

    def build(self) -> None:
        self.delta = 0
        for src, dst, cap in self.edges:
            self.cap[src][dst] += cap
            self.adj_list[src].append(dst)
            self.adj_list[dst].append(src) # residual edge
            self.delta = max(self.delta, self.cap[src][dst])
        highest_bit_set = self.delta.bit_length() - 1
        self.delta = 1 << highest_bit_set

    def residual_capacity(self, src: int, dst: int) -> int:
        return self.cap[src][dst] - self.flow[src][dst]

    def main_dfs(self, source: int, sink: int) -> int:
        self.build()
        maxflow = 0
        while True:
            self.reset()
            cur_flow = self.dfs(source, sink, math.inf)
            if cur_flow == 0:
                break
            maxflow += cur_flow
        return maxflow

    def neighborhood(self, node: int) -> List[int]:
        return (i for i in self.adj_list[node])

    def dinics_bfs(self, source: int, sink: int) -> bool:
        self.distances = [-1] * self.size
        self.distances[source] = 0
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for nei in self.neighborhood(node):
                if self.distances[nei] == -1 and self.residual_capacity(node, nei) > 0:
                    self.distances[nei] = self.distances[node] + 1
                    queue.append(nei)
        return self.distances[sink] != -1

    def dinics_dfs(self, node: int, sink: int, flow: int) -> int:
        if flow == 0: return 0
        if node == sink: return flow
        while self.ptr[node] < len(self.adj_list[node]):
            nei = self.adj_list[node][self.ptr[node]]
            self.ptr[node] += 1
            if self.distances[nei] == self.distances[node] + 1 and self.residual_capacity(node, nei) > 0:
                cur_flow = self.dinics_dfs(nei, sink, min(flow, self.residual_capacity(node, nei)))
                if cur_flow > 0:
                    self.flow[node][nei] += cur_flow
                    self.flow[nei][node] -= cur_flow
                    return cur_flow
        return 0

    def main_dinics(self, source: int, sink: int) -> int:
        self.build()
        maxflow = 0
        while self.dinics_bfs(source, sink):
            self.ptr = [0] * self.size # pointer to the next edge to be processed (optimizes for dead ends)
            while True:
                cur_flow = self.dinics_dfs(source, sink, math.inf)
                if cur_flow == 0:
                    break
                maxflow += cur_flow
        return maxflow
    
    def general_path_cover(self, source: int, sink: int) -> int:
        self.path, self.paths = [], []
        """
        Since it is possible for there to be a cycle in the graph for the path, that is it could go from node source -> 1 -> 4 -> 1 -> ... -> sink, and that is a valid path
        since it uses disjoint edges to get from source to sink. 
        So if you use a parent array to store prevent cycles, you can get stopped for instance if you go from 1 -> 4 then it won't get back to 1 and could be at dead end.
        You in a sense get stuck in a cycle. 
        But this is already known because the parent array only works for trees or acyclic graphs and not for graphs with cycles.
        So you have to use a set of visited edges (node pairs) to prevent being in cycle for infinite, and because only edge can be used once in a general cover path
        """
        self.vis = set()
        for nei in self.neighborhood(source):
            if self.flow[source][nei] != 1: continue
            self.vis.add((source, nei)) 
            self.path.append(source)
            self.path_dfs(nei, sink)
            self.path.pop()
        return self.paths

    def path_dfs(self, node: int, sink: int) -> None:
        if node == sink:
            self.paths.append([i + 1 for i in self.path + [node]])
            return
        for nei in self.neighborhood(node):
            if (node, nei) in self.vis: continue
            if self.flow[node][nei] == 1:
                self.vis.add((node, nei))
                self.path.append(node)
                self.path_dfs(nei, sink)
                self.path.pop()
                return

def main():
    n, m = map(int, input().split())
    edges = [None] * m
    for i in range(m):
        u, v = map(int, input().split())
        edges[i] = (u - 1, v - 1, 1)
    source, sink = 0, n - 1
    maxflow = FordFulkersonMaxFlow(n, edges)
    mf = maxflow.main_dinics(source, sink)
    print(mf)
    paths = maxflow.general_path_cover(source, sink)
    for path in paths:
        print(len(path))
        print(*path)
    
if __name__ == '__main__':
    main()
```

## Mail Delivery

### Solution 1:  Eulerian circuit + hierholzer's algorithm + undirected graph

```py
def eulerian_circuit(adj_list, degrees):
    # start node is 1 in this instance
    n = len(degrees)
    start_node = 1
    stack = [start_node]
    vis = [0] * (n + 1)
    vis[start_node] = 1
    while stack:
        node = stack.pop()
        for nei in adj_list[node]:
            if vis[nei]: continue
            vis[nei] = 1
            stack.append(nei)
    for i in range(n):
        if (degrees[i] & 1) or (degrees[i] > 0 and not vis[i]): return False
    return True

def hierholzers_undirected(adj_list):
    start_node = 1
    stack = [start_node]
    circuit = []
    while stack:
        node = stack[-1]
        if len(adj_list[node]) == 0:
            circuit.append(stack.pop())
        else:
            nei = adj_list[node].pop()
            adj_list[nei].remove(node)
            stack.append(nei)
    return circuit

def main():
    n, m = map(int, input().split())
    adj_list = [set() for _ in range(n + 1)]
    degrees = [0] * (n + 1)
    for _ in range(m):
        u, v = map(int, input().split())
        adj_list[u].add(v)
        adj_list[v].add(u)
        degrees[u] += 1
        degrees[v] += 1
    # all degrees are even and one connected component with edge (nonzero degrees)
    if not eulerian_circuit(adj_list, degrees):
        return "IMPOSSIBLE"
    # hierholzer's algorithm to reconstruct the eulerian circuit
    circuit = hierholzers_undirected(adj_list)
    return ' '.join(map(str, circuit))

if __name__ == '__main__':
    print(main())
```

## Teleporters Path

### Solution 1:  Eulerian path + hierholzer's algorithm + directed graph

```py
def is_eulerian_path(n, adj_list, indegrees, outdegrees):
    # start node is 1 in this instance
    start_node = 1
    end_node = n
    stack = [start_node]
    vis = [0] * (n + 1)
    vis[start_node] = 1
    while stack:
        node = stack.pop()
        for nei in adj_list[node]:
            if vis[nei]: continue
            vis[nei] = 1
            stack.append(nei)
    if outdegrees[start_node] - indegrees[start_node] != 1 or indegrees[end_node] - outdegrees[end_node] != 1: return False
    for i in range(1, n + 1):
        if ((outdegrees[i] > 0 or indegrees[i] > 0) and not vis[i]): return False
        if (indegrees[i] != outdegrees[i] and i not in (start_node, end_node)): return False
    return True

def hierholzers_directed(n, adj_list):
    start_node = 1
    end_node = n
    stack = [start_node]
    euler_path = []
    while stack:
        node = stack[-1]
        if len(adj_list[node]) == 0:
            euler_path.append(stack.pop())
        else:
            nei = adj_list[node].pop()
            stack.append(nei)
    return euler_path[::-1]

def main():
    n, m = map(int, input().split())
    adj_list = [set() for _ in range(n + 1)]
    indegrees, outdegrees = [0] * (n + 1), [0] * (n + 1)
    for _ in range(m):
        u, v = map(int, input().split())
        adj_list[u].add(v)
        indegrees[v] += 1
        outdegrees[u] += 1
    # all degrees are even and one connected component with edge (nonzero degrees)
    if not is_eulerian_path(n, adj_list, indegrees, outdegrees):
        return "IMPOSSIBLE"
    # hierholzer's algorithm to reconstruct the eulerian circuit
    eulerian_path = hierholzers_directed(n, adj_list)
    return ' '.join(map(str, eulerian_path))

if __name__ == '__main__':
    print(main())
```

## Planets Queries I

### Solution 1:  binary jumping + functional graph or successor graph + time complexity O(nlogk + qlogk)

A functional graph means each node has an outdegree of 1, that is has one output. This means you will have islands but each island contains a cycle.  A cycle must always exist in a weakly connected component of a functional graph.  This is also a directed graph.  You can use binary jumping to find the power of twos successors from each node.  And you can convert any number to a summation of power of two jumps to calculate any possible jump.  Any jump can be composed of power of two jumps. 

```py
def main():
    LOG = 31
    n, q = map(int, input().split())
    successor = list(map(int, input().split()))
    queries = [tuple(map(int, input().split())) for _ in range(q)]
    succ = [[0] * n for _ in range(LOG)]
    succ[0] = [s - 1 for s in successor]
    for i in range(1, LOG):
        for j in range(n):
            succ[i][j] = succ[i - 1][succ[i - 1][j]]
    for x, k in queries:
        x -= 1
        for i in range(LOG):
            if (k >> i) & 1:
                x = succ[i][x]
        print(x + 1)

if __name__ == '__main__':
    main()
```

## Planets Queries II

### Solution 1:  binary jumping + functional graph or successor graph + cycles + two cases

binary jumping is being used to check if their is a node that is k distance away. 

Needs to use a dfs to store cycle information and distance from a node belonging to a cycle. for non cycle nodes.
Use a parent array and backtracking to reconstruct cycle path and update the distances.  

```py
def main():
    n, q = map(int, input().split())
    successors = [x - 1 for x in map(int, input().split())]
    LOG = 19
    succ = [[0] * n for _ in range(LOG)]
    succ[0] = successors[:]
    for i in range(1, LOG):
        for j in range(n):
            succ[i][j] = succ[i - 1][succ[i - 1][j]]
    cycle_count = 0
    cycle_ids = [-1] * n # map node -> cycle_id
    cycle_indices = [-1] * n # map node -> cycle_index
    cycle_lens = []
    tree_dist = [0] * n # map node -> distance from root node or cycle node
    vis = [False] * n
    def dfs(u):
        nonlocal cycle_count
        parent = {u: None}
        is_cycle = False
        while True:
            v = successors[u]
            if v in parent: 
                is_cycle = True
                break
            if vis[v]: break
            parent[v] = u
            u = v
        if is_cycle:
            crit_point = parent[successors[u]]
            cycle_path = []
            while u != crit_point:
                cycle_ids[u] = cycle_count
                cycle_path.append(u)
                u = parent[u]
            cycle_path = cycle_path[::-1]
            cycle_lens.append(len(cycle_path))
            for i, node in enumerate(cycle_path):
                cycle_indices[node] = i
                vis[node] = True
            cycle_count += 1
        while u is not None:
            vis[u] = True
            tree_dist[u] = tree_dist[successors[u]] + 1
            u = parent[u]
    for u in range(n):
        if vis[u]: continue
        vis[u] = True
        dfs(u)
    def cycle_distance(u, v):
        if cycle_ids[u] != cycle_ids[v]: return -1
        u_i, v_i = cycle_indices[u], cycle_indices[v]
        cid = cycle_ids[u]
        return v_i - u_i if v_i >= u_i else cycle_lens[cid] - (u_i - v_i)
    for _ in range(q):
        u, v = map(int, input().split())
        u -= 1
        v -= 1
        if cycle_ids[u] != -1 and cycle_ids[v] != -1: # case 1: both nodes belong to cycles
            res = cycle_distance(u, v)
        else: # case 2: both nodes do not belong to cycles
            k = tree_dist[u] - tree_dist[v]
            if k < 0: res = -1
            else:
                w = u # dummy node
                for i in range(LOG):
                    if (k >> i) & 1:
                        w = succ[i][w]
                if v != w and cycle_ids[v] != -1 and cycle_ids[w] != -1: # became case 1
                    res = cycle_distance(w, v)
                    res = res + (k if res >= 0 else 0)
                else:
                    res = k if w == v else -1
        print(res)

if __name__ == '__main__':
    main()
```

## Giant Pizza

### Solution 1:  2SAT, strongly connected components, Tarjan's algorithm, topological sort

```cpp
int N, M, timer, scc_count;
vector<vector<int>> adj, cond_adj;
vector<int> disc, low, comp;
stack<int> stk;
vector<bool> on_stack;
 
void dfs(int u) {
    disc[u] = low[u] = ++timer;
    stk.push(u);
    on_stack[u] = true;
    for (int v : adj[u]) {
        if (not disc[v]) dfs(v);
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
 
signed main() {
	cin >> M >> N;
    adj.assign(2 * N, vector<int>());
    disc.assign(2 * N, 0);
    low.assign(2 * N, 0);
    comp.assign(2 * N, -1);
    on_stack.assign(2 * N, false);
    scc_count = -1;
    for (int i = 0; i < M; i++) {
        char s1, s2;
        int u, v;
        cin >> s1 >> u >> s2 >> v;
        u--; v--;
        if (s1 == '-') u = N + u;
        if (s2 == '-') v = N + v;
        // implications
        adj[(u + N) % (2 * N)].push_back(v);
        adj[(v + N) % (2 * N)].push_back(u);
    }
    for (int i = 0; i < 2 * N; i++) {
        if (not disc[i]) dfs(i);
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
    for (int i = 0; i < N; i++) {
        cout << (ans[i] ? '+' : '-') << " ";
    }
    cout << endl;
}

```

## 

### Solution 1:  

```cpp

```

## 

### Solution 1:  

```cpp

```