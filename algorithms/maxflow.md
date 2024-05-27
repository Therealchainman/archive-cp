# Maximum flow algorithms

Max Flow

# Ford Fulkerson Algorithm

## Dinics Algorithm for finding augmenting paths

### Dinics implemented in cpp can it is fast on cses OJ

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