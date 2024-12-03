

## L. The Shortest Path

### Solution 1:  Johnson's algorithm

```py
from typing import List
import math
import heapq
from itertools import product

import os,sys
from io import BytesIO, IOBase

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

def bellmanFord(n: int, src: int, edges: List[List[int]]) -> List[int]:
    dist = [math.inf]*n
    dist[src] = 0
    for _ in range(n-1):
        any_relaxed = False
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                any_relaxed = True
                dist[v] = dist[u] + w
        if not any_relaxed: break
    # check for any negative cycles
    for u, v, w in edges:
        if dist[v] > dist[u] + w: return []
    return dist

def dijkstra(n: int, src: int, adj_list: List[List[int]]) -> List[int]:
    dist = [math.inf]*n
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: continue
        for v, w in adj_list[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    return dist

def johnsons(n: int, edges: List[List[int]]) -> List[List[int]]:
    # create a new vertex that is connected to all other vertices with weight 0
    # new vertex that will be the source for bellman fourd is going to be n
    # run bellman ford to find shortest paths from the new vertex to all other vertices
    dist = bellmanFord(n+1, n, edges + [[n, i, 0] for i in range(n)])
    if not dist: return [] # if it has negative cycle
    # reweight the edges
    for i in range(len(edges)):
        u, v, w = edges[i]
        edges[i][2] = w + dist[u] - dist[v]
    # run dijkstra for each vertex
    adj_list = [[] for _ in range(n)]
    for u, v, w in edges:
        adj_list[u].append((v, w))
    shortest_paths = [dijkstra(n, i, adj_list) for i in range(n)]
    # undo the reweighting
    for u, v in product(range(n), repeat = 2):
        if shortest_paths == math.inf: continue
        shortest_paths[u][v] = shortest_paths[u][v] + dist[v] - dist[u]
    return shortest_paths

def main():
    n, m = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v, w = map(int, input().split())
        edges.append([u-1, v-1, w])
    shortest_paths = johnsons(n, edges)
    minVal = math.inf
    for row in shortest_paths:
        for val in row:
            minVal = min(minVal, val)
    return minVal if minVal != math.inf else -math.inf

if __name__ == '__main__':
    t = int(input())
    for _ in range(t):
        print(main())
```

```cpp
#include <bits/stdc++.h>
using namespace std;

inline int read()
{
	int x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}

vector<int> bellmanFord(int n, int src, vector<vector<int>>& edges) {
    vector<int> dist(n, INT_MAX);
    dist[src] = 0;
    for (int i = 0; i < n-1; i++) {
        bool any_relaxed = false;
        for (auto& e : edges) {
            int u = e[0], v = e[1], w = e[2];
            if (dist[u] != INT_MAX && dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                any_relaxed = true;
            }
        }
        if (!any_relaxed) break;
    }
    for (auto& e : edges) {
        int u = e[0], v = e[1], w = e[2];
        if (dist[u] != INT_MAX && dist[u] + w < dist[v]) {
            return {};
        }
    }
    return dist;
}

vector<int> dijkstra(int n, int src, vector<vector<pair<int, int>>>& adj) {
    vector<int> dist(n, INT_MAX);
    dist[src] = 0;
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    pq.push({0, src});
    while (!pq.empty()) {
        int u = pq.top().second;
        int d = pq.top().first;
        pq.pop();
        if (d > dist[u]) continue;
        for (auto& e : adj[u]) {
            int v = e.first, w = e.second;
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});
            }
        }
    }
    return dist;
}

vector<vector<int>> johnsons(int n, vector<vector<int>>& edges) {
    vector<int> h = bellmanFord(n, n-1, edges);
    if (h.empty()) return {};
    for (auto& e : edges) {
        int u = e[0], v = e[1], w = e[2];
        e[2] = w + h[u] - h[v];
    }
    if (h.empty()) return {};
    vector<vector<pair<int, int>>> adj(n);
    for (auto& e : edges) {
        int u = e[0], v = e[1], w = e[2];
        adj[u].push_back({v, w});
    }
    vector<vector<int>> dist(n);
    for (int i = 0; i < n; i++) {
        dist[i] = dijkstra(n, i, adj);
        for (int j = 0; j < n; j++) {
            if (dist[i][j] != INT_MAX) {
                dist[i][j] += h[j] - h[i];
            }
        }
    }
    return dist;
}

int shortest_path() {
    int n = read(), m = read();
    vector<vector<int>> edges;
    for (int i = 0; i < m; i++) {
        int u = read(), v = read(), w = read();
        edges.push_back({u-1, v-1, w});
    }
    vector<vector<int>> dist = johnsons(n, edges);
    if (dist.empty()) return INT_MIN;
    int result = INT_MAX;
    for (int i = 0; i < n; i++) {
        for (int j = 0;j < n; j++) {
            result = min(result, dist[i][j]);
        }
    }
    return result;
}

int main() {
    int t = read();
    for (int i = 0; i < t; i++) {
        int res = shortest_path();
        if (res == INT_MIN) {
            cout << "-inf" << endl;
        } else {
            cout << res << endl;
        }
    }
    return 0;
}
```
