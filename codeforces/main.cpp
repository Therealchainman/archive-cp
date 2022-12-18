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