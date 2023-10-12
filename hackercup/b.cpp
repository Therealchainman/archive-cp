#include <bits/stdc++.h>
using namespace std;
#define int long long

inline int read() {
	int x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}

string name = "jacks_candy_shop_input.txt";

vector<int> parents, indegrees, candies, sz, heavy;
vector<vector<int>> adj;
vector<priority_queue<int>> heaps;

int dfs(int u, int p) {
    int hc = 0;
    for (int v : adj[u]) {
        if (v == p) continue;
        int csz = dfs(v, u);
        if (csz > hc) {
            hc = csz;
            heavy[u] = v;
        }
        sz[u] += csz;
    }
    return sz[u];
}

void solve() {
    int N = read(), M = read(), A = read(), B = read();
    parents.assign(N, -1);
    indegrees.assign(N, 0);
    indegrees[0]++;
    adj.assign(N, vector<int>());
    for (int i = 1; i < N; i++) {
        parents[i] = read();
        adj[i].push_back(parents[i]);
        adj[parents[i]].push_back(i);
        indegrees[i]++;
        indegrees[parents[i]]++;
    }
    candies.assign(N, 0);
    for (int i = 0; i < M; i++) {
        int candy = (A * i + B) % N;
        candies[candy]++;
    }
    sz.assign(N, 1);
    heavy.assign(N, -1);
    dfs(0, -1);
    int res = 0;
    deque<int> dq;
    for (int i = 0; i < N; i++) {
        if (indegrees[i] == 1) {
            dq.push_back(i);
        }
    }
    heaps.assign(N, priority_queue<int>());
    while (!dq.empty()) {
        int u = dq.front();
        dq.pop_front();
        for (int v : adj[u]) {
            if (v == parents[u]) {
                indegrees[v]--;
                if (indegrees[v] == 1) {
                    dq.push_back(v);
                }
            }
            if (v == heavy[u]) {
                heaps[u] = heaps[v];
            }
        }
        heaps[u].push(u);
        for (int v : adj[u]) {
            if (v == parents[u]) continue;
            if (v == heavy[u]) continue;
            while (!heaps[v].empty()) {
                int c = heaps[v].top();
                heaps[u].push(heaps[v].top());
                heaps[v].pop();
            }
        }
        while (candies[u] > 0 && !heaps[u].empty()) {
            res += heaps[u].top();
            heaps[u].pop();
            candies[u]--;
        }
    }
    cout << res;
}

int32_t main() {
	ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    string in = "inputs/" + name;
    string out = "outputs/" + name;
    freopen(in.c_str(), "r", stdin);
    freopen(out.c_str(), "w", stdout);
    int T = read();
    for (int i = 1; i <= T ; i++) {
        cout << "Case #" << i << ": ";
        solve();
        cout << endl;
    }
    return 0;
}

/*
problem solve

g++ "-Wl,--stack,1078749825" b.cpp -o main
*/
