#include <bits/stdc++.h>
using namespace std;
#define int long long
#define x first
#define h second

inline int read() {
	int x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}

string name = "wiki_race_sample_input.txt";

int N;
vector<int> parent;
vector<vector<int>> adj;
vector<set<string>> topics, other_topics;

int dfs(int u) {
    int heavy = -1, max_topics = 0;
    for (int v : adj[u]) {
        if (parent[u] == v) continue;
        if (topics[v].size() > max_topics) {
            max_topics = topics[v].size();
            heavy = v;
        }
    }
    for (int v : adj[u]) {
        if (parent[u] == v) continue;
        dfs(v);
    }
    cout << "u: " << u << "heavy: " << heavy << "max_topics: " << max_topics << endl;
    return 0;
}

int solve() {
    N = read();
    parent.assign(N, -1);
    adj.assign(N, vector<int>());
    for (int i = 1; i < N; i++) {
        parent[i] = read() - 1;
        adj[parent[i]].push_back(i);
    }
    topics.assign(N, set<string>());
    other_topics.assign(N, set<string>());
    for (int i = 0; i < N; i++) {
        int M = read();
        for (int j = 0; j < M; j++) {
            string top;
            cin >> top;
            cout << top << endl;
            topics[i].insert(top);
        }
        cout << topics[i].size() << endl;
    }
    int res = dfs(0);
    return res;
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
        cout << "Case #" << i << ": " << solve() << endl;
    }
    return 0;
}

/*
problem solve

g++ "-Wl,--stack,1078749825" d.cpp -o main
*/
