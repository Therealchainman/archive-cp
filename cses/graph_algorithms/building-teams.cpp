#include <bits/stdc++.h>
using namespace std;
vector<int> colors;
vector<vector<int>> graph;
const string IMP = "IMPOSSIBLE";
bool isBipartite(int n) {
    queue<int> q;
    for (int u = 1;u<=n;u++) {
        if (colors[u]==-1) {
            colors[u]=0;
            q.push(u);
            while (!q.empty()) {
                int v = q.front();
                q.pop();
                for (int w : graph[v]) {
                    if (colors[w]==-1) {
                        colors[w] = colors[v]^1;
                        q.push(w);
                    } else {
                        if (colors[w]==colors[v]) {
                            return false;
                        }
                    }
                }
            }
        }
    }
    return true;
}
int main() {
    int n, m, a, b;
    cin >> n >> m;
    graph.resize(n+1);
    for (int i = 0;i<m;i++) {
        cin>>a>>b;
        graph[a].push_back(b);
        graph[b].push_back(a);
    }
    colors.assign(n+1,-1);
    if (isBipartite(n)) {
        for (int i = 1;i<=n;i++) {
            cout << colors[i]+1 << " ";
        }
        cout<<endl;
    } else {
        cout << IMP << endl;
    }
}