#include <bits/stdc++.h>
using namespace std;

/*
This one is kinda tricky, but really all I'm doing is a normal BFS of the graph
since there are M edges, that will give about a time comlexity of O(N) for the most part.
anything that is not good will be trimmed from bfs, so it is not visiting any element twice.  
*/

int main() {
    int N, M, h, u, v,d;
    cin>>N>>M;
    vector<vector<int>> graph(N);
    vector<long long> H, dist(N,INT_MIN);
    dist[0] = 0;
    for (int i = 0;i<N;i++) {
        cin>>h;
        H.push_back(h);
    }
    for(int i=0;i<M;i++){
        cin>>u>>v;
        u--;
        v--;
        graph[u].push_back(v);
        graph[v].push_back(u);
    }
    queue<pair<long long,int>> q;
    q.emplace(0,0);
    long long best = 0;
    while (!q.empty()) {
        tie(d,u) = q.front();
        q.pop();
        if (d < dist[u]) continue;
        for (int nei : graph[u]) {
            long long diff = H[u]-H[nei];
            if (diff<0) {
                diff*=2;
            }
            long long ncost = d + diff;
            if (ncost>dist[nei]) {
                best = max(best, ncost);
                dist[nei] = ncost;
                q.emplace(ncost,nei);
            }
        }
    }
    cout<<best<<endl;
}