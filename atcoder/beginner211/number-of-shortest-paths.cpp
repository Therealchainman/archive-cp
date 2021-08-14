#include <bits/stdc++.h>
/*
Easily mislead to think dijkstra, but bfs will work here. 
*/
using namespace std;
const int mod = 1e9+7;
int main() {
    int N, M, a,b;
    cin>>N>>M;
    unordered_map<int, vector<int>> graph(N);
    for (int i = 0;i<M;i++) {
        cin>>a>>b;
        graph[a].push_back(b);
        graph[b].push_back(a);
    }
    unordered_set<int> unvisited;
    for (int i = 1;i<=N;i++) {
        unvisited.insert(i);
    }
    int minDist = 1e9, level = 0;
    vector<int> dp(N+1,0); // store the count of nodes that reach this one 
    queue<vector<int>> q;
    unordered_set<int> visited;
    q.push({0,1});
    dp[1]=1;
    unvisited.erase(1);
    while (!q.empty()) {
        auto item = q.front();
        q.pop();
        int dist = item[0], city=item[1];
        if (dist==minDist) {
            break;
        } else if (dist>level) {
            for (int v : visited) {
                unvisited.erase(v);
            }
            visited.clear();
            level = dist;
        }
        for (int nei : graph[city]) {
            if (unvisited.find(nei)!=unvisited.end()) {
                if (nei==N) {
                    minDist = dist+1;
                }
                dp[nei] = (dp[city]+dp[nei])%mod;
                if (visited.count(nei)==0) {
                    q.push({dist+1,nei});
                    visited.insert(nei);
                }
            }
        }
    }
    printf("%d\n", dp[N]);
}