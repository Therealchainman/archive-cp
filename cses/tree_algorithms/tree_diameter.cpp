#include <bits/stdc++.h>
using namespace std;

/*
There are n-1 edges in a tree.  

The diameter of a tree is the longest path between two nodes in the tree, in this problem the answer is
the number of edges between the two farthest nodes. 
*/
void dfs(int node, vector<vector<int>>& graph, vector<vector<int>>& dist, int i) {
    for (int& nei : graph[node]) {
        if (dist[i][nei]==0){
            dist[i][nei] = dist[i][node] + 1;
            dfs(nei, graph, dist,i);
        }
    }
}
int main() {
    int n, a,b;
    // freopen("input.txt", "r", stdin);
    // freopen("output.txt", "w", stdout);
    cin>>n;
    vector<vector<int>> graph(n+1);
    for(int i=0;i<n-1;i++){
        cin>>a>>b;
        graph[a].push_back(b);
        graph[b].push_back(a);
    }
    vector<vector<int>> dist(2,vector<int>(n+1,0));
    dist[0][1]=1;
    dfs(1,graph,dist,0);
    int node = max_element(dist[0].begin(), dist[0].end()) - dist[0].begin();
    dist[1][node]= 1;
    dfs(node,graph,dist,1);
    int diameter = *max_element(dist[1].begin(), dist[1].end())-1;
    cout<<diameter<<endl;
}