#include <bits/stdc++.h>
using namespace std;

int main() {
    int N,u,v,w;
    cin>>N;
    vector<vector<int>> edges;
    for (int i = 0;i<N;i++) {
        cin>>u>>v>>w;
        edges.push_back({u,v,w});
    }
    int ans = 0;
    
    cout<<ans<<endl;
}
