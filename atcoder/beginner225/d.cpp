#include <bits/stdc++.h>
using namespace std;
/*
This problem is easy, it is just some form of doubly linked list implementation
but really just need vector that points to front and back, and then you can add and remove two trains in O(1)
Then O(n) to find all 
*/
int main() {
    // freopen("inputs/input.txt","r",stdin);
    // freopen("outputs/output.txt","w",stdout);
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
