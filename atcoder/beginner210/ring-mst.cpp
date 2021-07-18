#include <bits/stdc++.h>
using namespace std;
/*
approach1:  Kruskal's algo with minimum spanning tree and so on.  
*/
struct P {
    int x, y;
};

struct Edge {
    int i,j,w;
};
struct UnionFind {
    vector<int> parent, sizes;
    void init() {
        parent.resize(n);
        iota(parent.begin(),parent.end(),0);
        sizes.assign(n,1);
    }

    int find(int i)
    {
        if (i==parent[i]) {
            return i;
        }
        return parent[i]=find(parent[i]);
    }

    bool uunion(int i, int j)
    {
        i = find(i), j=find(j);
        if (i!=j) {
            if (sizes[j]>sizes[i]) {
                swap(i,j);
            }
            parent[j]=i;
            sizes[i]+=sizes[j];
            return true;
        }
        return false;
    }
};
int main() {
    int N, M, a, c;
    cin>>N>>M;
    vector<int> A(M,0), C(M,0);
    for (int i = 0;i<M;i++) {
        cin>>a>>c;
        A[i]=a;
        C[i]=c;
    }
    // construct the graph

    // Perform kruskals'algorithm, keeping mincost
    int minCost = 0;
    // iterate through the sorted edges and connect until
    // the entire tree is connected
    UnionFind ds;
    ds.init(n);
    if (ds.uunion(i,j)) {
        // update mincost
    }
}