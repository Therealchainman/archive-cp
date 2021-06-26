#include <bits/stdc++.h>
using namespace std;

/*
Kruskal's algorithm to find maximum spanning tree in a graph
*/

struct Edge {
    int u, v, w;
};

struct UnionFind {
    vector<int> parents, sizes;
    void init(int n) {
        parents.resize(n);
        iota(parents.begin(),parents.end(),0);
        sizes.assign(n, 1);
    }

    int ufind(int i) {
        if (i==parents[i]) {
            return i;
        }
        return parents[i]=ufind(parents[i]);
    }

    bool uunion(int i, int j) {
        i = ufind(i), j=ufind(j);
        if (i!=j) {
            if (sizes[i]<sizes[j]) {
                swap(i,j);
            }
            parents[j]=i;
            sizes[i]+=sizes[j];
            return true;
        }
        return false;
    }
};


int main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T,N,a,b,trash;
    cin >> T;
    for (int t = 1; t <= T; t++)
    {
        cin>>N;
        vector<vector<int>> A(N, vector<int>(N,0));
        for (int i = 0;i<N;i++) {
            for (int j = 0;j<N;j++) {
                cin>>a;
                A[i][j]=a;
            }
        }
        vector<vector<int>> B(N, vector<int>(N,0));
        for (int i = 0;i<N;i++) {
            for (int j =0;j<N;j++) {
                cin>>b;
                B[i][j]=b;
            }
        }
        // Don't need the checksum values
        for (int i = 0;i<N;i++) {
            cin>>trash;
        }
        for (int i = 0;i<N;i++) {
            cin>>trash;
        }
        vector<Edge> q;
        int total = 0;
        for (int i = 0;i<N;i++) {
            for (int j = 0;j<N;j++) {
                if (B[i][j]>0) {
                    q.push_back(Edge{i,N+j,B[i][j]});
                    total+=B[i][j];
                }
            }
        }
        sort(q.begin(),q.end(),[](const Edge& a, const Edge& b) {
            return a.w<b.w;
        });
        // The entire priority queue is populated with the edges
        int maxCost = 0;
        UnionFind ds;
        ds.init(2*N);
        // Construct the maximum spanning tree
        while (!q.empty()) {
            Edge e = q.back();
            q.pop_back();
            if (ds.uunion(e.u,e.v)) {
                maxCost+=e.w;
            }
        }
        cout << "Case #" << t << ": " << total-maxCost << endl;
    }
}