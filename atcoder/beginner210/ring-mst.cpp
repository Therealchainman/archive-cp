#include <bits/stdc++.h>
using namespace std;
/*
approach1:  Kruskal's algo with minimum spanning tree and so on will time out because
the (ElogE) but E=N*M in this problem, there are too many edges.  So to improve you can sort based on costs
and iterate through the m operations taking lowest cost first. 
But there is some confusing magic to be done to actually solve with 
modular arithmetic and gcd that I can't find a good explanation anywhere.  
 

How can I understand this problem? where to start?

Make pair of Ai and Cost. Sort on Cost in ascending order, 
if cost is same compare gcd(Ai,N). Calculate final answer by iterating on 
the vector pair and adding maximum possible edges with the current cost (edges = N — gcd(N−Ai)). 
Update N = gcd after every iteration. If we reach n = 1, i.e we made N-1 edges, break.
*/
 
int main() {
    int N, M;
    cin>>N>>M;
    vector<pair<long long, long long>> A(M);
    for (int i = 0;i<M;i++) {
        cin>>A[i].second>>A[i].first;
    }
    sort(A.begin(),A.end());
    long long g = N, gb = N, minCost = 0; // WTF is g and gb lol
    for (int i = 0;i<M;i++) {
        g = __gcd(g, A[i].second);
        minCost += (A[i].first*(gb-g));
        gb = g;
    }
    if (g>1) {
        printf("%d\n", -1);
    } else {
        printf("%lld\n", minCost);
    }
}
