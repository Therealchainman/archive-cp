#include <bits/stdc++.h>
using namespace std;
/*
greedy array type problem.  
*/

int main() {
    // freopen("inputs/input.txt","r",stdin);
    // freopen("outputs/output.txt","w",stdout);
    int N, K, a,b,c;
    cin >> N >> K;
    K--;
    vector<int> P(N); // points on the third day
    for (int i = 0; i < N; i++) {
        cin>>a>>b>>c;
        P[i] = a+b+c;
    }
    vector<int> q = P;
    sort(q.rbegin(), q.rend());
    for (int i = 0;i<N;i++) {
        cout<<(P[i]+300>=q[K] ? "Yes" : "No")<<endl;
    }
}