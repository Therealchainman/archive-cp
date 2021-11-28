#include <bits/stdc++.h>
using namespace std;
/*
dfs
*/

int main() {
    // freopen("inputs/input.txt","r",stdin);
    // freopen("outputs/output.txt","w",stdout);
    int N, X, a;
    cin >> N >> X;
    vector<int> A(N+1,0);
    for(int i=1; i<=N; i++) {
        cin >> a;
        A[i] = a;
    }
    vector<bool> visited(N+1,false);
    int cnt = 0;
    for(int i=X;!visited[i];i=A[i]) {
        visited[i] = true;
        cnt++;
    }
    cout<<cnt<<endl;
}