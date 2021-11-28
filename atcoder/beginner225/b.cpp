#include <bits/stdc++.h>
using namespace std;
/*

*/
int main() {
    // freopen("inputs/input.txt","r",stdin);
    // freopen("outputs/output.txt","w",stdout);
    int N,a,b;
    cin>>N;
    N--;
    vector<vector<int>> A;
    for (int i = 0;i<N;i++) {
        cin>>a>>b;
        A.push_back({a,b});
    }
    int canda = A[0][0], candb = A[0][1];
    for (int i = 1;i<N;i++) {
        a = A[i][0], b = A[i][1];
        if (canda!= a && canda!=b) {
            canda=-1;
        }
        if (candb!=a && candb!=b) {
            candb=-1;
        }
        if (canda==-1 && candb==-1) {
            break;
        }
    }
    int ans = canda!=-1 ? canda : candb;
    if (ans==-1) {
        cout<<"No"<<endl;
    } else {
        cout<<"Yes"<<endl;
    }
}
