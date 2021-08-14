#include <bits/stdc++.h>
using namespace std;

int main() {
    int T,n,a,b;
    cin>>T;
    while (T--) {
        cin>>n;
        vector<int> A, B;
        for (int i = 0;i<n;i++) {
            cin>>a;
            A.push_back(a);
        }
        for (int i = 0;i<n;i++) {
            cin>>b;
            B.push_back(b);
        }
        sort(A.begin(),A.end());
        sort(B.begin(),B.end());
        vector<int> PA(n+1,0);
        deque<int> PB(n+1,0);
        for (int i = 0;i<n;i++) {
            PA[i+1]=PA[i]+A[i];
        }
        for (int i = 0;i<n;i++) {
            PB[i+1]=PB[i]+B[i];
        }
        int i;
        for (i = 0;PA.end()[-1]-PA[(i+n)/4]<PB.end()[-1]-PB[(i+n)/4];i++) {
            PA.push_back(PA.end()[-1]+100);
            PB.push_front(0);
        }
        printf("%d\n",i);
    }
}