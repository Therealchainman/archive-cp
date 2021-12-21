#include <bits/stdc++.h>
using namespace std;

int main() {
    int T, N, X;
    cin>>T;
    while (T--) {
        cin>>N>>X;
        int sum = 0;
        for (int i = 0;i<N;i++) {
            int a;
            cin>>a;
            sum += a;
        }
        sum = abs(sum);
        cout<<(sum+X-1)/X<<endl;
    }
}