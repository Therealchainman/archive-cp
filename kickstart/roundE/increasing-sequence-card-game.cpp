#include <bits/stdc++.h>
using namespace std;

int main() {
    long long T, N;
    cin>>T;
    for (int t = 1;t<=T;t++) {
        cin>>N;
        double E = 0;
        if (N<1000) {
            for (int i = 1;i<=N;i++) {
                E += 1.0/i;
            }
        } else {
            const double gamma = 0.5772156649;
            E += gamma + log(N)+ 1.0/(2*N) + 1.0/(12*N*N);
        }
        printf("Case #%d: %f\n",t,E);
    }
}