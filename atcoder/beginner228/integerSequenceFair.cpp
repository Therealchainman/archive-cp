#include <bits/stdc++.h>
using namespace std;
/*
binary exponentiation with use of fermat's little theorem, taking a^(b^c)) type problem but where a>mod
*/
long long MOD = 998244353;

// a^b
long long power(long long a, long long b, long long mod) {
    long long result = 1;
    a%=mod;
    while (b>0) {
        if (b%2 == 1) {
            result = (result*a)%mod;
        }
        a = (a*a)%mod;
        b/=2;
    }
    return result;
}
int main() {
    // freopen("inputs/input.txt","r",stdin);
    // freopen("outputs/output.txt","w",stdout);
    long long N,K,M;
    cin >> N >> K >> M;
    if (M%MOD==0) {
        cout<<0<<endl;
        return 0;
    }
    long long res = power(K,N, MOD-1);
    cout<<power(M, res, MOD)<<endl;
    return 0;
}