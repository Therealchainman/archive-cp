#include <bits/stdc++.h>
using namespace std;

template <class T>
string toString(vector<T>& vec) {
    stringstream res;
    copy(vec.begin(), vec.end(), ostream_iterator<T>(res, " "));
    return res.str().c_str(); 
}
int mod = 1e9+7;
int power(long long a, int b) {
    long long result = 1;
    while (b>0) {
        if (b%2 == 1) {
            result = (result*a)%mod;
        }
        a = (a*a)%mod;
        b/=2;
    }
    return result;
}
//maybe binary search here
int V(long long x, int P) {
    int lo=0, hi=1e8;
    x%=mod;
    while (lo < hi) {
        int mid = lo+hi+1>>1;
        int p = power(P,mid);
        if (x>=p && x%p==0) {
            lo = mid;
        } else {
            hi = mid-1;
        }
    }
    return lo%mod;
}
int main() {
    int T, P, pos, L, R, N, Q, val, typ, S;
    cin>>T;
    for (int t=1;t<=T;t++) {
        cin>>N>>Q>>P;
        vector<int> arr;
        arr.push_back(0);
        for (int i=0;i<N;i++) {
            cin>>val;
            arr.push_back(val);
        }
        vector<int> res, memo(N+1,-1);
        for (int q=0;q<Q;q++) {
            cin>>typ;
            if (typ==1) {
                cin>>pos>>val;
                arr[pos]=val;
                memo[pos]=-1;
            } else {
                cin>>S>>L>>R;
                int ans = 0;
                for (int i =L;i<=R;i++) {
                    if (memo[i]==-1) {
                        long long fi = power(arr[i],S);
                        long long se = power((arr[i])%P,S);
                        int nxt = (fi-se+mod)%mod;
                        memo[i]=V(nxt, P);
                    }
                    ans+=memo[i];
                }
                res.push_back(ans);
            }
        }
        printf("Case #%d: %s\n", t, toString(res).c_str());
    }
}