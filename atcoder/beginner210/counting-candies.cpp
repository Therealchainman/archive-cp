#include <bits/stdc++.h>
using namespace std;
/*
Sliding window algorithm
K=3
1,2,3,4
*/
int main() {
    int N, K,c, ans=0;
    vector<int> C;
    cin>>N>>K;
    for (int i = 0;i<N;i++) {
        cin>>c;
        C.push_back(c);
    }
    unordered_map<int,int> counts;
    for (int i = 0;i<N;i++) {
        counts[C[i]]++;
        if (i<K-1) {
            continue;
        }
        ans = max(ans, (int)counts.size());
        counts[C[i-K+1]]--;
        if (counts[C[i-K+1]]==0) {
            counts.erase(C[i-K+1]);
        }
    }
    printf("%d\n",ans);
}


int main() {
    int N,A,X,Y;
    cin>>N>>A>>X>>Y;
    int ans = 0;
    ans += (min(N,A)*X);
    N-=A;
    ans += (N*Y);
    cout<<ans<<endl;

}