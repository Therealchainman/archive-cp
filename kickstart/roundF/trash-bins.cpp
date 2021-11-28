#include <bits/stdc++.h>
using namespace std;
/*
prefix/suffix array for count of distance
Initialize prefix and suffix to be INF until you reach the first 1 from respective start points. 
Then you create the suffix vector,
Then iterate through the elements in S, and update the prefix 
and increment the distance based on the minimum of the prefix and suffix.  



*/
const int INF = 1e9;
int main() {
    int T, N;
    string S;
    cin>>T;
    for (int t=1;t<=T;t++) {
        cin>>N;
        cin>>S;
        vector<long long> suffix(N+1,INF);
        bool found = false;
        long long dist = 0, prefix=INF;  // the total distance all the neighbors must travel to trash bins. 
        for (int i = N-1;i>=0;i--) {
            found |= (S[i]=='1');
            if (found) {
                suffix[i] = S[i]=='0' ? suffix[i+1] + 1 : 0LL;
            }
        }
        found = false;
        for (int i = 0;i<N;i++) {
            found |= (S[i]=='1');
            if (found) {
                prefix = S[i]=='0' ? prefix+1 : 0LL;
            }
            dist += min(prefix, suffix[i]);
        }
        printf("Case #%d: %lld\n", t, dist);
    }
}


/*
I couldn't find the edge case it is failing on it seems.  


*/