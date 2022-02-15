#include <bits/stdc++.h>
using namespace std;

template <class T>
string toString(vector<T>& vec) {
    stringstream res;
    copy(vec.begin(), vec.end(), ostream_iterator<T>(res, " "));
    return res.str().c_str(); 
}

int main() {
    int T, N, M, A, B, s;
    cin>>T;
    for (int t=1;t<=T;t++) {
        cin>>N>>M; // number of problem sets, number of students
        map<int,int> cnts; // (difficulty -> frequency)
        for (int i = 0;i<N;i++) {
            cin>>A>>B;
            for (int j=A;j<=B;j++) {
                cnts[j]++;
            }
        }
        vector<int> S(M, 0);
        for (int i = 0;i<M;i++) {
            cin>>s;
            auto it = cnts.lower_bound(s);
            int x = it->first;
            it--;
            int y = it->first;
            if (cnts.find(y)!=cnts.end() && abs(s-y)<=abs(s-x)) { 
                swap(x,y); // x always answer
            } else if (cnts.find(x)==cnts.end()) {
                swap(x,y);
            }
            S[i]=x;
            cnts[x]--;
            if (cnts[x]==0) {
                cnts.erase(x); // remove from map
            }
        }
        printf("Case #%d: %s\n", t, toString(S).c_str());
    }
}