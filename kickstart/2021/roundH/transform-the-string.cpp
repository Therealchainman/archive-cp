#include <bits/stdc++.h>
using namespace std;
/*
Iterate through using a vector data structure, and modular arithmetic. 
*/
const int INF = 1e9;
int main() {
    int T;
    string S, F;
    cin>>T;
    for (int t=1;t<=T;t++) {
        cin>>S>>F;
        vector<int> vis(26,0);
        for (char &c : F) {
            vis[c-'a']=1;
        }
        int cnt = 0;
        for (char &c : S) {
            for (int i = 0;i<=13;i++) {
                if (vis[(c-'a'+i)%26] || vis[(c-'a'-i+26)%26]) {
                    cnt += i;
                    break;
                }
            }
        }
        printf("Case #%d: %d\n", t, cnt);
    }
}