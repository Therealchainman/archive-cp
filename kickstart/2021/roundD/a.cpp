#include <bits/stdc++.h>
using namespace std;

int main() {
    int T, g;
    vector<vector<int>> G;
    cin>>T;
    for (int t = 1;t<=T;t++) {
        G = vector<vector<int>>(3,vector<int>(3,0));
        for (int i = 0;i<3;i++) {
            for (int j = 0;j<3;j++) {
                if (i==1) {
                    if (j==0) {
                        cin>>g;
                        G[i][j]=g;
                    } else if (j==2) {
                        cin>>g;
                        G[i][2]=g;
                    }
                } else {
                    cin>>g;
                    G[i][j]=g;
                }
            }
        }
        unordered_map<int,int> series;
        int ans = 0;
        int diff = G[0][1]-G[0][0];
        if (G[0][1]-G[0][0]==G[0][2]-G[0][1]) {
            ans++;
        }
        if (G[2][1]-G[2][0]==G[2][2]-G[2][1]) {
            ans++;
        }
        if (G[1][0]-G[0][0]==G[2][0]-G[1][0]) {
            ans++;
        }
        if (G[1][2]-G[0][2]==G[2][2]-G[1][2]) {
            ans++;
        }
        int a = G[2][2]+G[0][0];
        if (a%2==0) {
            series[a/2]++;
        }
        a = G[1][2]+G[1][0];
        if (a%2==0) {
            series[a/2]++;
        }
        a = G[2][1]+G[0][1];
        if (a%2==0) {
            series[a/2]++;
        }
        a = G[0][2]+G[2][0];
        if (a%2==0) {
            series[a/2]++;
        }
        int mx = 0;
        for (auto s : series) {
            mx = max(mx, s.second);
        }
        ans+=mx;
        printf("Case #%d: %d\n", t, ans);
    }
}