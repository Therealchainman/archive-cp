#include <bits/stdc++.h>
using namespace std;
const long long INF = INT_MAX;
int main() {
    int H, W,c;
    long long C;
    cin>>H>>W>>C;
    vector<vector<long long>> grid(H+1,vector<long long>(W+1,0)), dp(H+1,vector<long long>(W+1,INF)), X(H+1,vector<long long>(W+1,INF));
    for (int i = 1;i<=H;i++) {
        for (int j = 1;j<=W;j++) {
            cin>>c;
            grid[i][j]=c;
        }
    }
    long long ans = INF;
    for (int rep = 0;rep<2;rep++) {
        for (int i = 1;i<=H;i++) {
            for (int j=1;j<=W;j++) {
                dp[i][j]=min({grid[i][j],dp[i-1][j]+C,dp[i][j-1]+C});
            }
        }
        for (int i = 1;i<=H;i++) {
            for (int j=1;j<=W;j++) {
                X[i][j]=min(dp[i-1][j],dp[i][j-1]) + C+grid[i][j];
                ans = min(ans,X[i][j]);
            }
        }
        reverse(grid.begin(),grid.end());
    }
    cout<<ans<<endl;
}