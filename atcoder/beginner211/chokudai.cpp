#include <bits/stdc++.h>
using namespace std;
/*
Good dp problem, the kind I'm not good at apparently
*/
const int mod = 1e9+7;
int main() {
    string S;
    cin>>S;
    string T = "chokudai";
    int n = S.size();
    vector<vector<int>> dp(n+1,vector<int>(9,0));
    for (int i=0;i<=n;i++) {
        dp[i][0] = 1;
    }
    for (int i = 1;i<=n;i++) {
        for (int j = 1;j<9;j++) {
            if (S[i-1]==T[j-1]) {
                dp[i][j]=(dp[i-1][j]+dp[i-1][j-1])%mod;
            } else {
                dp[i][j]=dp[i-1][j];
            }
        }
    }
    printf("%d\n", dp[n][8]);
}