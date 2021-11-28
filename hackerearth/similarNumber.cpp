#include <bits/stdc++.h>
using namespace std;
// const int MOD = 1e9+7;
// const int N = 1e4+2, M = 1e2+2;
// int dp[N][M];
// int N,K;

// int similar(int n, int k, string& s) {
//     if (k==K) return 0;
//     if (n==N) {return 1;}
//     if (dp[n][k]>0) {
//         return dp[n][k];
//     }

//     int ans = s[i]=='0' || s[i]=='9'? 2 : 3;
//     for (int i = n+1;i<N;i++) {
//         if (s[i] )
//     }
//     return dp[n][k] = ans;
// }
// int main() {
//     cin>>N>>K;
//     string number;
//     cin>>number;
//     similar(0, 0, number);
//     return 0;
// }

typedef long long ll;

constexpr int MAX_N = 1e4 + 14, MAX_K = 114, MOD = 1e9 + 7;

int n, k, dp[MAX_N][MAX_K];

int main() {
    ios::sync_with_stdio(0), cin.tie(0);
    string s;
    cin >> n >> k >> s;
    dp[0][0] = 1;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j <= k; ++j) {
            dp[i + 1][j] = dp[i][j];
            printf("before dp=%d\n", dp[i+1][j]);
            if (j)
                dp[i + 1][j] = (dp[i + 1][j] + dp[i][j - 1] * ll(1 + (s[i] != '0' && s[i] != '9'))) % MOD;
            printf("i+1=%d,j=%d,dp[i+1][j]=%d\n", i+1,j,dp[i+1][j]);
        }
    cout << accumulate(dp[n] + 1, dp[n] + k + 1, 0ll) % MOD << '\n';
}