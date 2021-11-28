#include <bits/stdc++.h>
using namespace std;
/*
recursive problem 
*/

// string recurse(int i, vector<string>& vec) {
//     if (vec.size()==0) {
//         return "";
//     }
//     for (int i = 0;i<26;i++) {
//         vector<string> next
//         if (vec[i])
//     }
// }
int main() {
    // freopen("inputs/input.txt","r",stdin);
    // freopen("outputs/output.txt","w",stdout);
    int N,K;
    cin>>N>>K;
    string s;
    vector<string> vec;
    for (int i = 0;i<N;i++) {
        cin>>s;
        vec.push_back(s);
    }
    string res = "";
    sort(vec.begin(),vec.end());
    for (int i = 0;i<K;i++) {
        res+=vec[i];
    }
    cout<<res<<endl;
}

/*
This is a really good solution provided by someone, I will try to understand it when I get a chance.
Looks good, but it uses a bizarre sorting and then DP.  
int main() {
    int n,k;
    cin>>n>>k;
    vector<string>str(n);
    for(int i=0;i<n;i++) cin>>str[i];
    sort(str.begin(),str.end(),[&](string a,string b)->bool{
        return a+b>b+a;
    });
    string s;
    for(int i=0;i<2600;i++) s=s+'z';
    vector<vector<string>>dp(n+1,vector<string>(k+1,s));
    for(int i=0;i<=n;i++) dp[i][0]="";
    for(int i=1;i<=n;i++){
        for(int j=1;j<=min(k,i);j++){
            dp[i][j]=dp[i-1][j];
            string temp=dp[i-1][j-1]+str[i-1];
            dp[i][j]=min(dp[i][j],str[i-1]+dp[i-1][j-1]);
        }
    }
    cout<<dp[n][k]<<endl;
    return 0;
*/