#include <bits/stdc++.h>
using namespace std;
const int INF = 1e9;
/*
Use dp[count of characters typed][current hand using]=F(W[0:i])=dp[i][j]
Dynamic programming problem where we want to find the minimum number of times Timmy switch hands. 
*/
int main() {
    int T;
    // freopen("inputs/weak_typing_chapter_1_validation_input.txt", "r", stdin);
    // freopen("outputs/weak_typing_chapter_1_validation_output.txt", "w", stdout);
    freopen("inputs/weak_typing_chapter_1_input.txt", "r", stdin);
    freopen("outputs/weak_typing_chapter_1_output.txt", "w", stdout);
    cin>>T;
    for (int t = 1;t<=T;t++) {
        int N;
        string W;
        cin>>N>>W;
        int left = 0, right = 0; // initialize the start with left and right hand at 0
        for (int i = 0;i<N;i++) {
            // suppose I'm using my left hand currently
            // TODO: explore using inline &int for setting min and maximum later (more optimum? benchmark?)
            int pleft = left;
            left = W[i]=='O' ? INF : min(left, right+1);
            // suppose I'm using my right hand currently
            right = W[i]=='X' ? INF : min(right, pleft+1);
        }
        int ans = min(left, right);
        printf("Case #%d: %d\n", t, ans);
    }
}

