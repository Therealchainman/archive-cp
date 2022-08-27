#include <bits/stdc++.h>
using namespace std;
/*
Find the minimum number of Xs that need to be placed on the board so that you win. 
1 edge case for if the minimum number of Xs is equal to 1
for that edge case you want want to count each point only once, and only if you can fill a row or column at that point
with Xs. 
Else you consider each row or column filled with Xs as unique set of cells for if minmum number of Xs > 1
*/
char NEUTRAL = '.';
const int INF = 1e9;
const string IMP = "Impossible";
int main() {
    int T;
    // freopen("inputs/xs_and_os_validation_input.txt", "r", stdin);
    // freopen("outputs/xs_and_os_validation_output.txt", "w", stdout);
    freopen("inputs/xs_and_os_input.txt", "r", stdin);
    freopen("outputs/xs_and_os_output.txt", "w", stdout);
    cin>>T;
    string s;
    for (int t = 1;t<=T;t++) {
        int N;
        cin>>N;
        vector<string> board(N);
        for (int i = 0;i<N;i++) {
            cin>>s;
            board[i] = s;
        }
        int minCountXs = INF;
        // check the rows
        auto findMinCountXs = [&](const bool row = true) {
            for (int i = 0;i<N;i++) {
                int cntOs = 0, cntEmpty = 0;
                for (int j = 0;j<N;j++) {
                    if (row) {
                        cntOs += board[i][j]=='O';
                        cntEmpty += board[i][j]=='.';
                    } else {
                        cntOs += board[j][i]=='O';
                        cntEmpty += board[j][i]=='.';
                    }
 
                }
                // If there is O in this row can't win
                if (cntOs>0) {
                    continue;
                }
                // update the minimum count of Xs. 
                minCountXs = min(cntEmpty, minCountXs);
                if (minCountXs == 1) {
                    break;
                }
            }
        };
        findMinCountXs();
        findMinCountXs(false);
        auto dfs1 = [&](const int i, const int j) {
            int cntEmpty = 0, cntOs = 0;
            // check row
            for (int r = 0;r<N;r++) {
                cntEmpty += board[r][j]=='.';
                cntOs += board[r][j]=='O';
            }
            if (cntEmpty==1 && cntOs==0) {
                return true;
            }
            cntEmpty = 0, cntOs = 0;
            // check column
            for (int c = 0;c<N;c++) {
                cntEmpty += board[i][c]=='.';
                cntOs += board[i][c]=='O';
            }
            return cntEmpty==1 && cntOs==0;
        };
        auto dfs2 = [&](const bool row = true) {
            int cntWins = 0;
            for (int i = 0;i<N;i++) {
                int cntOs = 0, cntEmpty = 0;
                for (int j = 0;j<N;j++) {
                    if (row) {
                        if (board[i][j]=='O') {
                                cntOs++;
                                break;
                            } else if (board[i][j]=='.') {
                                cntEmpty++;
                            }
                    } else {
                        if (board[j][i]=='O') {
                            cntOs++;
                            break;
                        } else if (board[j][i]=='.') {
                            cntEmpty++;
                        }
                    }

                }
                // If there is O in this row can't win
                if (cntOs>0) {
                    continue;
                }
                // update the count of wins if this had same count of empty as minimum Xs found to win.  
                cntWins += cntEmpty==minCountXs;
            }
            return cntWins;
        };
        int cntCollections = 0;
        if (minCountXs==1) {
            for (int i = 0;i<N;i++) {
                for (int j = 0;j<N;j++) {
                    cntCollections += (board[i][j]=='.' && dfs1(i,j));
                }
            }
            printf("Case #%d: %d %d\n", t, minCountXs, cntCollections);
        } else if (minCountXs < INF) {
            cntCollections = dfs2() + dfs2(false);
            printf("Case #%d: %d %d\n", t, minCountXs, cntCollections);
        } else {
            printf("Case #%d: %s\n", t, IMP.c_str());
        }
    }
}