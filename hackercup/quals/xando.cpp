#include <bits/stdc++.h>
using namespace std;
/*

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
        vector<vector<char>> board(N,vector<char>(N));
        for (int i = 0;i<N;i++) {
            cin>>s;
            for (int j = 0;j<N;j++) {
                board[i][j]=s[j];
            }
        }
        vector<bool> seen(N*N);
        int cnt = INF, ways = 0;
        // check the rows
        for (int i = 0;i<N;i++) {
            int no = 0, nx = 0, ne = 0, start = 0;
            for (int j = 0;j<N;j++) {
                if (board[i][j]=='X') {
                    nx++;
                } else if (board[i][j]=='O') {
                    no++;
                    break;
                } else {
                    ne++;
                    start = N*i+j;
                }
            }
            // If there is O in this row can't win
            if (no>0) {
                continue;
            }
            // if we only need one X in another place, check we haven't used this cell
            if (ne==1 && ne==cnt) {
                ways = seen[start] ? ways : ways+1;
                continue;
            }
            // no change to number required Xs.
            if (ne==cnt ) {
                ways++;
            }
            if (ne==1) {
                seen[start]=true;
                cnt = ne;
                ways = 1;
            } else if (ne<cnt) {
                cnt = ne;
                ways = 1;
            }
        }
        // check the columns
        for (int i = 0;i<N;i++) {
            int no = 0, nx = 0, ne = 0, start = 0;
            for (int j = 0;j<N;j++) {
                if (board[j][i]=='X') {
                    nx++;
                } else if (board[j][i]=='O') {
                    no++;
                    break;
                } else {
                    ne++;
                    start = N*j+i;
                }
            }
            // If there is O in this col can't win
            if (no>0) {
                continue;
            }
            // if we only need one X in another place, check we haven't used this cell
            if (ne==1 && ne==cnt) {
                ways = seen[start] ? ways : ways+1;
                continue;
            }
            // no change to number required Xs.
            if (ne==cnt ) {
                ways++;
            }
            if (ne==1) {
                seen[start]=true;
                cnt = ne;
                ways = 1;
            } else if (ne<cnt) {
                cnt = ne;
                ways = 1;
            }
        }
        if (ways==0) {
            printf("Case #%d: %s\n", t, IMP.c_str());
        } else {
            printf("Case #%d: %d %d\n", t, cnt, ways);
        }
    }
}