#include <bits/stdc++.h>
using namespace std;
/*
BFS algorithm 
*/
int main() {
    int T, N;
    string input;
    cin>>T;
    while (T--) {
        cin>>N;
        vector<string> grid;
        bool found = false;
        for (int i = 0;i<2;i++) {
            cin>>input;
            grid.push_back(input);
        }
        queue<pair<int,int>> q;
        q.emplace(0,0);
        auto check = [&](const int i, const int j) {
            return i>=0 && i<2 && j>=0 && j<N && grid[i][j]=='0';
        };
        while (!q.empty()) {
            int r, c;
            tie(r,c) = q.front();
            q.pop();
            if (r==1 && c==N-1) {
                found = true;
                break;
            }
            for (int dr = -1;dr<2;dr++) {
                for (int dc = -1;dc<2;dc++) {
                    int nr = r+dr, nc = c+dc;
                    if (check(nr,nc)) {
                        grid[nr][nc]='1';
                        q.emplace(nr,nc);
                    }
                }
            }
        }
        if (found) {
            cout<<"YES"<<endl;
        } else {
            cout<<"NO"<<endl;
        }
    }
}