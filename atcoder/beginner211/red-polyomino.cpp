#include <bits/stdc++.h>
using namespace std;
/*
Let's try again later
*/
const int mod = 1e9+7;
vector<pair<int,int>> DIRECTIONS = {{1,0},{-1,0},{0,1},{0,-1}};
int dfs(int row, int col, int k, vector<vector<char>>& grid, vector<vector<int>>& seen) {
    printf("row=%d,col=%d,k=%d\n",row,col,k);
    if (k==0) {
        return 1;
    }
    int N = grid.size();
    auto check = [&](const int i, const int j) {
        return i>=0 && i<N && j>=0 && j<N && seen[i][j]==0 && grid[i][j]=='.';
    };
    int ans = 0;
    for (auto dir : DIRECTIONS) {
        int dr = dir.first, dc = dir.second;
        int nr = row+dr, nc = col+dc;
        if (check(nr,nc)) {
            seen[nr][nc]=1;
            ans += dfs(nr,nc,k-1,grid,seen);
        }
    }
    return ans;
}
int main() {
    int N, K;
    string S;
    cin>>N>>K;
    vector<vector<char>> grid(N,vector<char>(N,'#'));
    for (int i = 0;i<N;i++) {
        cin>>S;
        for (int j = 0;j<N;j++) {
            grid[i][j]=S[j];
        }
    }
    int ans = 0;
    vector<vector<int>> vis(N,vector<int>(N,0)); 
    for (int r=0;r<N;r++) {
        for (int c = 0;c<N;c++) {
            if (grid[r][c]=='.') {
                vector<vector<int>> seen = vis;
                if (r==1 && c==0) {
                    ans += dfs(r,c,K-1,grid,seen);
                    printf("r=%d,c=%d,ans=%d\n",r,c,ans);
                }

            }
        }
    }
    printf("%d\n", ans);
}