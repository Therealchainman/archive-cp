#include <bits/stdc++.h>
using namespace std;
#define x first
#define y second

#define x first
#define y second
const int INF = 1e9;
int dist[50][50][50][50], N;
vector<vector<vector<vector<int>>>> dp;
vector<pair<int, int>> pos;

void min_distance(int kx, int ky) {
    int moves[8][2] = {{2, 1}, {1, 2}, {2, -1}, {1, -2}, {-2, 1}, {-1, 2}, {-2, -1}, {-1, -2}};
    queue<pair<int, int>> q;
    q.emplace(kx, ky);
    dist[kx][ky][kx][ky] = 0;
    while (!q.empty()) {
        auto [x, y] = q.front();
        q.pop();
        for (int i = 0; i < 8; ++i) {
            int nx = x + moves[i][0];
            int ny = y + moves[i][1];
            if (nx >= 0 && nx < 50 && ny >= 0 && ny < 50 && dist[kx][ky][nx][ny] == INF) {
                dist[kx][ky][nx][ny] = dist[kx][ky][x][y] + 1;
                q.emplace(nx, ny);
            }
        }
    }
}

int minimax(int idx, int depth, int mask, bool ismax)
{
    if (depth == N) return 0;
    if (dp[idx][depth][mask][ismax] != -1) return dp[idx][depth][mask][ismax];
    if (ismax) {
        int ans = 0;
        for (int i = 0; i < N; ++i) {
            if ((mask >> i) & 1) continue;
            ans = max(ans, dist[x][y][pos[i].x][pos[i].y] + minimax(i, depth + 1, mask | (1 << i), ismax ^ 1));
        }
        return dp[idx][depth][mask][ismax] = ans;
    } else {
        int ans = INF;
        for (int i = 0; i < N; ++i) {
            if ((mask >> i) & 1) continue;
            ans = min(ans, dist[x][y][pos[i].x][pos[i].y] + minimax(i, depth + 1, mask | (1 << i), ismax ^ 1));
        }
        return dp[idx][depth][mask][ismax] = ans;
    }
}
 

class Solution {
public:
    int maxMoves(int kx, int ky, vector<vector<int>>& poss) {
        N = poss.size();
        pos.resize(N);
        for (int i = 0; i < N; i++) {
            pos[i].x = poss[i][0];
            pos[i].y = poss[i][1];
        }
        pos.emplace_back(kx, ky);
        memset(dist, -1, sizeof(dist));
        for (auto [x, y] : pos) {
            min_distance(x, y);
        }
        dp.assign(N, vector<vector<vector<int>>>(N, vector<vector<int>(1 << N, vector<int>(2, -1))));
        // (x, y, depth, mask, turn)
        return minimax(N, 0, 0, true);
        // return 0;
    }
};