# minimax

Example of using minimax algorithm, where you max on the first turn and min on the second turn. 
This example includes bitmask dp
This is example of chess board and iterating over the moves from knights. 
This also includes a bfs to precompute the shortest distance to each position. 

```cpp
#define x first
#define y second
const int M = 50, INF = 1e9, P = 15;
const vector<pair<int, int>> MOVES = {{-2, 1}, {-2, -1}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {1, 2}, {-1, 2}};
int N, end_mask;
vector<pair<int, int>> pos;
int dist[M][M][M][M], dp[P + 1][1 << P][2];
bool in_bounds(int x, int y) {
    return x >= 0 && x < M && y >= 0 && y < M;
}
void bfs(int kx, int ky) {
    queue<pair<int, int>> q;
    q.emplace(kx, ky);
    dist[kx][ky][kx][ky] = 0;
    while (!q.empty()) {
        auto [x, y] = q.front();
        q.pop();
        for (auto [dx, dy] : MOVES) {
            int nx = x + dx, ny = y + dy;
            if (in_bounds(nx, ny) && dist[kx][ky][nx][ny] == -1) {
                dist[kx][ky][nx][ny] = dist[kx][ky][x][y] + 1;
                q.emplace(nx, ny);
            }
        }
    }
}
int minimax(int idx, int mask, bool ismax) {
    auto [x, y] = pos[idx];
    if (mask == end_mask) return 0;
    if (dp[idx][mask][ismax] != -1) return dp[idx][mask][ismax];
    int ans = ismax ? 0 : INF;
    for (int i = 0; i < N; i++) {
        if ((mask >> i) & 1) continue;
        if (ismax) {
            ans = max(ans, dist[x][y][pos[i].x][pos[i].y] + minimax(i, mask | (1 << i), ismax ^ 1));
        } else {
            ans = min(ans, dist[x][y][pos[i].x][pos[i].y] + minimax(i, mask | (1 << i), ismax ^ 1));
        }
    }
    return dp[idx][mask][ismax] = ans;
}
class Solution {
public:
    int maxMoves(int kx, int ky, vector<vector<int>>& poss) {
        N = poss.size();
        pos.resize(N);
        for (int i = 0; i < N; i++) {
            pos[i].x = poss[i][0], pos[i].y = poss[i][1];
        }
        pos.emplace_back(kx, ky);
        memset(dist, -1, sizeof(dist));
        for (const auto &[x, y] : pos) {
            bfs(x, y);
        }
        memset(dp, -1, sizeof(dp));
        end_mask = (1 << N) - 1;
        return minimax(N, 0, true); // (idx, mask, ismax)
    }
};