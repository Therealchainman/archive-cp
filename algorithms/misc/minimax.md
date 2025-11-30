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
```

## General Sum Game, backward induction

We have a finite rooted tree (or any finite acyclic, directed graph) where two players, P₁ and P₂, move a shared token from the root toward a leaf. They alternate turns—P₁ starts—choosing which child to visit next. Each node u has two non‑negative payoffs, A[u] for P₁ and B[u] for P₂, which are collected exactly once when the node is first visited. The game ends when no children remain (i.e. at a leaf), and each player’s total score is the sum of their payoffs over the visited path.

Because there are no cycles and perfect information, the unique rational outcome is given by backward induction: at each node, the current mover anticipates the opponent’s optimal responses all the way to a leaf, and picks the branch that maximizes their own final score (ties broken by maximizing the opponent’s).

This is like a variation of minimax, but there is no minimizing player, they are both maximizing, bust mostly for themselves. 

Payoffs are general‑sum.
In classic zero‑sum minimax you have a single “value” V(u), and one player maximizes that value while the other minimizes it (equivalently, they maximize –V). Here both players have their distinct payoff and each actively maximizes only their own score.

### Connection to backward induction

This algorithm is a textbook application of backward induction (or subgame‑perfect equilibrium) in extensive‑form games:
Identify subgames (every node is a subgame root).
Solve leaf subgames trivially by payoff at the node.
Propagate backward, at each decision node choosing the action that best serves the mover’s objectives, given the already‑computed outcomes of smaller subgames.

### Example with tree

kuroni and tf are playing a game on tree, at kuroni's turn, they want to maximize their score, and if there is a tie, choose that will maximize tfg's score.
And same for tfg, they alternate between turns.  And Kuroni goest first and starts on the root node.

```cpp
int N;
vector<vector<int>> adj, pref;
vector<vector<pair<int, int>>> dp; // 0: kuroni turn -> (kuroni score, tfg score), 1: tfg turn -> (kuroni score, tfg score)

void dfs(int u, int p = -1) {
    pair<int, int> bestKuroni = {0, 0}, bestTfg = {0, 0}; // kuroni turn, tfg turn, (kuroni score, tfg score)
    for (int v : adj[u]) {
        if (v == p) continue;
        dfs(v, u);
        if (dp[v][1][0] > bestKuroni.first || (dp[v][1].first == bestKuroni.first && dp[v][1].second > bestKuroni.second)) {
            bestKuroni = dp[v][1];
        }
        if (dp[v][0].second > bestTfg.second || (dp[v][0].second == bestTfg.second && dp[v][0].first > bestTfg.first)) {
            bestTfg = dp[v][0];
        }
    }
    dp[u][0] = bestKuroni;
    dp[u][1] = bestTfg;
    for (int i = 0; i < 2; i++) {
        dp[u][i].first += pref[u][0];
        dp[u][i].second += pref[u][1];
    }
}
...
dfs(0);
cout << dp[0][0].first << " " << dp[0][0].second << endl;
```