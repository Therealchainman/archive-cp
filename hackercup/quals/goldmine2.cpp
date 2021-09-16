#include <bits/stdc++.h>
using namespace std;
/*
dp on a tree!!!

dp[i][j][enter][exit]
enter subtree from anywhere or from root
exit subtree from anywhere or from root

pick a child to use first, 
can use or not use child 0
*/
const int INF = 1e9;
vector<vector<int>> graph;
void recurse(int node, int parent) {
  int cntChildren = 0;
  for (int child : graph[node]) {
    if (child==parent) {continue;}
    recurse(child, node);
    cntChildren++;
  }
  for (int i = 0;i<)
}
int main() {
    int T;
    // freopen("inputs/gold_mine_chapter_2_validation_input.txt", "r", stdin);
    // freopen("outputs/gold_mine_chapter_2_validation_output.txt", "w", stdout);
    freopen("inputs/gold_mine_chapter_2_input.txt", "r", stdin);
    freopen("outputs/gold_mine_chapter_2_output.txt", "w", stdout);
    cin>>T;
    for (int t = 1;t<=T;t++) {
        int N, c, a, b;
        cin>>N;
        vector<int> C(N);
        for (int i = 0;i<N;i++) {
            cin>>c;
            C[i]=c;
        }
        graph.resize(N);
        for (int i = 0;i<N-1;i++) {
            cin>>a>>b;
            a--; b--;
            graph[a].push_back(b);
            graph[b].push_back(a);
        }
        recurse(0, -1);
        int ans;
        printf("Case #%d: %d %d\n", t, ans);
    }
}
// const int LIM = 52;

// inline int& setmax(int &l, int r) {
//   return l = max(l, r);
// }

// int N, K, C[LIM];
// vector<int> adj[LIM];

// // dp[i][j][k] = max value in i's subtree,
// // with j new paths present,
// // and with a free path ongoing from i's parent if k=1.
// int dp[LIM][LIM][2];

// // dp2[i][j][k][c] = max value after first i children of current node,
// // with j new paths present,
// // and with a free path available for use if k=1,
// // and with at least one child connected if c=1.
// int dp2[LIM][LIM][2][2];

// void rec(int i, int parent) {
//   // TODO: a counter for the number of children nodes
//   int nc = 0; 
//   // iterating through all of the children nodes to node i
//   for (int j : adj[i]) {
//     if (j != parent) {
//       rec(j, i); // recursive call with the child node and the parent node. 
//       nc++; // increase the number of children nodes for the current stack item
//     }
//   }
//   printf("number children initially = %d\n", nc);
//   printf("i=%d, parent=%d\n", i+1, parent+1);
//   // Sub-DP.
//   for (int j = 0; j <= nc; j++) {
//     memset(dp2[j], -1, sizeof dp2[j]);
//   }
//   // initialize the dp2[j], for the jth node set all the elements in the dp equal to -1. 
//   dp2[0][0][0][0] = 0;
//   // set the base case here equal to 0 
//   nc = 0; // start another count of number of children 
//   for (int j : adj[i]) {
//     if (j == parent) {
//       continue;
//     }
//     for (int a1 = 0; a1 <= K; a1++) {
//       for (int x : {0, 1}) { // if a free path is available for use
//         for (int y : {0, 1}) { // if at least one child is connected
//           int d = dp2[nc][a1][x][y]; // 
//           printf("G=%d\n", d);
//           printf("set = %d, free path here=%d, at least one child connected = %d\n", a1, x, y);
//           if (d < 0) {
//             continue;
//           }
//           for (int a2 = 0; a2 <= K - a1; a2++) {
//             printf("I think you connect child here, a2=%d\n", a2);
//             // Connect to child.
//             int d2 = dp[j][a2][1];
//             printf("F=%d, node=%d, numpaths=%d, free=%d\n", d2, j+1, a2, 1);
//             if (d2 >= 0) {
//               setmax(dp2[nc + 1][a1 + a2 + (x ? 0 : 1)][1 - x][1], d + d2); // it toggles if it is a free path 
//             }
//             // Don't connect to child.
//             d2 = dp[j][a2][0];
//             if (d2 >= 0) {
//               setmax(dp2[nc + 1][a1 + a2][x][y], d + d2);
//             }
//           }
//         }
//       }
//     }
//     nc++;
//   }
//   // Combine into main DP.
//   memset(dp[i], -1, sizeof dp[i]);
//   for (int o : {0, 1}) {
//     for (int j = 0; j <= K; j++) {
//       for (int x : {0, 1}) {
//         for (int y : {0, 1}) {
//           if (!i && !y) {
//             continue;  // Root must connect to at least one child.
//           }
//           int d = dp2[nc][j][x][y];
//           if (d < 0) {
//             continue;
//           }
//           // Include current node's value?
//           if (o || y) {
//             d += C[i];
//           }
//           // Free path ongoing from parent?
//           setmax(dp[i][j - (o && x ? 1 : 0)][o], d);
//         }
//       }
//     }
//   }
// }

// int solve() {
//   cin >> N >> K;
//   for (int i = 0; i < N; i++) {
//     cin >> C[i];
//     adj[i].clear();
//   }
//   for (int i = 0; i < N - 1; i++) {
//     int a, b;
//     cin >> a >> b;
//     a--, b--;
//     adj[a].push_back(b);
//     adj[b].push_back(a);
//   }
//   // DP
//   rec(0, -1);
//   int ans = C[0];
//   for (int i = 0; i <= K; i++) {
//     setmax(ans, dp[0][i][0]);
//   }
//   return ans;
// }

// int main() {
//   int T;
//   freopen("inputs/gold_mine_chapter_2_input.txt", "r", stdin);
//   freopen("outputs/gold_mine_chapter_2_output.txt", "w", stdout);
//   cin >> T;

//   for (int t = 1; t <= T; t++) {
//     cout << "Case #" << t << ": " << solve() << endl;
//   }
//   return 0;
// }