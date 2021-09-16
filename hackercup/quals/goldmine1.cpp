#include <bits/stdc++.h>
using namespace std;
/*
Recursion on a TREEEEEE!!!
You can figure out that this is a rooted n-ary tree based on the fact that their are N-1 edges and N nodes
with that it becomes much easier to solve the problem. 
just find the best sums travelling from root

*/
char NEUTRAL = '.';
const int INF = 1e9;
const string IMP = "Impossible";

int dfs(int parent, int node, int sum, vector<vector<int>>& graph, vector<int>& C) {
    if (graph[node].size()==1) {
        return sum;
    }
    int ans = 0;
    for (auto &child : graph[node]) {
        if (child==parent) {continue;}
        sum+=C[child];
        ans = max(ans, dfs(node, child, sum, graph, C));
        sum -= C[child]; // backtrack part
    }
    return ans;
}

int main() {
    int T;
    // freopen("inputs/gold_mine_chapter_1_validation_input.txt", "r", stdin);
    // freopen("outputs/gold_mine_chapter_1_validation_output.txt", "w", stdout);
    freopen("inputs/gold_mine_chapter_1_input.txt", "r", stdin);
    freopen("outputs/gold_mine_chapter_1_output.txt", "w", stdout);
    cin>>T;
    for (int t = 1;t<=T;t++) {
        int N, c, a, b;
        cin>>N;
        vector<int> C(N);
        for (int i = 0;i<N;i++) {
            cin>>c;
            C[i]=c;
        }
        vector<vector<int>> graph(N);
        for (int i = 0;i<N-1;i++) {
            cin>>a>>b;
            a--; b--;
            graph[a].push_back(b);
            graph[b].push_back(a);
        }
        vector<int> paths;
        for (int i : graph[0]) {
            paths.push_back(dfs(0, i, C[0]+C[i], graph, C));
        }
        sort(paths.rbegin(), paths.rend());
        if (paths.size()>1) {
            printf("Case #%d: %d\n", t, paths[0]+paths[1]-C[0]);
        } else if (paths.size()==1) {
            printf("Case #%d: %d\n", t, paths[0]);
        } else {
            printf("Case #%d: %d\n", t, C[0]);
        }
    }
}