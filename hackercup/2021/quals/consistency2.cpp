#include <bits/stdc++.h>
using namespace std;
const int INF = 1e9;
/*
1) Build a directed graph where a replacement node has an edge to a replaced node
2) Store the count of all 26 characters
3) Iterate through 26 characters as candidates for making string consistent
4) BFS to find if this candidate character can replace all characters in the string, to replace itself it is a cost of 0.
5) update the answer with the minimum cost only if the number of characters replaced is equal to the length of the string.
*/
int main() {
    int T;
    // freopen("inputs/consistency_chapter_2_validation_input.txt", "r", stdin);
    // freopen("outputs/consistency_chapter_2_validation_output.txt", "w", stdout);
    freopen("inputs/consistency_chapter_2_input.txt", "r", stdin);
    freopen("outputs/consistency_chapter_2_output.txt", "w", stdout);
    cin>>T;
    for (int t = 1;t<=T;t++) {
        string S;
        int K;
        cin>>S>>K;
        char A, B;
        int n = S.size();
        unordered_map<char,int> counts;
        unordered_map<char,unordered_set<char>> graph;
        while (K--) {
            cin>>A>>B;
            graph[B].insert(A);
        }
        for (char &c : S) {
            counts[c]++;
        }
        int ans = INF;
        /*
        I'm checking if I can change all characters to c, 
        I look through ch characters and see if they exist in the string
        */
        for (char c='A';c<='Z';c++) {
            int cost = 0, cnt = 0;
            queue<pair<char,int>> q;
            q.emplace(c, 0);
            unordered_set<char> visited;
            visited.insert(c);
            while (!q.empty()) {
                char ch;
                int depth;
                tie(ch, depth) = q.front();
                q.pop();
                for (auto &nc : graph[ch]) {
                    if (visited.count(nc)==0) {
                        q.emplace(nc, depth+1);
                        visited.insert(nc);
                    }

                }
                cnt += counts[ch];
                if (ch==c) {continue;}
                cost += depth*counts[ch];
            }
            if (cnt==n) {
                ans = min(ans, cost);
            }
        }
        ans = ans<INF ? ans : -1;
        printf("Case #%d: %d\n", t, ans);
    }
}