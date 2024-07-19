# Leetcode Weekly Contest 405

## 3213. Construct String with Minimum Cost

### Solution 1:  Aho-Corasick data structure, dynamic programming, bfs, trie, output links

```cpp
const int INF = 1e9, K = 26;
struct Vertex {
    bool is_leaf = false;
    int cost = INF;
    int output_link = 0;
    int suffix_link = 0;
    int depth = 0;
    int transition[K];
    void init() {
        fill(begin(transition), end(transition), 0);
    }
};
vector<Vertex> trie;
void add_string(const string& s, const int cost) {
    int cur = 0, depth = 0;
    for (char ch : s) {
        int c = ch - 'a';
        depth++;
        if (trie[cur].transition[c] == 0) {
            trie[cur].transition[c] = trie.size();
            Vertex v;
            v.init();
            v.depth = depth;
            trie.push_back(v);
        }
        cur = trie[cur].transition[c];
    }
    trie[cur].is_leaf = true;
    trie[cur].cost = min(trie[cur].cost, cost);
}
void push_links() {
    int queue[trie.size()];
    queue[0] = 0;
    int state = 0, next_state = 1;
    while (state < trie.size()) {
        int v = queue[state++];
        int u = trie[v].suffix_link;
        if (trie[u].is_leaf) trie[v].output_link = u;
        else trie[v].output_link = trie[u].output_link;
        for (int c = 0; c < K; c++) {
            if (trie[v].transition[c] != 0) {
                trie[trie[v].transition[c]].suffix_link = v ? trie[u].transition[c] : 0;
                queue[next_state++] = trie[v].transition[c];
            } else {
                trie[v].transition[c] = trie[u].transition[c];
            }
        }
    }
}
class Solution {
public:
    int minimumCost(string target, vector<string>& words, vector<int>& costs) {
        int m = words.size(), n = target.size();
        trie.resize(1);
        trie[0].init();
        for (int i = 0; i < m; i++) {
            add_string(words[i], costs[i]);
        }
        push_links();
        int cur = 0;
        vector<int> dp(n + 1, INF);
        dp[0] = 0;
        cur = 0;
        for (int i = 1; i <= n; i++) {
            cur = trie[cur].transition[target[i - 1] - 'a'];
            if (trie[cur].is_leaf) {
                dp[i] = min(dp[i], dp[i - trie[cur].depth] + trie[cur].cost);
            }
            int output = trie[cur].output_link;
            while (output) {
                dp[i] = min(dp[i], dp[i - trie[output].depth] + trie[output].cost);
                output = trie[output].output_link;
            }
        }
        return dp[n] < INF ? dp[n] : -1;
    }
};
```