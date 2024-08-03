# Aho-Corasick Algorithm and Data Structure


## Implementation for solving dynamic programming

For this problem you need to add two variables in the Vertex struct, it needs to know the cost and depth of this vertex.  

Aho-Corasick data structure is similar to a trie, but it has transitions, suffix_links and output_links.  These are three kind of edges in the graph, that all represent something that is useful for problems. 

The transitions are the edges that go from one vertex to another, and they are the edges that represent the characters in the string, but are also like the transition in an automaton.  The suffix_link is the edge that goes to the longest suffix of the current vertex that is also in the trie.  The output_link is the edge that goes to the longest suffix of the current vertex that is also a leaf in the trie.




This solution works because it follows the output links to find all possible suffixes that are in the dictionary for the current prefix.  Cause you precompute every prefix as you move through the text.  And it minimizes the total cost to reach the current prefix. 



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
    for (int state = 0, next_state = 0; state < trie.size(); state++) {
        int v = queue[state];
        int u = trie[v].suffix_link;
        if (trie[u].is_leaf) trie[v].output_link = u;
        else trie[v].output_link = trie[u].output_link;
        for (int c = 0; c < K; c++) {
            if (trie[v].transition[c] != 0) {
                trie[trie[v].transition[c]].suffix_link = v ? trie[u].transition[c] : 0;
                queue[++next_state] = trie[v].transition[c];
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