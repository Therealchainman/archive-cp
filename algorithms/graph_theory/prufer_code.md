# Prufer Code

The Prufer code is a unique sequence associated with a labeled tree. It provides a way to encode the structure of the tree in a compact form. The Prufer code for a tree with N vertices is a sequence of length N - 2.

Repeate N - 2 times:

1. Find the leaf (degree 1 vertex) with the smallest label.
2. Append its neighbor’s label to the code.
3. Remove that leaf from the tree.

What makes it useful:
1. It provides a one-to-one correspondence between labeled trees and sequences of length N - 2, which can be used for counting and generating trees.
2. the degree of each vertex in the tree can be determined from the Prufer code. Specifically, the degree of a vertex is one more than the number of times it appears in the Prufer code.

A Prufer code uniquely determines a labeled tree, and the mapping is reversible. This means that given a Prufer code, we can reconstruct the original tree, and given a tree, we can generate its Prufer code.


## decoding prufer code

```cpp
int N;
vector<int> A, deg;

void solve() {
    cin >> N;
    deg.assign(N + 1, 1);
    A.resize(N - 2);
    for (int i = 0; i + 2 < N; ++i) {
        cin >> A[i];
        deg[A[i]]++;
    }
    set<int> leaves;
    for (int i = 1; i <= N; ++i) {
        if (deg[i] > 1) continue;
        leaves.emplace(i);
    }
    vector<pair<int, int>> edges;
    for (int u : A) {
        int v = *leaves.begin();
        leaves.erase(leaves.begin());
        edges.emplace_back(u, v);
        if (--deg[u] == 1) {
            leaves.emplace(u);
        }
    }
    edges.emplace_back(*leaves.begin(), N);
    for (auto &[u, v] : edges) {
        cout << u << " " << v << endl;
    }
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```