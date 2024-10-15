# Eolymp

# Eolymp Cup 2

## Finally a Problem Without a Legend

### Solution 1:  bit manipulation, floyd warshall, all pairs shortest path, auxiliary vertex

1. Create auxiliary vertex to represent bits and connect them to nodes of interest
1. This involves a neat trick to determine if two nodes will be fully covered. 

Tells you which bits in bi are not covered by ai, all the bits that are still set are the bits not covered. 

The condition states that we do not want all the bits of the ai | aj cannot cover all the bits set in bi & bj. 

If di & dj != 0, it means that ai | aj cannot cover bi & bj, since the problematic bits in both di and dj overlap.  This overlap implies that there is a shared bit which is set in bi & bj, but neither ai nor aj has it set.  Thus, an edge must exist between i and j. 

```cpp
const int BITS = 45, INF = 1e16;
int N, Q;
vector<int> A, B, C, D;
int dist[BITS][BITS];
vector<vector<int>> dp;

void floyd_warshall(int n) {
    // floyd warshall, all pairs shortest path
    for (int k = 0; k < n; k++) {  // Intermediate vertex
        for (int i = 0; i < n; i++) {  // Source vertex
            for (int j = 0; j < n; j++) {  // Destination vertex
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
            }
        }
    }
}

bool bit(int val, int i) {
    return (val >> i) & 1LL;
}

void solve() {
    cin >> N;
    A.resize(N);
    B.resize(N);
    C.resize(N);
    D.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    for (int i = 0; i < N; i++) {
        cin >> B[i];
    }
    for (int i = 0; i < N; i++) {
        cin >> C[i];
    }
    for (int i = 0; i < N; i++) {
        D[i] = B[i] & (~A[i]);
    }
    for (int i = 0; i < BITS; i++) {
        fill(dist[i], dist[i] + BITS, INF);
    }
    for (int i = 0; i < BITS; i++) {
        dist[i][i] = 0;
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < BITS; j++) {
            if (!bit(D[i], j)) continue;
            for (int k = 0; k < BITS; k++) {
                if (!bit(D[i], k)) continue;
                dist[j][k] = min(dist[j][k], 2LL * C[i]);
            }
        }
    }
    floyd_warshall(BITS);
    // calculate the shortest distance from each bit node to every normal node.
    dp.assign(BITS, vector<int>(N, INF));
    for (int i = 0; i < BITS; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < BITS; k++) {
                if (!bit(D[j], k)) continue;
                dp[i][j] = min(dp[i][j], dist[i][k] + C[j]);
            }
        }
    }
    cin >> Q;
    while (Q--) {
        int u, v;
        cin >> u >> v;
        u--; v--;
        int ans = INF;
        for (int i = 0; i < BITS; i++) {
            if (!bit(D[u], i)) continue;
            // cout << i << " " << dp[i][v] << endl;
            ans = min(ans, dp[i][v] + C[u]);
        }
        cout << ans << endl;
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
