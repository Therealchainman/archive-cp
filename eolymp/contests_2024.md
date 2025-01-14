# Eolymp

# Weekend Practice #3

## Math Test

```cpp
#include <bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'

int q, x, y, k;

void solve() {
   cin >> x >> y >> k;
   int d = x / y;
   int ans = 0;
   if (k <= d) {
      ans = x - k * y;
   } else {
      k -= d;
      if (k % 2 == 0) ans = x % y;
      else ans = y - x % y;
   }
   cout << ans << endl;
}

signed main() {
   cin >> q;
   while (q--) {
      solve();
   }
   return 0;
}
```

## Party gathering

```cpp
const int DEFAULT = -1;
int N;
vector<int> arr, diff, nxt, prv;
vector<bool> marked;
priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> minheap;

void erase(int x) {
    prv[nxt[x]] = prv[x];
    nxt[prv[x]] = nxt[x];
}

void solve() {
    cin >> N;
    arr.resize(N + 1);
    for (int i = 1; i <= N; i++) {
        cin >> arr[i];
    }
    sort(arr.begin() + 1, arr.end());
    diff.assign(N + 1, DEFAULT);
    nxt.assign(N + 1, N);
    prv.assign(N + 1, 0);
    for (int i = 1; i < N; i++) {
        nxt[i] = i + 1;
        prv[i] = i - 1;
        diff[i] = arr[i + 1] - arr[i];
        minheap.emplace(diff[i], i);
    }
    int ans = N;
    for (int k = 1; k <= N; k++) {
        // cout << "k: " << k << endl;
//         int cnt = 0;
        while (!minheap.empty() && minheap.top().first < k) {
            // cnt++;
            // if (cnt > 5) break;
            const auto [d, i] = minheap.top();
            minheap.pop();
            // cout << d << " " << i << " " << diff[i] << endl;
            if (d != diff[i]) continue;
            ans--;
            int nxt_i = nxt[i];
            int prv_i = prv[i];
            // cout << "nxt_i: " << nxt_i << endl;
            // cout << "prv_i: " << prv_i << endl;
            if (nxt_i == N) {
                // erase i
                erase(i);
                if (prv_i != 0) {
                    diff[prv_i] += diff[i];
                    minheap.emplace(diff[prv_i], prv_i);
                }
                diff[i] = DEFAULT;
            } else {
                // erase i + 1
                erase(nxt_i);
                diff[i] += diff[nxt_i];
                diff[nxt_i] = DEFAULT;
                // cout << "diff[i]" << diff[i] << endl;
                minheap.emplace(diff[i], i);
            }
        }
        // cout << "ans: " << ans << endl;
        cout << ans << " ";
    }
    cout << endl;
}

signed main() {
    solve();
    return 0;
}
```

## Sakurako and Old Dictionary

```cpp
#include <bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'

int N, K;
vector<string> arr;
vector<pair<string, int>> rev_arr;
string word;
vector<bool> vis;

void solve() {
    cin >> N >> K;
    arr.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> arr[i];
        string word = "";
        for (int j = arr[i].size() - 1; j >= 0; j--) {
            word += arr[i][j];
        }
        if (word < arr[i]) {
            rev_arr.emplace_back(word, i);
        }
    }
    sort(rev_arr.begin(), rev_arr.end());
    vector<string> result;
    vis.assign(N, false);
    for (int k = 0; k < K && k < rev_arr.size(); k++) {
        const auto [w, i] = rev_arr[k];
        vis[i] = true;
        result.push_back(w);
    }
    for (int i = 0; i < N; i++) {
        if (vis[i]) continue;
        result.push_back(arr[i]);
    }
    sort(result.begin(), result.end());
    string ans = "";
    for (const string &w : result) {
        ans += w;
    }
    cout << ans << endl;
}

signed main() {
    solve();
    return 0;
}
```

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
