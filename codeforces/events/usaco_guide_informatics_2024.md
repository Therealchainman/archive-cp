# USACO.Guide Informatics Tournament

## A. Two Way Homework

### Solution 1:  reverse iteration, suffix sum, greedy

```py
def main():
    n = int(input())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    ssum = 0
    for i in reversed(range(n)):
        ssum = min(A[i] + 2 * ssum, B[i] + ssum)
    print(ssum)
if __name__ == '__main__':
    T = 1
    for _ in range(T):
        main()
```

## B. Farmer John's Cities

### Solution 1:  forward dijkstra from source, backwards dijkstra from destination

```py
import math
import heapq
def dijkstra(adj, src):
    N = len(adj)
    min_heap = [(0, src)]
    dist = [math.inf] * N
    while min_heap:
        cost, u = heapq.heappop(min_heap)
        if cost >= dist[u]: continue
        dist[u] = cost
        for v, w in adj[u]:
            if cost + w < dist[v]: heapq.heappush(min_heap, (cost + w, v))
    return dist

def main():
    N, M, K, s, t = map(int, input().split())
    s -= 1; t -= 1
    adj = [[] for _ in range(N)]
    radj = [[] for _ in range(N)]
    for _ in range(M):
        u, v, w = map(int, input().split())
        u -= 1; v -= 1
        adj[u].append((v, w))
        radj[v].append((u, w))
    dist_s = dijkstra(adj, s) # distance from s to all nodes
    dist_t = dijkstra(radj, t) # distance from t to all nodes
    ans = dist_s[t]
    for _ in range(K):
        u, v, w = map(int, input().split())
        u -= 1; v -= 1
        ans = min(ans, dist_s[u] + w + dist_t[v])
    print(ans)

if __name__ == '__main__':
    T = 1
    for _ in range(T):
        main()
```

## C. Gardening is Hard

### Solution 1: 

fix row, and iterate backwards, adding this 2xcolumn each time
then fixing that start point how many can you end at

i iterated over the row
and then iterated over the column as right endpoint
and then counted all left endpoints

```py
#define print_debug true

#include "bits/stdc++.h"

#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>

const long long MOD = 1000000007;

using namespace __gnu_pbds;
using namespace std;

typedef tree<int, null_type, less<int>, rb_tree_tag,
             tree_order_statistics_node_update>
    indexed_set;

typedef long long ll;

template<typename T> istream& operator>>(istream& in, vector<T>& a) {for(auto &x : a) in >> x; return in;};
template<typename T1, typename T2> istream& operator>>(istream& in, pair<T1, T2>& x) {return in >> x.first >> x.second;}

template<typename T1, typename T2> ostream& operator<<(ostream& out, const pair<T1, T2>& x) {return out << x.first << ' ' << x.second;}
template<typename T> ostream& operator<<(ostream& out, vector<T>& a) {for(auto &x : a) out << x << ' '; return out;};
template<typename T> ostream& operator<<(ostream& out, vector<vector<T>>& a) {for(auto &x : a) out << x << '\n'; return out;};
template<typename T1, typename T2> ostream& operator<<(ostream& out, vector<pair<T1, T2>>& a) {for(auto &x : a) out << x << '\n'; return out;};

void fileIO(string name) {
    freopen((name + ".in").c_str(), "r", stdin);
    freopen((name + ".out").c_str(), "w", stdout);
}

int main() {
    iostream::sync_with_stdio(false);
    cin.tie(NULL);
    
    int n, k;
    cin >> n >> k;

    vector<vector<int>> v(n, vector<int>(n));
    cin >> v;
    ll ans = 0;
    for (int i = 0; i < n - 1; ++i) {
        vector<ll> blanks = {0};
        vector<ll> wells;
        ll ccans = 0;
        for (int j = 0; j < n + 1; ++j) {
            if (j == n || v[i][j] == 3 || v[i + 1][j] == 3) {
                // process
                // cerr << "blanks: " << blanks << "\n wells: " << wells << "\n";
                // cerr << "\n";

                ll cans = 0;

                // same parity well: dist 2 * k - 1
                // diff parity well: dist 2 * k
                ll total_left = 0;
                for (int l = 0; l < wells.size(); ++l) {
                    if (l != 0) {
                        bool span = (wells[l] != wells[l - 1]);
                        if (blanks[l] + 1 > 2 * k - 1 + span) {
                            total_left = 0;
                            // cerr << "clear\n";
                        }
                    }

                    total_left += min(blanks[l] + 1, (ll)k);
                    ll right = 1;
                    if (l + 1 < blanks.size()) right = min(blanks[l + 1] + 1, (ll)k);
                    ans += total_left * right;
                    // cerr << total_left << " " << right << "\n";
                    cans += total_left * right;
                }
                ccans += cans;
                // cerr << "cans: " << cans << "\n\n";
                blanks.clear();
                wells.clear();
                blanks.push_back(0);
            } else if (v[i][j] == 1 && v[i + 1][j] == 1) {
                blanks.back()++;
            } else {
                wells.push_back(v[i + 1][j] == 2 ? 1 : 0);
                blanks.push_back(0);
            }   
        }
        // cerr << "CCANS: " << ccans << "\n";
    }

    cout << ans << "\n";
}
```

## D. Cheating the Group System

### Solution 1: 

```cpp
 sort(v.rbegin(), v.rend());
    ll ans = (ll)n*(v[0] - 1) + 1 - v[0];
    ll rem = ans;
    for(int i = 1; i < v.size(); i++){
        ll need = (ll)n*(v[i] - 1) + 1;
        //cout << need << " " << rem << endl;
        if(rem >= need){ // we just take it out of the remaining
            ans -= v[i];
            rem -= v[i];
        } else {
            ans += need - rem - v[i];
            rem = need - v[i];
        }
        ans++;
    }
    cout << ans << endl;
```

## E. Hori and Cake

### Solution 1:  dp on tree, combinatorics

subtree size merging trick

the dp is n^2
since there are only n^2 pairs of nodes
that can be merged
at their lca
or smth like that

liek dp[i][j] = take j from subtree i
Rithwik — Today at 3:11 PM
Node, chosen
Yeah
Ok and then knapsack-esq transitions
Right?

s the number of ways to eat sibtree of node with chosen bites
Knapsack

dp[i+j] = dp1[i] * dp2[j] * (i+j choose i) 

dp[i][j] = dp[i][j-k] * dp[c][k] * choose(j, k)
and do that for all children and k

```py

```

## H. Modulo Queries

### Solution 1:  square root trick, modulo math, line sweep, square root blocks

```cpp
const int MAXN = 2e5 + 5, B = 450;
int n, q, l, r, x, arr[MAXN], psum[MAXN], pcount[MAXN], freq[MAXN], ans[MAXN];
int small[MAXN][B];
vector<pair<int, int>> events[MAXN];

void solve() {
    cin >> n >> q;
    memset(arr, 0, sizeof(arr));
    for (int i = 0; i < n; i++) {
        cin >> arr[i];
    }
    for (int i = 0; i < n; i++) {
        for (int j = 1; j < B; j++) {
            small[i][j] = arr[i] % j;
            if (i > 0) small[i][j] += small[i - 1][j];
        }
    }
    for (int i = 0; i < q; i++) {
        cin >> l >> r >> x;
        l--, r--;
        ans[i] = 0;
        if (x < B) {
            ans[i] = small[r][x];
            if (l > 0) ans[i] -= small[l - 1][x];
        } else {
            if (l == 0) {
                ans[i] += arr[l] % x; 
                l++;
            }
            while (l <= r && l % B) {
                ans[i] += arr[l] % x;
                l++;
            }
            while (l <= r && r % B) {
                ans[i] += arr[r] % x;
                r--;
            }
            if (l <= r) ans[i] += arr[l] % x;
            if (l < r) {
                events[l].emplace_back(-x, i);
                events[r].emplace_back(x, i);
            }
        }
    }
    memset(psum, 0, sizeof(psum));
    memset(pcount, 0, sizeof(pcount));
    memset(freq, 0, sizeof(freq));
    for (int i = 0; i < n; i++) {
        freq[arr[i]]++;
        if (i % B) continue;
        for (int j = 1; j < MAXN; j++) {
            psum[j] = freq[j] * j;
            if (j > 0) psum[j] += psum[j - 1];
            pcount[j] = freq[j];
            if (j > 0) pcount[j] += pcount[j - 1];
        }
        for (auto &[v, idx] : events[i]) {
            x = abs(v);
            int res = 0;
            for (int l = 0; l < MAXN; l += x) {
                r = min(l + x - 1, MAXN - 1);
                res += psum[r];
                if (l > 0) res -= psum[l - 1];
                int cnt = pcount[r];
                if (l > 0) cnt -= pcount[l - 1];
                res -= cnt * l;
            }
            if (v < 0) ans[idx] -= res;
            else ans[idx] += res;
        }
    }
    for (int i = 0; i < q; i++) {
        cout << ans[i] << " ";
    }
    cout << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    while (T--) {
        solve();
    }
    return 0;
}
```

```py
from itertools import product
B = 450
MAXN = int(2e5) + 5
psum = [0] * MAXN
pcount = [0] * MAXN

def main():
    N, Q = map(int, input().split())
    arr = list(map(int, input().split()))
    ans = [0] * Q
    small = [[0] * B for _ in range(N)] # prefix sum of modulo for (i, x), where x is the modulus
    for i, j in product(range(N), range(1, B)):
        small[i][j] = arr[i] % j
        if i > 0: small[i][j] += small[i - 1][j]
    events = [[] for _ in range(N)]
    for i in range(Q):
        l, r, x = map(int, input().split())
        l -= 1; r -= 1 # inclusive range [l, r]
        if x < B:
            ans[i] = small[r][x]
            if l > 0: ans[i] -= small[l - 1][x]
            continue
        if l == 0: 
            ans[i] += arr[l] % x 
            l += 1
        while l <= r and l % B:
            ans[i] += arr[l] % x
            l += 1
        while l <= r and r % B:
            ans[i] += arr[r] % x
            r -= 1
        if l <= r: ans[i] += arr[l] % x
        if l < r: # l and r are now multiples of B
            events[l].append((-x, i))
            events[r].append((x, i))
    freq = [0] * MAXN
    for i in range(N):
        freq[arr[i]] += 1 # only elements added up to index i
        if i % B: continue
        for val in range(MAXN): # iterating over all values
            psum[val] = freq[val] * val
            if val > 0: psum[val] += psum[val - 1]
            pcount[val] = freq[val]
            if val > 0: pcount[val] += pcount[val - 1]
        for v, idx in events[i]:
            x = abs(v)
            res = 0
            for l in range(0, MAXN, x): # iterating over all values
                r = min(l + x - 1, MAXN - 1)
                cur = psum[r] # range sum query
                if l > 0: cur -= psum[l - 1]
                over = pcount[r] # range count query to correct for them
                if l > 0: over -= pcount[l - 1]
                cur -= l * over # subtract l from count of elements in this range
                res += cur
            if v < 0:
                ans[idx] -= res
            else:
                ans[idx] += res
    print(*ans)

if __name__ == '__main__':
    T = 1
    for _ in range(T):
        main()
```