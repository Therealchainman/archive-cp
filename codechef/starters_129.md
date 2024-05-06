# Starters 129

## SumXor

### Solution 1:  bit manipulation, xor, bit frequency for integers less than N, prefix and suffix product, dynamic programming, counting

```py
BITS = 30
MOD = 998244353
def count_bits(x, b):
    cycle_len = 2 ** (b + 1)
    cycle_cnt = x // cycle_len
    cycle_rem = x % cycle_len - 2 ** b + 1
    return cycle_cnt * 2 ** b + max(0, cycle_rem)

def main():
    N = int(input())
    L = list(map(int, input().split()))
    R = list(map(int, input().split()))
    # figure this out how to get bit frequency for an interval
    bfreq = [[0] * BITS for _ in range(N)]
    for i in range(N):
        for j in range(BITS):
            bfreq[i][j] = count_bits(R[i], j) % MOD
            if L[i] > 0: bfreq[i][j] = (bfreq[i][j] - count_bits(L[i] - 1, j)) % MOD
    pfreq = [0] * N
    sfreq = [0] * N
    for i in range(N):
        pfreq[i] = R[i] - L[i] + 1
        if i > 0: pfreq[i] = (pfreq[i] * pfreq[i - 1]) % MOD 
    for i in reversed(range(N)):
        sfreq[i] = R[i] - L[i] + 1
        if i < N - 1: sfreq[i] = (sfreq[i] * sfreq[i + 1]) % MOD
    ans = 0
    for b in range(BITS):
        for l in range(N):
            x = 1 # number of ways for b to not be set in last element
            y = 0 # number of ways for b to be set in last element
            for r in range(l, N):
                cnt_b = bfreq[r][b]
                cnt_no_b = R[r] - L[r] + 1 - cnt_b
                x, y = (x * cnt_no_b + y * cnt_b) % MOD, (x * cnt_b + y * cnt_no_b) % MOD
                ways = y
                if l > 0: ways = (ways * pfreq[l - 1]) % MOD
                if r < N - 1: ways = (ways * sfreq[r + 1]) % MOD
                ans = (ans + ways * (1 << b)) % MOD
    print(ans)

if __name__ == "__main__":
    T = int(input())
    for _ in range(T):
        main()
```

## Red Blue Decomposition

### Solution 1:  tree, dfs, flipping, parity

```cpp
const int RED = 0, BLUE = 1;
int N;
vector<vector<int>> adj;
vector<int> color, flip;

int dfs1(int u, int p) {
    int sz = 1;
    int act = 0;
    for (int v : adj[u]) {
        if (v == p) continue;
        int csz = dfs1(v, u);
        if (csz & 1) {
            if (act) flip[v] ^= 1;
            act ^= 1;
        }
        sz += csz;
    }
    if (sz % 2 == 0) color[u] = BLUE;
    return sz;
}

void dfs2(int u, int p, int f) {
    for (int v : adj[u]) {
        if (v == p) continue;
        dfs2(v, u, flip[v] ^ f);
    }
    if (flip[u]) color[u] ^= 1;
}

void solve() {
    cin >> N;
    adj.assign(N, {});
    for (int i = 0; i < N - 1; i++) {
        int u, v;
        cin >> u >> v;
        u--, v--;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    color.assign(N, RED);
    flip.assign(N, 0);
    dfs1(0, -1);
    dfs2(0, -1, 0);
    string ans;
    ans.reserve(N);
    for (int i = 0; i < N; i++) {
        ans += (color[i] == RED ? 'R' : 'B');
    }
    cout << ans << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    int T;
    cin >> T;
    while (T--) {
        solve();
    }
    return 0;
}
```