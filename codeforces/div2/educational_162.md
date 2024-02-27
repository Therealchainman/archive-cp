# Codeforces Educational 162 Div 2

## B. Monsters Attack!

### Solution 1: sorting, comparison

```py
def main():
    N, K = map(int, input().split())
    health = list(map(int, input().split()))
    pos = list(map(int, input().split()))
    monsters = sorted([(abs(p), h) for p, h in zip(pos, health)])
    cnt = 0
    for p, h in monsters:
        cnt += h
        if cnt > p * K: return print("NO")
    print("YES")
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## C. Find B

### Solution 1:  prefix sums, greedy

```py
def main():
    N, Q = map(int, input().split())
    arr = list(map(int, input().split()))
    pones = [0] * N
    pdec = [0] * N
    for i in range(N):
        pdec[i] = arr[i] - 1
        if arr[i] == 1: pones[i] = 1 
        if i > 0: 
            pones[i] += pones[i - 1]
            pdec[i] += pdec[i - 1]
    for _ in range(Q):
        l, r = map(int, input().split())
        l -= 1; r -= 1
        cnt_ones = pones[r]
        cnt_dec = pdec[r]
        if l > 0: 
            cnt_ones -= pones[l - 1]
            cnt_dec -= pdec[l - 1]
        if l == r or cnt_ones > cnt_dec: print("NO")
        else: print("YES")
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## D. Slimes

### Solution 1:  prefix sum?

```py

```

## E. Count Paths

### Solution 1:  small to large merging, counters, number of ways

```py
const int MAXN = 2e5 + 5;
vector<vector<int>> adj;
int colors[MAXN], N, ans;
vector<map<int, int>> counts;

void dfs(int u, int p) {
    for (int v : adj[u]) {
        if (v == p) continue;
        dfs(v, u);
        if (counts[u].size() < counts[v].size()) {
            swap(counts[u], counts[v]);
        }
        for (auto &[c, cnt] : counts[v]) {
            if (c != colors[u]) ans += counts[u][c] * cnt;
            counts[u][c] += cnt;
        }
    }
    ans += counts[u][colors[u]];
    counts[u][colors[u]] = 1;
}

void solve() {
    cin >> N;
    adj.assign(N, vector<int>());
    for (int i = 0; i < N; i++) {
        cin >> colors[i];
    }
    for (int i = 0; i < N - 1; i++) {
        int u, v;
        cin >> u >> v;
        u--, v--;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    ans = 0;
    counts.assign(N, map<int, int>());
    dfs(0, -1);
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

## 

### Solution 1: 

```py

```

