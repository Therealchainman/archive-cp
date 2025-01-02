# ICPC International Collegiate Programming Contest

# World Finals 2023

## Riddle of the Sphinx

### Solution 1:  linear independence, create third equation from three linearly independent vectors, math, logic

```py
def ask(a, b, c):
    print(a, b, c)
    return int(input())
def summation(a, b, c):
    return a + b + c
def main():
    a = ask(1, 0, 0)
    b = ask(0, 1, 0)
    c = ask(0, 0, 1)
    sum1 = ask(1, 1, 1)
    sum2 = ask(1, 1, 2)
    if summation(a, b, c) == sum1:
        print(a, b, c)
    elif summation(a, b, 2 * c) == sum2:
        print(a, b, c)
    elif summation(sum1 - b - c, b, 2 * c) == sum2:
        print(sum1 - b - c, b, c)
    elif summation(a, sum1 - a - c, 2 * c) == sum2:
        print(a, sum1 - a - c, c)
    else:
        print(a, b, sum1 - a - b)

if __name__ == "__main__":
    main()
```

## Alea Iacta Est

### Solution 1:  directed graph, cycle dependence, min heap, dynamic programming, expectation value

```cpp
const double INF = 1e9;
int N, W, trimask;
string w;
vector<int> vis, counts;
vector<double> expectation, ans;
vector<string> dice;
map<string, vector<int>> dice_states; // maps dice states to the trimask encoding
priority_queue<pair<double, int>, vector<pair<double, int>>, greater<pair<double, int>>> min_heap;

string getw(int b) {
    string s;
    for (int d = 0; d < N; d++) s += dice[d][((b>>(3*d))&7)-1];
    sort(s.begin(), s.end());
    return s;
};

// d_i is dice index
void dfs(int d_i, int mask, string w) {
    if (d_i == N) {
        sort(w.begin(), w.end());
        dice_states[w].push_back(mask);
        return;
    }
    for (int i = 0; i < dice[d_i].size(); i++) {
        dfs(d_i + 1, mask | ((i + 1) << (3 * d_i)), w + dice[d_i][i]);
    }
}

void calc(int d_i, int prob, int mask, double e) {
    if (d_i == N) {
        if (prob == 1) return;
        counts[mask]++;
        expectation[mask] += e;
        double best_expectation = (prob + expectation[mask]) / counts[mask];
        ans[mask] = best_expectation;
        min_heap.emplace(best_expectation, mask);
        return;
    }
    calc(d_i + 1, prob, mask, e);
    calc(d_i + 1, prob * dice[d_i].size(), mask & ~(7 << (3 * d_i)), e); // remove 3 bit block (3 bits represent a single dice roll)
}

void recurse(int d_i, int mask, double e) {
    if (d_i == N) {
        if (!vis[mask]) {
            calc(0, 1, mask, e);
        }
        vis[mask] = 1;
        return;
    }
    if (mask & (7 << (3 * d_i))) {
        recurse(d_i + 1, mask, e);
    } else {
        for (int i = 0; i < dice[d_i].size(); i++) {
            recurse(d_i + 1, mask | ((i + 1) << (3 * d_i)), e);
        }
    }
}

void solve() {
    cin >> N >> W;
    dice.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> dice[i];
    }
    dfs(0, 0, "");
    vis.assign(1 << (3 * N), 0);
    counts.assign(1 << (3 * N), 0);
    expectation.assign(1 << (3 * N), 0.0);
    ans.assign(1 << (3 * N), INF);
    for (int i = 0; i < W; i++) {
        cin >> w;
        sort(w.begin(), w.end());
        for (int mask : dice_states[w]) {
            recurse(0, mask, 0.0);
        }
    }
    if (ans[0] == INF) {
        cout << "impossible" << endl;
        return;
    }
    while (!min_heap.empty()) {
        auto [e, mask] = min_heap.top();
        min_heap.pop();
        if (vis[mask]) continue;
        vis[mask] = 1;
        recurse(0, mask, e);
    }
    cout << fixed << setprecision(7) << ans[0] << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Three Kinds of Dice

### Solution 1:  probability, intransitive dice, convex hull, rotation, geometry

```cpp
int N1, N2;
vector<int> D1, D2;

void load(int& N, vector<int>& D) {
    cin >> N;
    D.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> D[i];
    }
}

struct P {
    double x, y;
};

void solve() {
    // freopen("input.txt", "r", stdin);
    // freopen("output.txt", "w", stdout);
    load(N1, D1);
    load(N2, D2);
    sort(D1.begin(), D1.end());
    sort(D2.begin(), D2.end());
    int prob = 0;
    for (int i = 0, l = 0, r = 0; i < N1; i++) {
        while (D2[l] < D1[i]) l++;
        while (D2[r] <= D1[i]) r++;
        prob += 2 * l + (r - l);
    }
    assert(prob != N1 * N2);
    if (prob < N1 * N2) {
        swap(D1, D2);
        swap(N1, N2);
    }
    vector<int> poss;
    for (auto x : D1) { if (x > 1) poss.push_back(x-1); poss.push_back(x); poss.push_back(x+1); }
    for (auto x : D2) { if (x > 1) poss.push_back(x-1); poss.push_back(x); poss.push_back(x+1); }
    sort(poss.begin(), poss.end());
    poss.erase(unique(poss.begin(), poss.end()), poss.end());
    vector<P> points;
    for (int pi = 0, i1 = 0, i2 = 0, j1 = 0, j2 = 0; pi < poss.size(); pi++) {
      while (D1[i1] < poss[pi]) i1++;
        while (D2[i2] < poss[pi]) i2++;
        while (D1[j1] <= poss[pi]) j1++;
        while (D2[j2] <= poss[pi]) j2++;
        double x = (2*i1+(j1-i1)) / 2.0 / N1;
        double y =  (2*i2+(j2-i2)) / 2.0 / N2;
        points.push_back(P{x, y});
    }
    for (int rep = 0; rep < 2; rep++) {
        vector<int> hull; // hull that holds the indices of the points
        for (int i = 0; i < points.size(); i++) {
            auto [x3, y3] = points[i];
            while (hull.size() >= 2) {
                auto [x1, y1] = points[hull.end()[-2]];
                auto [x2, y2] = points[hull.end()[-1]];
                if ((x3 - x1) * (y2 - y1) < (x2 - x1) * (y3 - y1)) break;
                hull.pop_back();
            }
            hull.push_back(i);
        }
        double ans = 1.0;
        for (int i = 1; i < hull.size(); i++) {
            auto [x1, y1] = points[hull[i - 1]];
            auto [x2, y2] = points[hull[i]];
            if (x1 >= 0.5 || x2 < 0.5) continue;
            ans = y1 + (y2 - y1) / (x2 - x1) * (0.5 - x1);
        }
        if (rep == 0) {
            cout << fixed << setprecision(7) << ans << " ";
        } else {
            cout << fixed << setprecision(7) << 1 - ans << endl;
        }
        for (P &a : points) {
            swap(a.x, a.y);
            a.x = 1 - a.x;
            a.y = 1 - a.y;
        }
        reverse(points.begin(), points.end());
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

## Waterworld

### Solution 1:  arimetic average, spherical geometry

```py
def main():
    n, m = map(int, input().split())
    mat = [list(map(int, input().split())) for _ in range(n)]
    ans = 0
    for row in mat:
        ans += sum(row)
    ans /= n * m 
    print(ans)

if __name__ == "__main__":
    main()
```

## Bridging the Gap

### Solution 1:  dynamic programming, optimization

```py
import math
sys.setrecursionlimit(10 ** 6)
def ceil(x, y):
    return (x + y - 1) // y
def main():
    N, C = map(int, input().split())
    arr = sorted(map(int, input().split()), reverse = True)
    LIM = ceil(N, C) + 1
    memo = [[math.inf] * LIM for _ in range(N)]
    # O(N * (N / C))
    def dfs(idx, back):
        if back <= 0 or back >= LIM: return math.inf
        if idx + C >= N: return arr[idx]
        if memo[idx][back] < math.inf: return memo[idx][back]
        best = math.inf
        ssum = 0
        back -= 1
        for i in range(C):
            if back == 0 and idx > 0:
                if C - i - 1 > 0:
                    best = min(best, dfs(idx + C - i - 1, back + i) + arr[idx - 1] + ssum + arr[idx - 1])
            else:
                best = min(best, dfs(idx + C - i, back + i) + arr[idx] + ssum)
            ssum += arr[N - i - 1]
        ssum += arr[N - C]
        best = min(best, dfs(idx, back + C) + ssum)
        memo[idx][back + 1] = best
        return memo[idx][back + 1]
    ans = dfs(0, 1)
    print(ans)

if __name__ == "__main__":
    main()
```

```cpp
const int INF = 1e18;
int N, C;
vector<int> arr;
vector<vector<int>> memo;

int ceil(int x, int y) {
    return (x + y - 1) / y;
}

int dfs(int idx, int back) {
    if (back < 0 || back >= memo[0].size()) return INF;
    if (idx + C >= N) return arr[idx];
    if (memo[idx][back] < INF) return memo[idx][back];
    int best = INF;
    int ssum = 0;
    int resp;
    for (int i = 0; i < C; i++) {
        if (back == 0 && idx > 0) {
            if (C - i - 1 > 0) {
                resp = dfs(idx + C - i - 1, back + i - 1);
                if (resp < INF) best = min(best, resp + 2 * arr[idx - 1] + ssum);
            }
        } else {
            resp = dfs(idx + C - i, back + i - 1);
            if (resp < INF) best = min(best, resp + arr[idx] + ssum);
        }
        ssum += arr[N - i - 1];
    }
    ssum += arr[N - C];
    resp = dfs(idx + C, back + C - 1);
    if (resp < INF) best = min(best, resp + ssum);
    // cout << idx << " " << best << endl;
    memo[idx][back] = best;
    return memo[idx][back];
}

void solve() {
    // freopen("input.txt", "r", stdin);
    // freopen("output.txt", "w", stdout);
    cin >> N >> C;
    arr.resize(N);
    for (int i = 0; i < N; ++i) {
        cin >> arr[i];
    }
    sort(arr.rbegin(), arr.rend());
    int LIM = N + 1;
    memo.assign(N, vector<int>(LIM, INF));
    int ans = dfs(0, 0);
    cout << ans << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

# Regional Contests 2024

## Balanced Tree Path

### Solution 1:  dfs, stack, tree

([)]

```cpp
const int INF = 1e9;
int N, ans;
string sym;
const string OPEN = "([{", CLOSE = ")]}";
vector<vector<int>> adj;
stack<int> stk;

void dfs(int u, int p) {
    char br = sym[u];
    char last;
    if (OPEN.find(br) != string::npos) stk.push(br);
    else {
        if (stk.empty()) return;
        last = stk.top();
        if (OPEN.find(last) != CLOSE.find(br)) return;
        stk.pop();
    }
    if (stk.empty()) ans++;
    for (int v : adj[u]) {
        if (v != p) dfs(v, u);
    }
    if (OPEN.find(br) != string::npos) stk.pop();
    else stk.push(last);
}

void solve() {
    cin >> N;
    cin >> sym;
    adj.assign(N, vector<int>());
    for (int i = 0; i < N - 1; i++) {
        int u, v;
        cin >> u >> v;
        u--, v--;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    ans = 0;
    for (int i = 0; i < N; i++) {
        while (!stk.empty()) stk.pop();
        dfs(i, -1);
    }
    cout << ans << endl;
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

## ABC String

### Solution 1:  greedy, count

```py
def main():
    s = input()
    CHARS = "ABC"
    counts = [0] * 3
    ans = 0
    for ch in s:
        idx = CHARS.index(ch)
        counts[idx] += 1
        diff = max(counts) - min(counts)
        ans = max(ans, diff)
    print(ans)

if __name__ == '__main__':
    main()
```

## Ordered Problem Set

### Solution 1:  prefix max, brute force

```py
def main():
    n = int(input())
    arr = [int(input()) for _ in range(n)]
    ans = []
    def check():
        pmax = cmax = 0
        sz = n // k
        for i in range(n):
            if i % sz == 0:
                pmax = max(pmax, cmax)
                cmax = 0
            cmax = max(cmax, arr[i])
            if arr[i] < pmax: return False
        return True
    for k in range(2, n + 1):
        if n % k != 0: continue
        if check(): ans.append(k)
    if len(ans) > 0: 
        for num in ans: print(num)
    else: print(-1)

if __name__ == '__main__':
    main()
```

## Item Selection

### Solution 1:  greedy, implementation, math, set, set difference

```py
import math
def main():
    # n items, m per page, page s, p preselected, q to be selected
    n, m, s, p, q = map(int, input().split())
    s -= 1
    minp, maxp = math.inf, -math.inf
    # preselected items
    pre = [0] * p
    # items to be selected
    sel = [0] * q
    for i in range(p):
        x = int(input()) - 1
        pre[i] = x 
    for i in range(q):
        x = int(input()) - 1
        sel[i] = x
    i = j = 0
    while i < p or j < q:
        if i < p and (j == q or pre[i] < sel[j]):
            minp = min(minp, pre[i] // m)
            maxp = max(maxp, pre[i] // m)
            i += 1
        elif j < q and (i == p or sel[j] < pre[i]):
            minp = min(minp, sel[j] // m)
            maxp = max(maxp, sel[j] // m)
            j += 1
        else:
            i += 1
            j += 1
    if minp == math.inf: return print(0)
    ans = i = j = 0
    for page in range((n + m - 1) // m):
        l, r = page * m, min(n, (page + 1) * m)
        pcnt = r - l
        take, rem = set(), set()
        while i < p and pre[i] // m == page: rem.add(pre[i]); i += 1
        while j < q and sel[j] // m == page: take.add(sel[j]); j += 1
        cost = len(take - rem) + len(rem - take)
        if len(take) > 0:
            # deselect all, select or select all, deselect
            cost = min(cost, len(take) + 1, pcnt - len(take) + 1)
        elif len(rem) > 0: # just need to deslect all
            cost = min(cost, 1)
        ans += cost
    ans += maxp - minp
    # traverse the shorter segment a second time
    ans += min(abs(s - minp), abs(s - maxp))
    print(ans)

if __name__ == '__main__':
    main()
```

## Streets Behind

### Solution 1:  math

Cannot find the math formula to solve this one. 

```py
import math
def main():
    n, k, a, b = map(int, input().split())
    total = n + k 
    n2 = n
    ans = 0
    g = math.gcd(a, b)
    a //= g
    b //= g
    delta = n * b // a - n 
    rem = (n * b) % a
    if delta <= 0: return print(-1)
    while n < total:
        # 0, 1, 2, 3, 4
        # 0, 2, 4
        # starts at 0, goes till 0 +1
        # starts at 0 goes till 1 +2
        # starts at 1 goes till 2 +3
        # starts at 2 goes till 1 +4
        # starts at 1 goes till 1 +5
        # starts at 1 goes till 2 +6
        # starts at 2 goes till 4 +7
        # starts at 4 goes till 
        # starts at 4, 0, 1, 2, 3, 4, 0 
        # add it to 
        ub = max(1, (total - n) // delta)
        x = min(ub, (a - 1 - rem) // delta + 1)
        n += delta * x
        change = max(1, (rem + delta * x) // a)
        rem = (rem + delta * x) % a
        delta += change
        print("x", x, "n", n, "rem", rem, "delta", delta)
        ans += x
    print(ans)
    ans2 = 0
    while n2 < total:
        floor = n2 * b // a 
        print("delta", floor - n2)
        n2 = floor 
        print("n2", n2)
        ans2 += 1
    print("ans2", ans2)
    assert(ans == ans2)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
"""
1
10 1000000000 13 23
1
10 1000000000 10 12
1
10 1000000000 5 6
"""
```

```py
def main():
    n, k, a, b = map(int, input().split())
    total = n + k 
    n2 = n
    ans = 0
    deltas = []
    while n < total:
        # res.append(n)
        delta_n = n * b // a - n
        if delta_n == 0: return print(-1)
        deltas.append(delta_n)
        left, right = n, total 
        while left < right:
            mid = (left + right) >> 1
            dn = mid * b // a - mid
            if dn <= delta_n:
                left = mid + 1
            else:
                right = mid
        delta = left - n
        cnt = (delta + delta_n - 1) // delta_n
        if cnt == 0: return print(-1)
        n += cnt * delta_n
        ans += cnt
    print(ans)
```

# 2024 ICPC Asia Taichung Regional Contest

## D. Drunken Maze

### Solution 1:  grid, direction, bfs, queue

```cpp
struct Position {
    int r, c, lastDirection, cnt;
    Position() {}
    Position(int r, int c, int lastDirection, int cnt) : r(r), c(c), lastDirection(lastDirection), cnt(cnt) {}
};

int R, C;
vector<vector<char>> grid;
vector<vector<bool>> vis[4][4];
queue<Position> q;
const vector<pair<int, int>> DIRECTIONS = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};

bool in_bounds(int r, int c) {
    return r >= 0 && r < R && c >= 0 && c < C;
}

void solve() {
    cin >> R >> C;
    grid.assign(R, vector<char>(C));
    for (int r = 0; r < R; r++) {
        string S;
        cin >> S;
        for (int c = 0; c < C; c++) {
            grid[r][c] = S[c];
            if (grid[r][c] == 'S') {
                q.emplace(r, c, 0, 0);
            }
        }
    }
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            vis[i][j].assign(R, vector<bool>(C, false));
        }
    }
    int steps = 0;
    while (!q.empty()) {
        int sz = q.size();
        for (int _ = 0; _ < sz; _++) {
            Position pos = q.front();
            q.pop();
            if (grid[pos.r][pos.c] == 'T') {
                cout << steps << endl;
                return;
            }
            for (int d = 0; d < 4; d++) {
                auto [dr, dc] = DIRECTIONS[d];
                int nr = pos.r + dr, nc = pos.c + dc;
                if (!in_bounds(nr, nc) || grid[nr][nc] == '#') continue;
                int nxtCount = d == pos.lastDirection ? pos.cnt + 1 : 1;
                if (nxtCount > 3) continue;
                if (vis[d][nxtCount][nr][nc]) continue;
                q.emplace(nr, nc, d, nxtCount);
                vis[d][nxtCount][nr][nc] = true;
            }
        }
        steps++;
    }
    cout << -1 << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## C. Cube

### Solution 1:  bitmask dynamic programming

1. dp over subsets of used y and z coordinates
1. The key is that it doesn't traverse only 2^(2n) pairs of (Y, Z) it only traverses sum of 
1. There is a trick to this one that you can get how many of one of the values of x coordinates, you processed from the current mask1,  because you must have taken the same number from both mask for y and mask for z values.

```cpp
const int INF = 1e15;
int N;
vector<vector<vector<int>>> mat;

bool isSet(int mask, int i) {
    return (mask >> i) & 1;
}

void solve() {
    cin >> N;
    mat.assign(N, vector<vector<int>>(N, vector<int>(N)));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                cin >> mat[i][j][k];
            }
        }
    }
    int endMask = (1 << N) - 1;
    vector<vector<int>> dp(1 << N, vector<int>(1 << N, INF));
    dp[0][0] = 0;
    for (int mask1 = 0; mask1 < (1 << N); mask1++) {
        int k = __builtin_popcount(mask1);
        for (int mask2 = 0; mask2 < (1 << N); mask2++) {
            int l = __builtin_popcount(mask2);
            if (l != k) continue;
            for (int i = 0; i < N; i++) {
                if (isSet(mask1, i)) continue;
                for (int j = 0; j < N; j++) {
                    if (isSet(mask2, j)) continue;
                    if (dp[mask1][mask2] == INF) continue;
                    int nmask1 = mask1 | (1 << i), nmask2 = mask2 | (1 << j);
                    dp[nmask1][nmask2] = min(dp[nmask1][nmask2], dp[mask1][mask2] + mat[i][j][k]);
                }
            }
        }
    }
    cout << dp[endMask][endMask] << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## H. Sheet Music

### Solution 1:  counting, dynamic programming, binomial coefficients, combinatorics

1. count the number of segments that don't have k consecutive of the same element
1. There are only two elements of interest which are < and >
1. Once you count the arrangements for every size with those elements above such that there are never k consecutive.
1. Then you just need to use combinatorics to count the number of places you can place the rest of the characters to be = signs.

```cpp
const int MOD = 998244353;
int N, K;
vector<int> dp;

int inv(int i, int m) {
  return i <= 1 ? i : m - (int)(m/i) * inv(m % i, m) % m;
}

vector<int> fact, inv_fact;

void factorials(int n, int m) {
    fact.assign(n + 1, 1);
    inv_fact.assign(n + 1, 0);
    for (int i = 2; i <= n; i++) {
        fact[i] = (fact[i - 1] * i) % m;
    }
    inv_fact.end()[-1] = inv(fact.end()[-1], m);
    for (int i = n - 1; i >= 0; i--) {
        inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % m;
    }
}

int choose(int n, int r, int m) {
    if (n < r) return 0;
    return (fact[n] * inv_fact[r] % m) * inv_fact[n - r] % m;
}

void solve() {
    cin >> N >> K;
    if (K == 1) {
        cout << 1 << endl;
        return;
    }
    factorials(N, MOD);
    dp.assign(N, 0);
    dp[0] = dp[1] = 2;
    for (int i = 2; i < N; i++) {
        dp[i] = 2LL * dp[i - 1] % MOD;
        if (i >= K) dp[i] = (dp[i] - dp[i - K] + MOD) % MOD;
    }
    dp[0]--;
    int ans = 0;
    for (int i = 0; i < N; i++) {
        int nonzeros = dp[i];
        int zeros = N - 1 - i;
        ans = (ans + nonzeros * choose(N - 1, zeros, MOD) % MOD) % MOD;
    }
    cout << ans << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## M. Selection Sort

### Solution 1:  sorting, greedy

1. basically you can sort prefix then suffix or suffix then prefix with overlapping. 
1. Or you may just need to sort prefix or just suffix
1. Or you may need to sort prefix and suffix with them being disjoint (non-overlapping), if there is a middle section that is sorted, and where they belong.

```cpp
int N;
vector<pair<int, int>> A;
vector<int> order;
vector<bool> seen;

void solve() {
    cin >> N;
    A.resize(N);
    order.resize(N);
    for (int i = 0; i < N; i++) {
        int x;
        cin >> x;
        A[i] = {x, i};
    }
    stable_sort(A.begin(), A.end());
    for (int i = 0; i < N; i++) {
        auto [x, idx] = A[i];
        order[idx] = i;
    }
    seen.assign(N, false);
    int ans = N * N;
    // prefix -> suffix
    for (int i = 0, p = 0; i < N; i++) {
        seen[order[i]] = true;
        while (p < N && seen[p]) p++;
        if (p > 0) {
            int prefixLength = i + 1;
            int suffixLength = N - p;
            ans = min(ans, prefixLength * prefixLength + suffixLength * suffixLength);
        }
    }
    seen.assign(N, false);
    // suffix -> prefix
    for (int i = N - 1, p = N - 1; i >= 0; i--) {
        seen[order[i]] = true;
        while (p >= 0 && seen[p]) p--;
        if (p < N - 1) {
            int suffixMatchedCount = N - p - 1;
            int prefixLength = N - suffixMatchedCount;
            int suffixLength = N - i;
            ans = min(ans, prefixLength * prefixLength + suffixLength * suffixLength);
        }
    }
    // sort prefix only
    for (int i = N - 1; i >= 0; i--) {
        if (order[i] != i) break;
        ans = min(ans, i * i);
    }
    // sort suffix only
    for (int i = 0; i < N; i++) {
        if (order[i] != i) break;
        ans = min(ans, (N - i - 1) * (N - i - 1));
    }
    // right end point
    vector<int> right(N + 1, N);
    int s = N, e = 0;
    for (int i = N - 1; i >= 0; i--) {
        right[i] = min(right[i + 1], order[i]);
    }
    // sort middle? 
    for (int i = 0, l = 0; i < N; i++) {
        l = max(l, order[i]);
        if (i == l && right[i] == i) {
            s = min(i, s);
            e = max(i, e);
        } else {
            if (s < e) {
                int plen = s + 1, slen = N - e - 1;
                ans = min(ans, plen * plen + slen * slen);
            }
            s = i;
            e = i;
        }
    }
    cout << ans << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

# contest name?

##

### Solution 1: 

```cpp

```

##

### Solution 1: 

```cpp

```

##

### Solution 1: 

```cpp

```

##

### Solution 1: 

```cpp

```