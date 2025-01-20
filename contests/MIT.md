# MITIT 

# MITIT 2024 Combined Round

tromino packing, I think it might be inclusion, exclusion principle
But yeah I'm not entirely certain, you'd have to figure out how to appropriately count them though
It is combinatorics related. 


tree coloring I don't know
dp on tree doesn't really work.
I think there may be some greedy idea

connnect buildings
minimum spanning tree
geometry, spatial 
circle
chords on a circle

when you draw a chord on the circle, you prevent anything on one side of the chord from being connected to anything across from the chord. 
so it divides up the remaining problem, cause they are now distinct from each other. 
I think you can divide up N^2 ways though, hmm that is not so great.  There will be N division steps
so N * N^2 is N^3, okay, not great but may pass some subtasks.
Now the question is how do you sum up these answers, take the best answer from each division I suppose. 

At each division you are forming a minimum spanning tree. 
how do you connect two divisions together? 
You need to take one of the two points that create the chord that is the dividing chord.  And for those nodes they need to connect to both a node on both sides and that should work. And just connect it to the node with minimum, okay this is a bit tricky, cause there could be a lot of other chords already formed before you get to this merge step.  And so that would not be easy, you'd have to pick chord so it doesn't intersect with any existing chords.  And still pick the smallest, pick the smallest chord that does not intersect. 

I don't know this is a bit weird, I think I'm off base a bit here.  I don't know if I have enough of an idea to begin implementing.  

In each division connect them back and so on. 

I think that would be N^2 time complexity if you just keep dividing the problem up into smaller and smaller problems. 


## Monotonically Increasing Tardiness Informatics Tournament

### Solution 1:  ceil division

```py
import math
N, M = map(int, input().split())
ans = 0
for _ in range(N):
    a, b = map(int, input().split())
    if a > M: continue
    ans = max(ans, math.ceil((M - a) / b))
print(ans + 1)
```

## Min-Max Game

### Solution 1:  sort, median

```py
N = int(input())
arr = sorted(map(int, input().split()))
print(arr[N // 2])
```

## Tromino Packing

### Solution 1:  dynamic programming

This only solves subproblem

```py
from itertools import product
from collections import Counter
mod = int(1e9) + 7
T = int(input())
for _ in range(T):
    N, M = map(int, input().split())
    in_bounds = lambda r, c: 0 <= r < N and 0 <= c < M
    grid = [input() for _ in range(N)]
    dp = [[0] * (M + 1) for _ in range(N + 1)]
    ans = 1
    for r in range(N):
        for c in range(M):
            for dr, dc in product([-1, 1], repeat = 2):
                if not in_bounds(r, c - dc) or grid[r][c - dc] == "#": continue
                if not in_bounds(r + dr, c) or grid[r + dr][c] == "#": continue
                dp[max(r, r + dr) + 1][c + 1] = (dp[max(r, r + dr) + 1][c + 1] + dp[r + 1][min(c, c - dc)]) % mod # add up the ones along the columns
                

    for r, c in product(range(N), range(M)):
        if grid[r][c] == "o":
            cnt = 0
            for dr, dc in product([-1, 1], repeat = 2):
                if not in_bounds(r, c - dc) or grid[r][c - dc] == "#": continue
                if not in_bounds(r + dr, c) or grid[r + dr][c] == "#": continue
                cnt += 1
            ans = (ans * cnt) % mod
    print(ans)
```

## Tree 2-Coloring

### Solution 1:  dp on tree?

```cpp

```

# MITIT 2024 Beginner Round

## A. MITIT

### Solution 1:  string

```py
Q = int(input())
for _ in range(Q):
    s = input()
    ans = False
    for len_ in range(1, len(s)):
        i = len(s) - 2 * len_
        if i <= 0: continue
        B = s[-len_:]
        C = s[i : -len_]
        if B == C: 
            ans = True
            break
    print("YES" if ans else "NO")
```

## B. Taking an Exam

### Solution 1:  math, sort

```py
T = int(input())
for _ in range(T):
    N, M = map(int, input().split())
    arr = sorted(map(int, input().split()))
    cnt = 0
    rem = M
    for d in arr:
        if rem - d < 0: break
        rem -= d
        cnt += 1
    print(M + cnt)
```

## C. Delete One Digit

### Solution 1:  divisibility rules, math, conditional logic

```py
def main():
    N = input()
    if N.count("1") == 0: return print(N, 2)
    if N.count("2") == 0:
        if len(N) & 1: N = N.replace("1", "", 1)
        return print(N, 11)
    dsum = sum(map(int, N))
    if dsum % 3 == 1: N = N.replace("1", "", 1)
    if dsum % 3 == 2: N = N.replace("2", "", 1)
    print(N, 3)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## D. Collecting Coins

### Solution 1:  binary search, dijkstra's algorithm

```cpp
const int MAXN = 2e5 + 5;
int N, M;
vector<vector<pair<int, int>>> adj;
int coins[MAXN], rew[MAXN];

bool dijkstra(int cutoff) {
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    vector<int> vis(N, false);
    pq.emplace(0, 0);
    while (!pq.empty()) {
        auto [cost, u] = pq.top();
        pq.pop();
        if (u == N - 1) return true;
        if (vis[u]) continue;
        vis[u] = true;
        for (auto [v, i] : adj[u]) {
            if (cost + coins[i] > cutoff) continue;
            if (rew[i] > coins[i]) return true;
            if (vis[v]) continue;
            pq.emplace(cost + coins[i] - rew[i], v);
        }
    }
    return false;
}

void solve() {
    cin >> N >> M;
    adj.assign(N, {});
    for (int i = 0; i < M; i++) {
        int u, v, c, r;
        cin >> u >> v >> c >> r;
        u--, v--;
        adj[u].emplace_back(v, i);
        adj[v].emplace_back(u, i);
        coins[i] = c;
        rew[i] = r;
    }
    int left = 0, right = 2e14;
    while (left < right) {
        int mid = (left + right) >> 1;
        if (!dijkstra(mid)) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    cout << left << endl;
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

# MITIT 2024 Spring Invitational Qualification

## 3-SAT 

### Solution 1: 

only partial results

```py
from collections import Counter
def main():
    N, M = map(int, input().split())
    outdegrees = Counter()
    for i in range(M):
        x, y, z = map(int, input().split())
        x -= 1; y -= 1; z -= 1
        clause = tuple(sorted(set([x, y, z])))
        outdegrees[clause] += 1
    ans = [0] * N
    if M & 1:
        print("YES")
        print(*[1] * N)
    else:
        for c, v in outdegrees.items():
            if v & 1:
                print("YES")
                for i in c:
                    ans[i] = 1
                print(*ans)
                return
        print("NO")
    
if __name__ == "__main__":
    T = int(input())
    for _ in range(T):
        main()
```

## Busy Marksman

### Solution 1:  greedy, pick from lanes with one target immediately, else pick from rest

```cpp
const int MAXN = 300'005, MAXM = 500'005;
int N, A[MAXN], ans[MAXM];

void solve() {
    cin >> N;
    int sum = 0;
    vector<int> rest, ones;
    for (int i = 0; i < N; i++) {
        cin >> A[i];
        sum += A[i];
        if (A[i] == 1) ones.push_back(i);
        else if (A[i] > 0) rest.push_back(i);
    }
    int i;
    for (i = 1; i <= sum; i++) {
        if (i & 1) {
            if (ones.size()) {
                int v = ones.end()[-1];
                ones.pop_back();
                ans[i - 1] = v + 1;
                A[v]--;
            } else if (rest.size()) {
                int v = rest.end()[-1];
                ans[i - 1] = v + 1;
                A[v]--;
                if (A[v] == 1) {
                    ones.push_back(v);
                    rest.pop_back();
                }
            } else {
                break;
            }
        } else {
            if (rest.size()) {
                int v = rest.end()[-1];
                ans[i - 1] = v + 1;
                A[v]--;
                if (A[v] == 1) {
                    ones.push_back(v);
                    rest.pop_back();
                }
            } else {
                break;
            }
        }
    }

    if (i > sum) {
        cout << "YES" << endl;
        for (int j = 0; j < sum; j++) cout << ans[j] << " ";
        cout << endl;
    } else {
        cout << "NO" << endl;
    }
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

## NM Chars

### Solution 1: 

```py

```

# M(IT)^2 2025 Winter Contest Advanced Round One

## Number Reduction

### Solution 1:  memoization, dfs, number theory, recursion

1. The key observation is that the number must be divisible the the prime integers 2, 3, 5, and 7.  Because if it is divisible by 11, that can't work cause 11 is not a digit, it ust be a digit so less than 10.

```cpp
const int INF = 1e18;
int N, ans;
const vector<int> PRIMES = {2, 3, 5, 7}, THRESHOLDS = {500000000000000000LL, 333333333333333333LL, 200000000000000000LL, 142857142857142857LL};
map<int, bool> memo;

vector<bool> digitsInNumber(int num) {
    vector<bool> digits(10, false);
    while (num > 0) {
        int digit = num % 10;
        digits[digit] = true;
        num /= 10;
    }
    return digits;
}

bool isValid(int x) {
    if (memo.count(x)) return memo[x];
    if (x == 1) return true;
    vector<bool> digits = digitsInNumber(x);
    for (int i = 2; i < 10; i++) {
        if (!digits[i] || x % i) continue;
        if (isValid(x / i)) return memo[x] = true;
    }
    return memo[x] = false;
}

void dfs(int p, int prod) {
    if (prod > N) return;
    if (p == 4) {
        if (prod <= N && isValid(prod)) ans++;
        return;
    }
    for (int i = 0; i < 64; i++) {
        int cur = prod;
        bool isGood = true;
        for (int j = 0; j < i; j++) {
            if (cur > THRESHOLDS[p]) {
                isGood = false;
                break;
            }
            cur *= PRIMES[p];
        }
        if (!isGood) break;
        dfs(p + 1, cur);
    }
}

void solve() {
    cin >> N;
    ans = 0;
    dfs(0, 1);
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

## Monster Fighting

### Solution 1:  multiset, greedy, backwards, binary search

```cpp
const int INF = 1e18;
int T, N;
multiset<int> powers[2];
vector<pair<int, int>> enemies;

int ceil(int x, int y) {
    return (x + y - 1) / y;
}

void solve() {
    cin >> N;
    powers[0].clear();
    powers[1].clear();
    for (int i = 0; i < N; i++) {
        int p, s;
        cin >> p >> s;
        powers[s].insert(p);
    }
    enemies.clear();
    for (int i = 0; i < N; i++) {
        int q, s;
        cin >> q >> s;
        enemies.emplace_back(q, s);
    }
    sort(enemies.rbegin(), enemies.rend());
    for (const auto &[q, t] : enemies) {
        auto itDiff = powers[t ^ 1].lower_bound(q);
        auto itSame = powers[t].lower_bound(ceil(q, 2));
        vector<int> cands(2, INF);
        if (itDiff != powers[t ^ 1].end()) {
            cands[t ^ 1] = *itDiff;
        }
        if (itSame != powers[t].end()) {
            cands[t] = *itSame;
        }
        if (cands[t] == INF && cands[t ^ 1] == INF) {
            cout << "NO" << endl;
            return;
        }
        if (cands[t] <= cands[t ^ 1]) {
            powers[t].erase(itSame);
        } else {
            powers[t ^ 1].erase(itDiff);
        }
    }
    cout << "YES" << endl;
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

## Drawing Lines

### Solution 1:  2SAT, SCC, Tarjan's algorithm

1. Draw out the 4 spatial relationships between LR and UD points, and consider the disjunction that is necessary, you require that they don't intersect which is only if specific combo of both have same boolean value
1. Can then derive the implication graph and set it up for 2SAT problem.

```cpp
struct Point {
    int x, y;
    string tp;
    Point() {}
    Point(int x, int y, string tp) : x(x), y(y), tp(tp) {}
};

int N, M, timer, scc_count;
vector<vector<int>> adj;
vector<int> disc, low, comp;
vector<Point> points;
stack<int> stk;
vector<bool> on_stack;

void dfs(int u) {
    disc[u] = low[u] = ++timer;
    stk.push(u);
    on_stack[u] = true;
    for (int v : adj[u]) {
        if (!disc[v]) dfs(v);
        if (on_stack[v]) low[u] = min(low[u], low[v]);
    }
    if (disc[u] == low[u]) { // found scc
        scc_count++;
        while (!stk.empty()) {
            int v = stk.top();
            stk.pop();
            on_stack[v] = false;
            low[v] = low[u];
            comp[v] = scc_count;
            if (v == u) break;
        }
    }
}

void solve() {
    cin >> N;
    points.clear();
    for (int i = 0; i < N; i++) {
        int x, y;
        string dir;
        cin >> x >> y >> dir;
        points.emplace_back(x, y, dir);
    }
    sort(points.begin(), points.end(), [](const Point &a, const Point &b) {
        return a.tp < b.tp;
    });
    adj.assign(2 * N, vector<int>());
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            if (points[i].tp == points[j].tp) continue;
            if (points[i].x > points[j].x && points[i].y > points[j].y) {
                adj[i + N].emplace_back(j + N);
                adj[j].emplace_back(i);
            } else if (points[i].x > points[j].x && points[i].y < points[j].y) {
                adj[i + N].emplace_back(j);
                adj[j + N].emplace_back(i);
            } else if (points[i].x < points[j].x && points[i].y > points[j].y) {
                adj[i].emplace_back(j + N);
                adj[j].emplace_back(i + N);
            } else if (points[i].x < points[j].x && points[i].y < points[j].y) {
                adj[i].emplace_back(j);
                adj[j + N].emplace_back(i + N);
            }
        }
    }
    disc.assign(2 * N, 0);
    low.assign(2 * N, 0);
    comp.assign(2 * N, -1);
    on_stack.assign(2 * N, false);
    scc_count = -1;
    timer = 0;
    for (int i = 0; i < 2 * N; i++) {
        if (!disc[i]) dfs(i);
    }
    for (int i = 0; i < N; i++) {
        if (comp[i] == comp[i + N]) {
            cout << "NO" << endl;
            cout << 0 << endl;
            return;
        }
    }
    cout << "YES" << endl;
    cout << 1 << endl;
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

# M(IT)^2 2025 Winter Contest Advanced Round Two

## Toy Marbles

### Solution 1:  functional graph, dfs, cycle detection, transpose graph

```cpp
struct Operation {
    int tp, u, v;
    Operation() {}
    Operation(int tp, int u, int v) : tp(tp), u(u), v(v) {}
};

int N;
vector<int> out, sources, colors;
vector<Operation> ans;
vector<bool> vis, inCycle;
vector<vector<int>> tadj;

void search(int u) {
    map<int, int> par;
    par[u] = -1;
    bool isCycle = false;
    while (!vis[u]) {
        vis[u] = true;
        int v = out[u];
        if (par.count(v)) {
            isCycle = true;
            break;
        }
        if (vis[v]) break;
        par[v] = u;
        u = v;
    }
    if (isCycle) {
        int critPoint = par[out[u]];
        vector<int> cycle;
        while (u != critPoint) {
            cycle.emplace_back(u);
            inCycle[u] = true;
            sources.emplace_back(u);
            u = par[u];
        }
        for (int i = 1; i < cycle.size(); i++) {
            swap(colors[cycle[i - 1]], colors[cycle[i]]);
            ans.emplace_back(1, cycle[i - 1], cycle[i]);
        }
    }
}

void dfs(int u, int p = -1) {
    if (!inCycle[u]) ans.emplace_back(2, out[u], u);
    for (int v : tadj[u]) {
        if (v == p) continue;
        if (inCycle[v]) continue;
        dfs(v, u);
    }
}

void solve() {
    cin >> N;
    colors.assign(N, 0);
    out.assign(N, 0);
    tadj.assign(N, vector<int>());
    for (int i = 0; i < N; i++) {
        cin >> colors[i];
        colors[i]--;
        out[i] = colors[i];
        tadj[colors[i]].emplace_back(i);
    }
    vis.assign(N, false);
    inCycle.assign(N, false);
    for (int i = 0; i < N; i++) {
        if (vis[i]) continue;
        search(i);
    }
    for (int u : sources) {
        dfs(u);
    }
    cout << ans.size() << endl;
    for (Operation op : ans) {
        cout << op.tp << " " << op.u + 1 << " " << op.v + 1 << endl;
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

## Snakes on a Grid

### Solution 1:  

```cpp
int R, C, Q;
int R1, C1, R2, C2;
vector<vector<int>> grid;
vector<vector<bool>> vis;

bool in_bounds(int i, int j) {
    return i >= R1 && i <= R2 && j >= C1 && j <= C2;
}

bool dfs(int i, int j, int k, string path) {
    char c = path[k];
    bool res = true;
    if (c == 'R') {
        if (in_bounds(i, j - 1) && grid[i][j] == grid[i][j - 1]) res &= false;; // left
        if (in_bounds(i + 1, j) && grid[i][j] == grid[i + 1][j]) res &= false; // down
        if (in_bounds(i, j + 1) && grid[i][j] == grid[i][j + 1]) {
            res &= dfs(i, j + 1, k ^ 1, path);
        }
    } else if (c == 'L') {
        if (in_bounds(i, j + 1) && grid[i][j] == grid[i][j + 1]) res &= false; // right
        if (in_bounds(i + 1, j) && grid[i][j] == grid[i + 1][j]) res &= false; // down
        if (in_bounds(i, j - 1) && grid[i][j] == grid[i][j - 1]) {
            res &= dfs(i, j - 1, k ^ 1, path);
        }
    } else if (c == 'D') {
        if (in_bounds(i - 1, j) && grid[i][j] == grid[i - 1][j]) res &= false; // up
        if (path[0] == 'R' && in_bounds(i, j + 1) && grid[i][j] == grid[i][j + 1]) res &= false; // right
        if (path[0] == 'L' && in_bounds(i, j - 1) && grid[i][j] == grid[i][j - 1]) res &= false; // left
        if (in_bounds(i + 1, j) && grid[i][j] == grid[i + 1][j]) {
            res &= dfs(i + 1, j, k ^ 1, path);
        }
    }
    return res;
}

void floodFill(int i, int j) {
    if (i < R1 || i > R2 || j < C1 || j > C2) return;
    if (vis[i][j]) return;
    vis[i][j] = true;
    if (i - 1 >= R1 && grid[i][j] == grid[i - 1][j]) floodFill(i - 1, j);
    if (i + 1 <= R2 && grid[i][j] == grid[i + 1][j]) floodFill(i + 1, j);
    if (j - 1 >= C1 && grid[i][j] == grid[i][j - 1]) floodFill(i, j - 1);
    if (j + 1 <= C2 && grid[i][j] == grid[i][j + 1]) floodFill(i, j + 1);
}

void solve() {
    cin >> R >> C;
    grid.assign(R, vector<int>(C, 0));
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            cin >> grid[i][j];
        }
    }
    vis.assign(R, vector<bool>(C, false));
    cin >> Q;
    while (Q--) {
        int r1, c1, r2, c2;
        cin >> r1 >> c1 >> r2 >> c2;
        R1 = r1, C1 = c1, R2 = r2, C2 = c2;
        int ans = 0;
        bool isGood = true;
        for (int i = r1; i <= r2; i++) {
            for (int j = c1; j <= c2; j++) {
                if (vis[i][j]) continue;
                if (j + 1 <= c2 && grid[i][j] == grid[i][j + 1]) continue;
                if (!dfs(i, j, 0, "RD") && !dfs(i, j, 1, "RD") && !dfs(i, j, 0, "LD") && !dfs(i, j, 1, "LD")) {
                    isGood = false;
                }
                floodFill(i, j);
            }
        }
        for (int i = r1; i <= r2; i++) {
            for (int j = c1; j <= c2; j++) {
                vis[i][j] = false;
            }
        }
        if (isGood) {
            cout << "YES" << endl;
        } else {
            cout << "NO" << endl;
        }
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

## MIT Tour

### Solution 1:  

```cpp

```