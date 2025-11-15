# CALICO Informatics Competition

# remix spring 2024

## Problem 1: Ice Cream Bars!

### Solution 1: binary search

Need a better solution to solve when N = 10^10_000

```py
def main():
    N = int(input())
    eval = lambda n: n * (n + 1) / 2
    l, r = 0, 10 ** 15
    while l < r:
        m = (l + r + 1) >> 1
        if eval(m) <= N:
            l = m
        else:
            r = m - 1
    print(l)

if __name__ == "__main__":
    T = int(input())
    for _ in range(T):
        main()
```

## Problem 6: Fractals Against Programmability

### Solution 1:  recursion, grid, split into bounding boxes

```cpp

```

## Problem 8: Bay Area’s Revolutionary Train

### Solution 1:  sortedlist, modular arithmetic, fenwick tree, binary search

```py
from collections import defaultdict

from bisect import bisect_left as lower_bound
from bisect import bisect_right as upper_bound

class FenwickTree:
    def __init__(self, x):
        bit = self.bit = list(x)
        size = self.size = len(bit)
        for i in range(size):
            j = i | (i + 1)
            if j < size:
                bit[j] += bit[i]

    def update(self, idx, x):
        """updates bit[idx] += x"""
        while idx < self.size:
            self.bit[idx] += x
            idx |= idx + 1

    def __call__(self, end):
        """calc sum(bit[:end])"""
        x = 0
        while end:
            x += self.bit[end - 1]
            end &= end - 1
        return x

    def find_kth(self, k):
        """Find largest idx such that sum(bit[:idx]) <= k"""
        idx = -1
        for d in reversed(range(self.size.bit_length())):
            right_idx = idx + (1 << d)
            if right_idx < self.size and self.bit[right_idx] <= k:
                idx = right_idx
                k -= self.bit[idx]
        return idx + 1, k

class SortedList:
    block_size = 700

    def __init__(self, iterable=()):
        self.macro = []
        self.micros = [[]]
        self.micro_size = [0]
        self.fenwick = FenwickTree([0])
        self.size = 0
        for item in iterable:
            self.insert(item)

    def insert(self, x):
        i = lower_bound(self.macro, x)
        j = upper_bound(self.micros[i], x)
        self.micros[i].insert(j, x)
        self.size += 1
        self.micro_size[i] += 1
        self.fenwick.update(i, 1)
        if len(self.micros[i]) >= self.block_size:
            self.micros[i:i + 1] = self.micros[i][:self.block_size >> 1], self.micros[i][self.block_size >> 1:]
            self.micro_size[i:i + 1] = self.block_size >> 1, self.block_size >> 1
            self.fenwick = FenwickTree(self.micro_size)
            self.macro.insert(i, self.micros[i + 1][0])

    # requires index, so pop(i)
    def pop(self, k=-1):
        i, j = self._find_kth(k)
        self.size -= 1
        self.micro_size[i] -= 1
        self.fenwick.update(i, -1)
        return self.micros[i].pop(j)

    def __getitem__(self, k):
        i, j = self._find_kth(k)
        return self.micros[i][j]

    def count(self, x):
        return self.upper_bound(x) - self.lower_bound(x)

    def __contains__(self, x):
        return self.count(x) > 0

    def lower_bound(self, x):
        i = lower_bound(self.macro, x)
        return self.fenwick(i) + lower_bound(self.micros[i], x)

    def upper_bound(self, x):
        i = upper_bound(self.macro, x)
        return self.fenwick(i) + upper_bound(self.micros[i], x)

    def _find_kth(self, k):
        return self.fenwick.find_kth(k + self.size if k < 0 else k)

    def __len__(self):
        return self.size

    def __iter__(self):
        return (x for micro in self.micros for x in micro)

    def __repr__(self):
        return str(list(self))

def main():
    N, M, K = map(int, input().split())
    sources = list(map(lambda x: int(x) - 1, input().split()))
    targets = list(map(lambda x: int(x) - 1, input().split()))
    station = defaultdict(list)
    for i in reversed(range(N)):
        station[sources[i]].append(i)
    waiting = SortedList(sources)
    dropoff = SortedList() # (distance at which drop off)
    dist = load = 0
    while waiting or dropoff:
        next_station = M + 1
        if load < K and waiting:
            idx = waiting.lower_bound(dist % M)
            if idx == len(waiting): idx = 0
            next_station = waiting[idx]
        if dropoff:
            idx = dropoff.lower_bound(dist % M)
            if idx == len(dropoff): idx = 0
            cand_station = dropoff[idx]
            if next_station < dist % M: 
                if cand_station < dist % M: # both on left side, so take minimum
                    next_station = min(next_station, cand_station)
                else: # next_station on left, cand_station on right
                    next_station = cand_station
            else:
                if cand_station >= dist % M:
                    next_station = min(next_station, cand_station) # both on right side, so take minimum
                if next_station == M + 1: # no one to pick up or drop off
                    next_station = cand_station
        # update dist to get to next_station
        if next_station >= dist % M: 
            delta = next_station - dist % M
        else:
            delta = M - dist % M + next_station
        dist += delta
        idx = dropoff.lower_bound(dist % M)
        if idx < len(dropoff) and dropoff[idx] == dist % M:
            dropoff.pop(idx)
            load -= 1
        else:
            idx = waiting.lower_bound(dist % M)
            load += 1
            waiting.pop(idx)
            dst = targets[station[next_station].pop()]
            dropoff.insert(dst)
    print(dist)

if __name__ == "__main__":
    T = int(input())
    for _ in range(T):
        main()
```

## Problem 3: Bay Area’s Railway Traversal

### Solution 1:  longest difference in modular arithmetic

```py
def main():
    N, M = map(int, input().split())
    sources = list(map(int, input().split()))
    targets = list(map(int, input().split()))
    ans = 0
    for src, dst in zip(sources, targets):
        if dst >= src: ans = max(ans, dst - src)
        else: ans = max(ans, M - src + dst)
    print(ans)

if __name__ == "__main__":
    T = int(input())
    for _ in range(T):
        main()
```

## Problem X: ________ DOLLARS

### Solution 1: 

```py

```

## Problem 11: hollup…Let him cook

### Solution 1: 

```py
def main():
    while True:
        try:
            D = int(input())
            t = input()
            if t == "INCREMENT": print(D + 1)
            elif t == "DECREMENT": print(D - 1)
            else: print(D)
        except:
            break

if __name__ == "__main__":
    main()
```

## Problem 3: @everyone

### Solution 1:  sorting, offline queries, queue, difference array

```py
def main():
    Q, N, M = map(int, input().split())
    pings = [[] for _ in range(N)]
    roles = [[] for _ in range(N)]
    ans = [0] * M
    for i in range(Q):
        action = input().split()
        if action[0] == "A":
            r, u = map(int, action[1:])
            r -= 1; u -= 1
            roles[r].append((i, u))
        elif action[0] == "R":
            r, u = map(int, action[1:])
            r -= 1; u -= 1
            roles[r].append((i, u))
        else:
            r = int(action[1])
            r -= 1
            pings[r].append(i)
    for r in range(N):
        i = 0
        n = len(pings[r])
        marked = set()
        for idx, p in enumerate(pings[r]):
            while i < len(roles[r]) and roles[r][i][0] < p:
                u = roles[r][i][1]
                if u in marked:
                    ans[u] -= n - idx
                    marked.remove(u)
                else:
                    ans[u] += n - idx
                    marked.add(u)
                i += 1
    print(*ans)

if __name__ == "__main__":
    T = int(input())
    for _ in range(T):
        main()
```

# calico spring 2024

## Problem 8: Cow Pinball

### Solution 1:  convergence, dfs, tree, probability, reroot tree

```cpp
int N, S, E, C, L;
vector<vector<int>> adj;
vector<int> vis, cycle;
double cprob;

bool dfs(int u, int p) {
    if (vis[u]) {
        C = u;
        return true;
    }
    vis[u] = 1;
    bool res = false;
    for (int v : adj[u]) {
        if (v == p) continue;
        if (dfs(v, u)) {
            cycle[u] = 1;
            res = true;
        }
    }
    vis[u] = 0;
    return C == u ? false : res;
}

void dfs2(int u, int p) {
    if (vis[u]) return;
    int c = adj[u].size();
    double prob = 1.0;
    if (c > 0) {
        prob = 1.0 / c;
    }
    vis[u] = 1;
    for (int v : adj[u]) {
        if (cycle[u] && cycle[v]) {
            cprob *= prob;
        }
        if (v == p) continue;
        dfs2(v, u);
    }
    vis[u] = 0;
}

double converge(double pr, int depth) {
    double eps = 1e-12;
    double cur = pr * depth;
    double ans = 0.0;
    int loops = 0;
    while (cur > eps) {
        ans += cur;
        loops++;
        cur = pr * pow(cprob, loops) * (depth + loops * L);
    }
    return ans;
}

double dfs1(int u, int p, double pr = 1.0, int depth = 0, bool is_cycle = false) {
    if (vis[u]) return 0.0;
    int c = adj[u].size();
    double prob = 1.0;
    if (c > 0) {
        prob = 1.0 / c;
    }
    double ans = 0;
    vis[u] = 1;
    for (int v : adj[u]) {
        if (v == p) continue;
        ans += dfs1(v, u, pr * prob, depth + 1, is_cycle | cycle[u]);
    }
    if (c == 0) {
        if (is_cycle) {
            ans += converge(pr, depth);
        } else {
            ans += pr * depth;
        }
    }
    vis[u] = 0;
    return ans;
}

void solve() {
    cin >> N >> S >> E;
    S--; E--;
    adj.assign(N, vector<int>());
    cycle.assign(N, 0);
    C = -1;
    for (int i = 1; i < N; i++) {
        int p;
        cin >> p;
        p--;
        if (S == i && E == p) {
            C = p;
            cycle[i] = cycle[p] = 1;
        }
        adj[p].push_back(i);
    }
    adj[S].push_back(E);
    vis.assign(N, 0);
    if (C == -1) {
        dfs(0, -1);
    }
    L = accumulate(cycle.begin(), cycle.end(), 0);
    cprob = 1.0;
    if (L > 0) {
        vis.assign(N, 0);
        dfs2(0, -1);
    }
    vis.assign(N, 0);
    double ans = dfs1(0, -1);
    cout << fixed << setprecision(10) << ans << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    // freopen("input.txt", "r", stdin);
    // freopen("output.txt", "w", stdout);
    int T;
    cin >> T;
    while (T--) {
        solve();
    }
    return 0;
}
```

# remix fall 2024

## Problem 5: Big Ben’s Jenga Bricks

### Solution 1:  cycle detection, lcm

```cpp
int inv(int i, int m) {
  return i <= 1 ? i : m - (m/i) * inv(m % i, m) % m;
}

const int MOD = 3359232, MAXN = 4374;
int N, arrSum;
int arr[MAXN];

void solve() {
    cin >> N;
    int M = N / 3;
    int ans = 0;
    int p = 1;
    for (int i = 1; i <= min(M, 8LL); i++) {
        p = p * 2LL % MOD;
        ans = (ans + p) % MOD;
    }
    M -= 9;
    if (M >= 0) {
        int cnt = M * inv(MAXN, MOD) % MOD;
        ans = (ans + cnt * arrSum % MOD) % MOD;
        int rem = M % MAXN;
        for (int i = 0; i <= rem; i++) {
            ans = (ans + arr[i]) % MOD;
        }
    }
    cout << ans << endl;
}

signed main() {
    arr[0] = 512;
    arrSum = 512;
    for (int i = 1; i < MAXN; i++) {
        arr[i] = arr[i - 1] * 2ll % MOD;
        arrSum = (arrSum + arr[i]) % MOD;
    }
    int T;
    cin >> T;
    while (T--) solve();
    return 0;
}
```

##

### Solution 1: 

```cpp
const string vowels = "aeiou", consonants = "mnptkswjl";
string S;
int N;
const vector<string> illegalPairs = {"wu", "wo", "ji", "ti", "nn", "nm"};

bool isVowel(char ch) {
    return vowels.find(ch) != string::npos;
}

bool isConsonant(char ch) {
    return consonants.find(ch) != string::npos;
}

bool validEnding(char ch) {
    return N > 1 && isVowel(S.end()[-2]);
}

void solve() {
    cin >> S;
    N = S.size();
    bool ans = true;
    for (const string& p : illegalPairs) {
        if (S.find(p) != string::npos) {
            ans = false;
            break;
        }
    }
    for (int i = 0; i < N; i++) {
        if (i > 0 && isVowel(S[i - 1]) && isVowel(S[i])) {
            ans = false;
            break;
        }
        if (!isVowel(S[i]) && !isConsonant(S[i])) {
            ans = false;
            break;
        }
    }
    if (N > 1 && isConsonant(S.end()[-1]) && isConsonant(S.end()[-2])) {
        ans = false;
    } else if (N == 1 && S.back() == 'n') {
        ans = false;
    } else if (S.back() != 'n' && isConsonant(S.back())) {
        ans = false;
    }
    if (ans) {
        cout << "pona" << endl;
    } else {
        cout << "ike" << endl;
    }
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    int T;
    cin >> T;
    while (T--) solve();
    return 0;
}
```

## Problem 6: Big Ben’s Big Brain Bamboozles a Bovine

### Solution 1:  divide and conquer

1. This might be similar to josephus problem, check that out. 
1. This is a tricky divide and conquer problem, mostly cause you have to figure out how to handle when K = 1 and N is odd.  

```cpp
int N, K;

int ceil(int x, int y) {
    return (x + y - 1) / y;
}

int recurse(int n, int k) {
    if (k % 2 == 0) return k / 2;
    if (n % 2 == 0) {
        return n / 2 + recurse(n / 2, ceil(k, 2));
    } else {
        if (k == 1) return ceil(n, 2);
        return ceil(n, 2) + recurse(n / 2, k / 2);
    }
}

void solve() {
    cin >> N >> K;
    int ans = recurse(N, K);
    cout << ans << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    int T;
    cin >> T;
    while (T--) solve();
    return 0;
}
```

# calico fall 2024

## Problem 5: Better Call McKirby

### Solution 1:  greedy binary search

This is on the right path but it gives WA

```cpp
const int INF = 1e18;
int N, B, D;
vector<int> heights;

int calc(int target) {
    int cost = 0;
    for (int x : heights) {
        cost += max(0LL, x - target);
    }
    return cost;
}

int calcDanger(int target) {
    int danger = 0;
    for (int x : heights) {
        danger += max(0LL, target - x);
    }
    return danger;
}

void solve() {
    cin >> B >> N;
    heights.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> heights[i];
    }
    int lo = 0, hi = INF;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (calc(mid) > B) lo = mid + 1;
        else hi = mid;
    }
    D = calcDanger(lo);
    hi = INF;
    while (lo < hi) {
        int mid = lo + (hi - lo + 1) / 2;
        if (calcDanger(mid) > D) hi = mid - 1;
        else lo = mid;
    }
    cout << lo << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    int T;
    cin >> T;
    while (T--) solve();
    return 0;
}
```

## Problem 6: Big Brother Ben

### Solution 1:  totient function, greatest common divisor, prefix sum, primes, sieve of Eratosthenes, principle of inclusion exclusion

```cpp
const int MAXN = 60'000;
int N;
int totient[MAXN], totient_sum[MAXN];
vector<int> primes[MAXN];

void sieve(int n) {
    iota(totient, totient + n, 0LL);
    memset(totient_sum, 0, sizeof(totient_sum));
    for (int i = 2; i < n; i++) {
        if (totient[i] == i) { // i is prime integer
            for (int j = i; j < n; j += i) {
                totient[j] -= totient[j] / i;
                primes[j].emplace_back(i);
            }
        }   
    }
    for (int i = 2; i < n; i++) {
        totient_sum[i] = totient_sum[i - 1] + totient[i];
    }
}

// count number of coprimes of x to m
int num_coprimes(int x, int m) {
    int ans = x;
    int endMask = 1 << primes[m].size();
    for (int mask = 1; mask < endMask; mask++) {
        int numBits = 0, val = 1;
        for (int i = 0; i < primes[m].size(); i++) {
            if ((mask >> i) & 1) {
                numBits++;
                val *= primes[m][i];
            }
        }
        if (numBits % 2 == 0) { // even add
            ans += x / val;
        } else {
            ans -= x / val;
        }
    }
    return ans;
}

void solve() {
    cin >> N;
    int lo = 0, hi = MAXN;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (totient_sum[mid] < N) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    N -= totient_sum[lo - 1];
    int M = lo;
    lo = 1, hi = M;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        int cnt = num_coprimes(mid, M);
        if (cnt < N) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    cout << lo << " " << M - lo << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    sieve(MAXN);
    int T;
    cin >> T;
    while (T--) solve();
    return 0;
}
```

## Problem 7: Nobody Jumps for the Rank 3

### Solution 1:  connected components, union find algorithm, grid, map, sorting, merging components

```cpp
int R, C;
vector<vector<int>> grid;

struct UnionFind {
    vector<int> parents, size;
    void init(int n) {
        parents.resize(n);
        iota(parents.begin(),parents.end(),0);
        size.assign(n,1);
    }

    int find(int i) {
        if (i==parents[i]) {
            return i;
        }
        return parents[i]=find(parents[i]);
    }

    bool same(int i, int j) {
        i = find(i), j = find(j);
        if (i!=j) {
            if (size[j]>size[i]) {
                swap(i,j);
            }
            size[i]+=size[j];
            parents[j]=i;
            return false;
        }
        return true;
    }
};

vector<pair<int, int>> neighborhood(int r, int c) {
    return {{r + 1, c}, {r - 1, c}, {r, c + 1}, {r, c - 1}};
}

bool in_bounds(int r, int c) {
    return r >= 0 && r < R && c >= 0 && c < C;
}

int map2DTo1D(int r, int c) {
    return r * C + c;
}

pair<int, int> map1DTo2D(int v) {
    return {v / C, v % C};
}

void solve() {
    cin >> R >> C;
    UnionFind dsu;
    dsu.init(R * C);
    map<int, vector<pair<int, int>>, greater<int>> pairs;
    vector<bool> vis(R * C, false);
    grid.assign(R, vector<int>(C, 0));
    for (int i = 0;i < R; i++) {
        for (int j = 0; j < C; j++) {
            cin >> grid[i][j];
            pairs[grid[i][j]].emplace_back(i, j);
        }
    }
    int islands = 0, ans = 0;
    for (const auto &[k, vec] : pairs) {
        for (const auto &[r, c] : vec) {
            int val = map2DTo1D(r, c);
            if (vis[val]) continue;
            set<pair<int, int>> cellsOfEqualHeight;
            queue<pair<int, int>> q;
            q.emplace(r, c);
            vis[val] = true;
            while (!q.empty()) {
                auto [i, j] = q.front();
                cellsOfEqualHeight.insert({i, j});
                q.pop();
                for (const auto &[nr, nc] : neighborhood(i, j)) {
                    int nv = map2DTo1D(nr, nc);
                    if (!in_bounds(nr, nc) || vis[nv] || grid[nr][nc] != k) continue;
                    q.emplace(nr, nc);
                    vis[nv] = true;
                }
            }
            unordered_set<int> neighbors;
            for (const auto &[i, j] : cellsOfEqualHeight) {
                for (const auto &[nr, nc] : neighborhood(i, j)) {
                    if (!in_bounds(nr, nc) || cellsOfEqualHeight.count({nr, nc}) || grid[nr][nc] <= k) continue;
                    neighbors.insert(dsu.find(map2DTo1D(nr, nc)));
                }
            }
            for (const auto &[i, j] : cellsOfEqualHeight) {
                for (const auto &[nr, nc] : neighborhood(i, j)) {
                    if (!in_bounds(nr, nc) || grid[nr][nc] < k) continue;
                    dsu.same(map2DTo1D(nr, nc), map2DTo1D(i, j));
                }
            }
            int delta = 1 - neighbors.size();
            islands += delta;
        }
        ans = max(ans, islands);
    }
    cout << ans << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    int T;
    cin >> T;
    while (T--) solve();
    return 0;
}
```

## Problem 8: Into the CALICOre

### Solution 1:  bfs, queue, grid, jumping, directed graph

```cpp
const char WALL = '#', CRYSTAL = '*', START = 'S', END = 'E';
const int INF = 1e9, BLUE = 0, RED = 1;
int R, C, K;
vector<vector<char>> grid;
vector<vector<pair<int, bool>>> adj;
vector<bool> vis[2];

vector<pair<int, int>> neighborhood(int r, int c) {
    return {{r + 1, c}, {r - 1, c}, {r, c + 1}, {r, c - 1}};
}

bool in_bounds(int r, int c) {
    return r >= 0 && r < R && c >= 0 && c < C;
}

int map2DTo1D(int r, int c) {
    return r * C + c;
}

pair<int, int> map1DTo2D(int v) {
    return {v / C, v % C};
}

struct State {
    int r, c, hairColor;
    State() {}
    State(int r, int c, int hairColor) : r(r), c(c), hairColor(hairColor) {}
};

void solve() {
    cin >> R >> C >> K;
    for (int i = 0; i < 2; i++) {
        vis[i].assign(R * C, false);
    }
    grid.assign(R, vector<char>(C));
    adj.assign(R * C, vector<pair<int, bool>>());
    queue<State> q;
    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C; c++) {
            cin >> grid[r][c];
            if (grid[r][c] == START) {
                q.emplace(r, c, RED);
                vis[RED][map2DTo1D(r, c)] = true;
            }
        }   
    }
    // dashing horizontal
    for (int r = 0; r < R; r++) {
        int lastCrystal = -1;
        for (int c = 0, pos = 0; c < C; c++) {
            if (grid[r][c] == WALL) {
                pos = c + 1;
                continue;
            }
            pos = max(pos, c - K);
            if (c > pos) {
                int u = map2DTo1D(r, c);
                int v = map2DTo1D(r, pos);
                bool hasCrystal = lastCrystal >= pos;
                adj[u].emplace_back(v, hasCrystal);
            }
            if (grid[r][c] == CRYSTAL) lastCrystal = c;
        }
        lastCrystal = C + 1;
        for (int c = C - 1, pos = C - 1; c >= 0; c--) {
            if (grid[r][c] == WALL) {
                pos = c - 1;
                continue;
            }
            pos = min(pos, c + K);
            if (c < pos) {
                int u = map2DTo1D(r, c);
                int v = map2DTo1D(r, pos);
                bool hasCrystal = lastCrystal <= pos;
                adj[u].emplace_back(v, hasCrystal);
            }
            if (grid[r][c] == CRYSTAL) lastCrystal = c;
        }
    }
    // dashing vertical
    for (int c = 0; c < C; c++) {
        int lastCrystal = -1;
        for (int r = 0, pos = 0; r < R; r++) {
            if (grid[r][c] == WALL) {
                pos = r + 1;
                continue;
            }
            pos = max(pos, r - K);
            if (r > pos) {
                int u = map2DTo1D(r, c);
                int v = map2DTo1D(pos, c);
                bool hasCrystal = lastCrystal >= pos;
                adj[u].emplace_back(v, hasCrystal);
            }
            if (grid[r][c] == CRYSTAL) lastCrystal = r;
        }
        lastCrystal = R + 1;
        for (int r = R - 1, pos = R - 1; r >= 0; r--) {
            if (grid[r][c] == WALL) {
                pos = r - 1;
                continue;
            }
            pos = min(pos, r + K);
            if (r < pos) {
                int u = map2DTo1D(r, c);
                int v = map2DTo1D(pos, c);
                bool hasCrystal = lastCrystal <= pos;
                adj[u].emplace_back(v, hasCrystal);
            }
            if (grid[r][c] == CRYSTAL) lastCrystal = r;
        }
    }
    int ans = 0;
    while (!q.empty()) {
        int N = q.size();
        for (int s = 0; s < N; s++) {
            const auto &[r, c, col] = q.front();
            q.pop();
            if (grid[r][c] == END) {
                cout << ans << endl;
                return;
            }
            for (const auto &[nr, nc] : neighborhood(r, c)) {
                if (!in_bounds(nr, nc)) continue;
                int nv = map2DTo1D(nr, nc);
                int ncol = grid[nr][nc] == CRYSTAL ? RED : col;
                if (grid[nr][nc] == WALL || vis[ncol][nv]) continue;
                q.emplace(nr, nc, ncol);
                vis[ncol][nv] = true;
            }
            if (col == BLUE) continue;
            int u = map2DTo1D(r, c);
            for (const auto &[v, w] : adj[u]) {
                int ncol = w ? RED : BLUE;
                if (vis[ncol][v]) continue;
                const auto &[nr, nc] = map1DTo2D(v);
                q.emplace(nr, nc, ncol);
                vis[ncol][v] = true;
            }
        }
        ans++;
    }
    cout << -1 << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    int T;
    cin >> T;
    while (T--) solve();
    return 0;
}
```

## Problem 9: Dijkstra in Dungeon

### Solution 1:  

```cpp

```

# CALICO Spring 2025

## The Fault in Our Bricks

### Solution 1: min and max, area of rectangle

```cpp
int N;
vector<pair<long double, long double>> points;

void solve() {
    cin >> N;
    points.clear();
    long double minX = 1e5, maxX = -1e5, minY = 1e5, maxY = -1e5;
    for (int i = 0; i < N; i++) {
        long double x, y;
        cin >> x >> y;
        minX = min(minX, x);
        maxX = max(maxX, x);
        minY = min(minY, y);
        maxY = max(maxY, y);
        points.emplace_back(x, y);
    }
    long double ans = (maxX - minX) * (maxY - minY);
    cout << fixed << setprecision(10) << ans << endl;
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

## Sussy Smooth Brain Tasking

### Solution 1: manhattan distance with wrap around on grid

```cpp
int R, C;

void solve() {
    cin >> R >> C;
    vector<pair<int, int>> points(R * C + 1, {0, 0});
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            int x;
            cin >> x;
            points[x] = {i, j};
        }
    }
    int ans = 0;
    for (int label = 1; label <= R * C; label++) {
        auto [r1, c1] = points[label - 1];
        auto [r2, c2] = points[label];
        int dr = min(abs(r1 - r2), R - abs(r1 - r2));
        int dc = min(abs(c1 - c2), C - abs(c1 - c2));
        ans += dr + dc;
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

## Miku in the Middle

### Solution 1: prefix sum, counting

```cpp
string S;

void solve() {
    cin >> S;
    int64 ans = 0, cnt = 0, cur = 0;
    for (char c : S) {
        if (c == 'u') {
            ans += cnt;
            cur++;
        } else if (c == 'w') {
            cnt += cur;
            cur = 0;
        }
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

## Indomitable Human Spirit vs AAA Company

### Solution 1: hmmm

```cpp

```

## Is a configuration of blocks stable?

### Solution 1:  center of mass, tree, subtree calculations, postorder dfs

1. know how center of mass calculation works, and how you can do it separate for the 2 coordinates, xcm and ycm
1. So you can accumulate the center of mass for each coordinates and the total mass of each subtree
1. Then you just have to check that the center of mass xcm is within the range x_r and x_l of it's parent node, cause that means it is stable
1. If you find it is outside of the range it is unstable. 

```cpp

```

## Ben in the Middle

### Solution 1: 

```cpp

```

## Gates to Infinite Winnings :money_mouth:

### Solution 1: 

```cpp

```

## PokéRogue Daily Run

### Solution 1: 

1. How do you solve this, some kind of dp but idk

```cpp

```

# CALICO Fall 2025

## Problem 8: Put the Fries in the Bag

### Solution 1: 

1. Never count | more than once, so you recurse, and the | in the base L get pushed farther and farther in the layers. 
1. There will always be a parentheses if there is a O because logically it could recurse infinitely. 
1. You can do recursion and build out the number of depth you add for each recursion.  So treat each O as a seed point. 
1. And start with (0, 0, 0) if you have 3 Os, then recurse out with a branching factor of 3 each time. and add whatever the depth is (number of open parentheses) for each branch. 
1. But the problem with this approach is recursing 10^9 times is too slow.
1. But if you were able to do this and calculate the count of each depth, then just loop over the S, and take the | and there base depth, and just add to that and check the depth is under N.

```cpp

```

## Problem 10: “more broke than a. . . ”

### Solution 1: 

1. I think you are only sending how much you owe to a specific person.
1. The uniform at random part is confusing at first, but ultimately you have to send out the total of how much you owe to each person, so you must somehow have an cash flow of that much into your node.
1. With these patterns the key is to count how much is owed to you and how much you owe out.  Because I will need to see the difference between these two values. 
1. so one thing to realize is any node that indegree 0, that means you have no incoming money, so you have to give this node how much money they owe everybody they owe.  so a sum over the owed amount to other nodes. 
1. Then there is possible you do have some indegree > 0 and outdegree > 0.  In that case you need out(u) - in(u), so if you have to give out more that means will need some extra dollars. 
1. There is one special edge case if you find an SCC, where the sum of out(u) - in(u) == 0, but there is at least one internal edge, and no incoming edges from other SCCs, then you need to inject 1 dollar to prime the cycle.

```cpp

```

## Problem 11: New Tournament

### Solution 1: 

1. N is up to 2^15, this means there will be at most 15 rounds each tournament.
1. The hard part is figuring out the starting position of everyone after possibly 10^5 rotations. 
1. functional graph that maps each index to where they move to after one rotation, and then use this functional graph with binary jumping to find where each index ends up after K rotations.
1. So this can be done for each player N in logK time, but the problem because you have 10^4 queries. 
1. O(NQlogK) 
1. player 1 and 2 are rather interesting, because in the final round the power of the last two players might be 1 and 2, not necessarily, but it will always be 1. 
1. Given the current position of each player, You can run a recursive algorithm that halves at each step, and so runs 15 times. Where you take an array query for the smallest player in that range, and then determine if they are on the left or right half if you split that range. 
1. Then you'd take the side they are not on and query for the smallest player in that side, and then continue the process. 
1. This would take probably log(N)^2 time?

After any match between a and b, the survivor’s power becomes min(a,b). Inductively, for any node of the fixed knockout bracket (size = power of two, aligned as (1,2), (3,4), ...), the survivor’s power is the minimum of that segment, and the side that advances to the next round is the side whose segment-minimum is larger.

At the root, compare the minima of the left and right halves. The global minimum sits in exactly one half; the other half advances. Repeating this at each level means you always “dodge” the global minimum.

So, going from the root down to the leaves, the champion always takes the opposite child compared to the child that holds m. In a complete binary tree over positions 0..N-1, choosing the opposite child at every one of the r splits is exactly “flip every one of the r bits” of m.

For the full problem with shuffles P^K:

Precompute permutation cycles of P and P^{-1} so you can map any player x to its position after S total shuffles in O(1) using cycle offsets.

To answer a query with cumulative S:

Start at the root. For the current node’s two equal halves, find each half’s minimum label after S shuffles. You can do this by checking labels in increasing order and stopping at the first one whose position at time S lands inside the half. With random inputs, the expected checks per half are constant and you only do this for log2 N levels.

Walk to the child with larger minimum and repeat until a leaf.

You now have the winner’s position; use the P^{-S} cycle to get which original player sits there and output that player.

```cpp

```