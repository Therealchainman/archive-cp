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