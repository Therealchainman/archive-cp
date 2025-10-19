# Meta Hacker Cup 2024

# Practice Round

## Problem A: Walk the Line

### Solution 1: greedy

1. Always take the fastest person to help each other person across the line.

```cpp
// string name = "walk_the_line_sample_input.txt";
// string name = "walk_the_line_validation_input.txt";
string name = "walk_the_line_input.txt";
const int INF = 1e18;
int N, K, x;

string solve() {
    int best = INF;
    cin >> N >> K;
    for (int i = 0; i < N; i++) {
        cin >> x;
        best = min(best, x);
    }
    int ans = max(best, (2 * N - 3) * best);
    if (ans <= K) return "YES";
    return "NO";
}

signed main() {
	ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    string in = "inputs/" + name;
    string out = "outputs/" + name;
    freopen(in.c_str(), "r", stdin);
    freopen(out.c_str(), "w", stdout);
    int T;
    cin >> T;
    for (int i = 1; i <= T ; i++) {
        cout << "Case #" << i << ": " << solve() << endl;
    }
    return 0;
}
```

## Problem B: Line by Line

### Solution 1:  math, probability

1. Calculate the probability of having N - 1 correct lines, need to convert into decimal form and not percentage so that it is correct.
2. Then that is the target probability with N lines, so just take the Nth root to get what is the necessary probability to multiple to it.  Cause you have to multiple by this value N times to get the target probability.
3. Then convert back to percentage and subtract the original percentage to get the difference.

```cpp
// string name = "line_by_line_sample_input.txt";
// string name = "line_by_line_validation_input.txt";
string name = "line_by_line_input.txt";
const int INF = 1e18;
int N, P;

long double solve() {
    cin >> N >> P;
    long double p = P / 100.0;
    long double prob1 = pow(p, N - 1);
    long double ans = pow(prob1, 1 / (long double)N) * 100.0;
    return ans - P;
}

signed main() {
	ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    string in = "inputs/" + name;
    string out = "outputs/" + name;
    freopen(in.c_str(), "r", stdin);
    freopen(out.c_str(), "w", stdout);
    int T;
    cin >> T;
    for (int i = 1; i <= T ; i++) {
        cout << "Case #" << i << ": " << fixed << setprecision(15) << solve() << endl;
    }
    return 0;
}
```

## Problem C: Fall in Line

### Solution 1:

1. It was probability problem.
1. If there were more than N / 2 points along the optimal path, it turns out the probability of picking two points at random and having them be on the optimal path is less than 1/4.  So the probability of picking two points no on optimal path is greater than 3/4.  So if you keep picking two points (3/4)^K, if you pick K times this is the probability that it fails K times.  The probability will get very low because the number is less than 1.  And therefore you will be guaranteed to have picked two points along the optimal path.
1. If the optimal path is under half it doesn't really matter you can just return N.

```cpp
string base = "fall_in_line";
// string name = base + "_sample_input.txt";
// string name = base + "_validation_input.txt";
string name = base + "_input.txt";

random_device rd;
mt19937_64 gen(rd());
int randint(int l, int r) {
    uniform_int_distribution<int> dist(l, r);
    return dist(gen);
}

const int INF = 1e9;
int N;
vector<pair<int, int>> points;

int outer_product(const pair<int, int>& v1, const pair<int, int>& v2) {
    return v1.x * v2.y - v1.y * v2.x;
}

// calculate number of points non-collinear with points p1 and p2.
int calc(const pair<int, int>& p1, const pair<int, int>& p2) {
    int ans = 0;
    for (int i = 0; i < N; i++) {
        pair<int, int> v1 = {points[i].x - p1.x, points[i].y - p1.y};
        pair<int, int> v2 = {points[i].x - p2.x, points[i].y - p2.y};
        if (outer_product(v1, v2) != 0) ans++;
    }
    return ans;
}

void solve() {
    cin >> N;
    points.resize(N);
    for (int i = 0; i < N; i++) {
        int x, y;
        cin >> x >> y;
        points[i] = {x, y};
    }
    int ans = INF;
    for (int i = 0; i < 100; i++) {
        int u = 0, v = 0;
        while (u == v) {
            u = randint(0, N - 1);
            v = randint(0, N - 1);
        }
        ans = min(ans, calc(points[u], points[v]));
    }
    cout << ans << endl;
}

signed main() {
	ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    string in = "inputs/" + name;
    string out = "outputs/" + name;
    freopen(in.c_str(), "r", stdin);
    freopen(out.c_str(), "w", stdout);
    int T;
    cin >> T;
    for (int i = 1; i <= T ; i++) {
        cout << "Case #" << i << ": ";
        solve();
    }
    return 0;
}
```

## Problem D1: Line of Delivery (Part 1)

### Solution 1:  fenwick tree, find the closest number to G and the distance to it

1. I didn't realize it at first but all the energies are unique, so basically the collisions are very simple to model
2. So it is easy to find the closest it gets to the target from the left of the goal and the right of the goal.
3. One of these is the answer, or both of them are.  But how do you find the index of the stone that is the closest to the goal?
4. I observed that the index of the stones is sorted, that is the leftmost stone is the last thrown and the rightmost stone is the first thrown.
5. So really if you just know how many stones are to at the given position, you know it is the N - count of stones at that position.
6. For example if you have N = 10, and count of stones = 3, that means it is 3rd from the last thrown stones, so that would be 8th stone thrown that reaches that position.

```cpp
string base = "line_of_delivery_part_1";
// string name = base + "_sample_input.txt";
// string name = base + "_validation_input.txt";
string name = base + "_input.txt";
const int INF = 1e18, MAXN = 1e6 + 5;
int N, G;
struct FenwickTree {
    vector<int> nodes;
    int neutral = 0;

    void init(int n) {
        nodes.assign(n + 1, neutral);
    }

    void update(int idx, int val) {
        while (idx < (int)nodes.size()) {
            nodes[idx] += val;
            idx += (idx & -idx);
        }
    }

    int query(int left, int right) {
        return right >= left ? query(right) - query(left - 1) : 0;
    }

    int query(int idx) {
        int result = neutral;
        while (idx > 0) {
            result += nodes[idx];
            idx -= (idx & -idx);
        }
        return result;
    }
};

FenwickTree ft;

void solve() {
    cin >> N >> G;
    int left_g = -INF, right_g = INF;
    unordered_multiset<int> nums;
    for (int i = 0; i < N; i++) {
        int x;
        cin >> x;
        nums.insert(x);
        if (x < G) {
            left_g = max(left_g, x);
        } else {
            right_g = min(right_g, x);
        }
        ft.update(x, 1);
    }
    int ans = INF, mini = INF;
    if (left_g != -INF) {
        ans = min(ans, G - left_g);
    }
    if (right_g != INF) {
        ans = min(ans, right_g - G);
    }
    if (G - left_g == ans) {
        int idx = N - ft.query(left_g);
        mini = min(mini, idx);
    }
    if (right_g - G == ans) {
        int idx = N - ft.query(right_g);
        mini = min(mini, idx);
    }
    cout << mini + 1 << " " << ans << endl;
    for (int x : nums) {
        ft.update(x, -1);
    }
}

signed main() {
	ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    string in = "inputs/" + name;
    string out = "outputs/" + name;
    freopen(in.c_str(), "r", stdin);
    freopen(out.c_str(), "w", stdout);
    ft.init(MAXN);
    int T;
    cin >> T;
    for (int i = 1; i <= T ; i++) {
        cout << "Case #" << i << ": ";
        solve();
    }
    return 0;
}
```

## Problem D2: Line of Delivery (Part 2)

### Solution 1:

1. This one looks more challenging because now the stones have unit width, that is they take up spots, and you can have duplicate energies now.  This time a stone will stop at position just in front of the next it collides with and transfers it's energy.
2. It is a little more complex than part 1
now if you have 2 balls before position 9, and you throw a ball with strength 9, it will send the last ball to position 11, because the balls take up space.  So that means it will go 9 + # of balls before it.
But it is complicated because what of the other balls between.

I get feeling it may involve stack.

```cpp

```

# Round 1

## Problem A: Subsonic Subway

### Solution 1:

```cpp
string base = "subsonic_subway";
// string name = base + "_sample_input.txt";
// string name = base + "_validation_input.txt";
string name = base + "_input.txt";

const long double INF = 1e18;
int N;
vector<pair<int, int>> stations;

void solve() {
    cin >> N;
    stations.resize(N);
    long double lower = 0.0, upper = INF;
    for (int i = 0; i < N; i++) {
        int l, r;
        cin >> l >> r;
        stations[i] = {l, r};
        long double min_speed = (long double) (i + 1) / r;
        long double max_speed = (long double) (i + 1) / l;
        lower = max(lower, min_speed);
        upper = min(upper, max_speed);
    }
    if (lower > upper) {
        cout << -1 << endl;
    } else {
        cout << fixed << setprecision(15) << lower << endl;
    }
}

signed main() {
	ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    string in = "inputs/" + name;
    string out = "outputs/" + name;
    freopen(in.c_str(), "r", stdin);
    freopen(out.c_str(), "w", stdout);
    int T;
    cin >> T;
    for (int i = 1; i <= T ; i++) {
        cout << "Case #" << i << ": ";
        solve();
    }
    return 0;
}
```

## Problem B: Prime Subtractorization

### Solution 1:

```cpp
string base = "prime_subtractorization";
// string name = base + "_sample_input.txt";
// string name = base + "_validation_input.txt";
string name = base + "_input.txt";


const int MAXN = 1e7 + 5;
int primes[MAXN], N, ans, dp[MAXN];

void sieve() {
    fill(primes, primes + MAXN, 1);
    primes[0] = primes[1] = 0;
    int p = 2;
    for (int p = 2; p * p <= MAXN; p++) {
        if (primes[p]) {
            for (int i = p * p; i < MAXN; i += p) {
                primes[i] = 0;
            }
        }
    }
}

void solve() {
    cin >> N;
    if (dp[N]) {
        cout << dp[N] + 1 << endl;
    } else {
        cout << dp[N] << endl;
    }
}

signed main() {
	ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    string in = "inputs/" + name;
    string out = "outputs/" + name;
    freopen(in.c_str(), "r", stdin);
    freopen(out.c_str(), "w", stdout);
    sieve();
    memset(dp, 0, sizeof(dp));
    for (int i = 5; i < MAXN; i++) {
        dp[i] = dp[i - 1];
        if (primes[i]) dp[i] += primes[i - 2];
    }
    int T;
    cin >> T;
    for (int i = 1; i <= T ; i++) {
        cout << "Case #" << i << ": ";
        solve();
    }
    return 0;
}
```

## Problem C: Substantial Losses

### Solution 1:

```py

sys.stdout = open(f"outputs/{name}", "w")
sys.stdin = open(f"inputs/{name}", "r")
M = 998244353
def main():
    W, G, L = map(int, input().split())
    v = (2 * L + 1) % M
    ans = ((W - G) * v) % M
    return ans

if __name__ == '__main__':
    T = int(input())
    for t in range(1, T + 1):
        print(f"Case #{t}: {main()}")
```

## Problem D: Substitution Cipher

### Solution 1:

```cpp

```

## Problem E: Wildcard Submissions

### Solution 1:

longest common prefix over subsets

compute minimum pairwise value for all subsets

It works because wildcard LCP is a pairwise bottleneck: the set’s first break is always witnessed by some pair, so the answer for S is the minimum over three buckets of pairs, which you get using {p1,p2}, S \ {p1}, and S \ {p2}.

recurrence relation, just need three pairs

You've calculated all these triplets by taking lcp of pairs, (1,2,3), (1,2,4),(1,3,4), (2,3,4) so then we can pick any two p1 and p2, such as (1, 3) then we have without p1, (2,3,4) and without p2, (1,2,4). Since I already agree that the lcp is corectly calculated in (2,3,4) and (1,2,4) triplets, then it only comes down the minimum of them, cause the 3 with the 2 and 4 might be 5, and 1 with 2 and 4 might be 6, so that means they are limited by the 3, but then the lcp(1,3) might be 4, so that limits it even further so the lcp of (1,2,3,4) is 4

This graph analogy really helped me understand the lcp dp.

1) Graph picture
Calculating minimum edge over all subgraphs
Think of each string as a node. Every unordered pair is an edge labeled with its pairwise LCP. For a subset S, LCP(S) is the smallest edge label in the subgraph induced by S. Pick p1 = 1 and p2 = 3 in S = {1,2,3,4}. The edges inside S split like this:
- The single edge between the two chosen nodes: {1,3}.
- All edges that do not touch node 1. Those are exactly the edges inside the induced subgraph on {2,3,4}.
- All edges that do not touch node 3. Those are exactly the edges inside the induced subgraph on {1,2,4}.
So the minimum edge in the whole S-induced subgraph equals
min( label({1,3}), min edge in {2,3,4}, min edge in {1,2,4} ).

Think of it as reusing the already known minimum inside smaller induced subgraphs to get the minimum for the bigger one.

```cpp

```

# Round 2

## Problem A1: Cottontail Climb (Part 1)

### Solution 1:  brute force, precalculate all possible values

1. There were only 45 possible values, so not very many.

```cpp
string base = "cottontail_climb_part_1";
// string name = base + "_sample_input.txt";
// string name = base + "_validation_input.txt";
string name = base + "_input.txt";

int A, B, M;
vector<int> values;

int calc(int x, int m) {
    int ans = 0;
    for (int v : values) {
        if (v > x) continue;
        if (v % m == 0) ans++;
    }
    return ans;
}

void solve() {
    cin >> A >> B >> M;
    int ans = calc(B, M);
    if (A > 0) ans -= calc(A - 1, M);
    cout << ans << endl;
}

signed main() {
	ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    string in = "inputs/" + name;
    string out = "outputs/" + name;
    freopen(in.c_str(), "r", stdin);
    freopen(out.c_str(), "w", stdout);
    for (int i = 1; i < 10; i++) {
        for (int j = 1; j < 10; j++) {
            int val = 0;
            for (int k = i; k < j; k++) {
                val = (val * 10) + k;
            }
            for (int k = j; k >= i; k--) {
                val = (val * 10) + k;
            }
            if (!val) continue;
            values.emplace_back(val);
        }
    }
    int T;
    cin >> T;
    for (int i = 1; i <= T ; i++) {
        cout << "Case #" << i << ": ";
        solve();
        cout.flush();
    }
    return 0;
}
```

## Problem A2: Cottontail Climb (Part 2)

### Solution 1:  digit dp, recursive

1.  Appears you didn't need digit dp for this problem.

```cpp
string base = "cottontail_climb_part_2";
// string name = base + "_sample_input.txt";
// string name = base + "_validation_input.txt";
string name = base + "_input.txt";

int A, B, M;
map<int, int> dp[10][19][10][2];

int convert(char x, char base = '0') {
    return x - base;
}

int dfs(int p, int i, int rem, int last, int tight, const string& num) {
    if (dp[p][i][last][tight].count(rem)) return dp[p][i][last][tight][rem];
    if (i > 2 * p) return rem == 0;
    if (i == num.size()) return 0;
    int ans = 0;
    for (int d = 1; d < 10; d++) {
        if (tight && d > convert(num[i])) break;
        if (i < p && d < last) continue;
        if (i == p + 1 && d >= last) continue;
        if (i > p && d > last) break;
        if (i == p && d <= last) continue;
        int nrem = (rem * 10 + d) % M;
        ans += dfs(p, i + 1, nrem, d, tight && d == convert(num[i]), num);
    }
    return dp[p][i][last][tight][rem] = ans;
}

void solve() {
    cin >> A >> B >> M;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 19; j++) {
            for (int k = 0; k < 10; k++) {
                for (int l = 0; l < 2; l++) {
                    dp[i][j][k][l].clear();
                }
            }
        }
    }
    string s1 = to_string(B);
    int ans = 0;
    for (int i = 0; i < 10; i++) {
        if (2 * i + 1 < s1.size()) ans += dfs(i, 0, 0, 0, 0, s1);
        else ans += dfs(i, 0, 0, 0, 1, s1);
    }
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 19; j++) {
            for (int k = 0; k < 10; k++) {
                for (int l = 0; l < 2; l++) {
                    dp[i][j][k][l].clear();
                }
            }
        }
    }
    if (A > 0) {
        s1 = to_string(A - 1);
        for (int i = 0; i < 10; i++) {
            if (2 * i + 1 < s1.size()) ans -= dfs(i, 0, 0, 0, 0, s1);
            else ans -= dfs(i, 0, 0, 0, 1, s1);
        }
    }
    cout << ans << endl;
}

signed main() {
	ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    string in = "inputs/" + name;
    string out = "outputs/" + name;
    freopen(in.c_str(), "r", stdin);
    freopen(out.c_str(), "w", stdout);
    int T;
    cin >> T;
    for (int i = 1; i <= T ; i++) {
        cout << "Case #" << i << ": ";
        solve();
        cout.flush();
    }
    return 0;
}
```

## Problem D: Four in a Burrow

### Solution 1: dfs with backtracking, recursion, height state, memoization

```cpp
int R = 6, C = 7;
vector<string> board;
vector<int> heights;
map<vector<int>, char> memo;

bool checkVertical(int col, const vector<int> &heights) {
    int h = heights[col];
    char p = board[h][col];
    int cnt = 0;
    for (int r = h; r >= 0; --r) {
        if (board[r][col] != p) break;
        cnt++;
    }
    return cnt >= 4;
}

bool checkHorizontal(int col, const vector<int> &heights) {
    int h = heights[col];
    char p = board[h][col];
    int cnt = 1;
    for (int c = col - 1; c >= 0; --c) {
        if (heights[c] <= h) break;
        if (board[h][c] != p) break;
        cnt++;
    }
    for (int c = col + 1; c < C; ++c) {
        if (heights[c] <= h) break;
        if (board[h][c] != p) break;
        cnt++;
    }
    return cnt >= 4;
}

bool checkDiagonal(int col, const vector<int> &heights) {
    int h = heights[col];
    char p = board[h][col];
    int cnt = 1;
    for (int r = h - 1, c = col - 1; r >= 0 && c >= 0; --r, --c) {
        if (heights[c] <= r) break;
        if (board[r][c] != p) break;
        cnt++;
    }
    for (int r = h + 1, c = col + 1; r < R && c < C; ++r, ++c) {
        if (heights[c] <= r) break;
        if (board[r][c] != p) break;
        cnt++;
    }
    if (cnt >= 4) return true;
    cnt = 1;
    for (int r = h - 1, c = col + 1; r >= 0 && c < C; --r, ++c) {
        if (heights[c] <= r) break;
        if (board[r][c] != p) break;
        cnt++;
    }
    for (int r = h + 1, c = col - 1; r < R && c >= 0; ++r, --c) {
        if (heights[c] <= r) break;
        if (board[r][c] != p) break;
        cnt++;
    }
    return cnt >= 4;
}

bool check(int idx, const vector<int> &heights) {
    return checkHorizontal(idx, heights) || checkVertical(idx, heights) || checkDiagonal(idx, heights);
}

char dfs(int idx) {
    if (idx == R * C) return '0';
    if (memo.find(heights) != memo.end()) {
        return memo[heights];
    }
    char turn = idx % 2 == 0 ? 'C' : 'F';
    bool canReach = false, cwin = false, fwin = false;
    for (int i = 0; i < C; ++i) {
        if (heights[i] == R) continue;
        if (board[heights[i]][i] != turn) continue;
        heights[i]++;
        char res = dfs(idx + 1);
        heights[i]--;
        bool canWin = check(i, heights);
        if (canWin && res != 'D') {
            if (turn == 'C') {
                cwin = true;
            } else {
                fwin = true;
            }
            canReach = true;
            continue;
        }
        if (res == 'C') {
            cwin = true;
        } else if (res == 'F') {
            fwin = true;
        } else if (res == '?') {
            cwin = fwin = true;
        }
        if (res != 'D') {
            canReach = true;
        }
    }
    if (!canReach) return memo[heights] = 'D';
    if (cwin && fwin) return memo[heights] = '?';
    if (cwin) return memo[heights] = 'C';
    if (fwin) return memo[heights] = 'F';
    return memo[heights] = '0';
}

void solve() {
    board.clear();
    for (int i = 0; i < R; i++) {
        string row;
        cin >> row;
        board.emplace_back(row);
    }
    reverse(board.begin(), board.end());
    heights.assign(7, 0);
    memo.clear();
    char ans = dfs(0);
    cout << ans << endl;
}
```

## Problem C: Bunny Hopscotch

### Solution 1: 2D BIT, binary search, map

1. Query in rectangle to find the count of cells with same bunny, then take total size of rectangle minus that to get the cells you can jump to that have a different bunny within the distance.

```cpp
int R, C;
int64 K;
unordered_map<int, vector<pair<int, int>>> adj;

struct BIT2D {
    int n;
    vector<vector<int64>> bit;
    BIT2D(int n) : n(n), bit(n + 1, vector<int64>(n + 1, 0)) {}

    void add(int r, int c, int64 delta) {
        for (int i = r; i <= n; i += i & -i) {
            for (int j = c; j <= n; j += j & -j) {
                bit[i][j] += delta;
            }
        }
    }

    int64 sum(int r, int c) const {
        int64 res = 0;
        for (int i = r; i > 0; i -= i & -i) {
            for (int j = c; j > 0; j -= j & -j) {
                res += bit[i][j];
            }
        }
        return res;
    }

    int64 rect(int r1, int c1, int r2, int c2) const {
        if (r1 > r2) swap(r1, r2);
        if (c1 > c2) swap(c1, c2);
        int64 res = sum(r2, c2) - sum(r1 - 1, c2) - sum(r2, c1 - 1) + sum(r1 - 1, c1 - 1);
        return res;
    }
};

bool possible(BIT2D& bit, int target) {
    int64 cnt = 0;
    for (const auto &[_, coords] : adj) {
        for (const auto &[r, c] : coords) {
            bit.add(r + 1, c + 1, 1);
        }
        for (const auto &[r, c] : coords) {
            int r1 = max(1, r - target + 1), r2 = min(r + target + 1, R);
            int c1 = max(1, c - target + 1), c2 = min(c + target + 1, C);
            int64 same = bit.rect(r1, c1, r2, c2);
            int64 total = (r2 - r1 + 1) * (c2 - c1 + 1);
            int64 delta = total - same;
            cnt += delta;
        }
        for (const auto &[r, c] : coords) {
            bit.add(r + 1, c + 1, -1);
        }
    }
    return cnt < K;
}

void solve() {
    cin >> R >> C >> K;
    adj.clear();
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            int x;
            cin >> x;
            adj[x].emplace_back(i, j);
        }
    }
    BIT2D bit(max(R, C) + 1);
    int64 lo = 0, hi = 800;
    while (lo < hi) {
        int64 mid = lo + (hi - lo) / 2;
        if (possible(bit, mid)) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    cout << lo << endl;
}
```

# Round 3

## Problem A: Set, Cover

### Solution 1: area on grid, row, case by case

1. Problem concept
You want the largest possible axis-aligned rectangle that must cover every definite 1 in an N×N grid, while you may convert up to K unknown cells into 1s to stretch that rectangle.
1. Geometric intuition
Think of two boxes. One is the tight box around all current 1s. The other captures where the unknowns sit, globally and by row. Turning an unknown into a 1 lets you “pull” the sides of the 1s box outward toward those unknown positions.
1. Case-based expansion
For each budget K, the algorithm chooses simple patterns that maximally tug the rectangle.
K = 0 keeps the current 1s box.
K = 1 tests adding a single unknown anywhere.
K = 2 uses two rows to pull left and right while also extending up or down.
K = 3 mixes a chosen row with global unknown extremes to pull to different corners.
K ≥ 4 treats the unknown region as fully pullable and covers both boxes together.
1. Outcome and efficiency
It computes the maximum area achievable under the flip budget without brute force. Summaries of rows and global extremes let it evaluate only a small set of rectangle candidates, keeping the work near quadratic at worst rather than exponential.

```cpp
int N, K;
vector<vector<char>> grid;

int area(int r1, int r2, int c1, int c2) {
    return (r2 - r1 + 1) * (c2 - c1 + 1);
}

void solve() {
    cin >> N >> K;
    grid.assign(N, vector<char>(N));
    int minRow = N, maxRow = 0, minCol = N, maxCol = 0;
    int minQRow = N, maxQRow = 0, minQCol = N, maxQCol = 0;
    vector<int> rowMinCol(N, N), rowMaxCol(N, -1);
    for (int r = 0; r < N; ++r) {
        string row;
        cin >> row;
        for (int c = 0; c < N; ++c) {
            grid[r][c] = row[c];
            if (grid[r][c] == '1') {
                minRow = min(minRow, r);
                maxRow = max(maxRow, r);
                minCol = min(minCol, c);
                maxCol = max(maxCol, c);
            } else if (grid[r][c] == '?') {
                minQRow = min(minQRow, r);
                maxQRow = max(maxQRow, r);
                minQCol = min(minQCol, c);
                maxQCol = max(maxQCol, c);
                rowMinCol[r] = min(rowMinCol[r], c);
                rowMaxCol[r] = max(rowMaxCol[r], c);
            }
        }
    }
    int ans = max(0, (maxRow - minRow + 1)) * max(0, (maxCol - minCol + 1));
    if (K == 0) {
        int rowLength = maxRow - minRow + 1;
        int colLength = maxCol - minCol + 1;
        ans = max(ans, area(minRow, maxRow, minCol, maxCol));
        cout << ans << endl;
        return;
    }
    if (K == 1) {
        for (int r = 0; r < N; ++r) {
            for (int c = 0; c < N; ++c) {
                if (grid[r][c] != '?') continue;
                ans = max(ans, area(min(minRow, r), max(maxRow, r), min(minCol, c), max(maxCol, c)));
            }
        }
        cout << ans << endl;
        return;
    }
    if (K == 2) {
        for (int r1 = 0; r1 < N; ++r1) {
            for (int r2 = 0; r2 < N; ++r2) {
                ans = max(ans, area(min({minRow, r1, r2}), max({maxRow, r1, r2}), min(minCol, rowMinCol[r1]), max(maxCol, rowMaxCol[r2])));
            }
        }
        cout << ans << endl;
        return;
    }
    if (K == 3) {
        for (int r = 0; r < N; ++r) {
            // upper left corner, bottom and right
            ans = max(ans, area(min(minRow, r), max(maxRow, maxQRow), min(minCol, rowMinCol[r]), max(maxCol, maxQCol)));
            // upper right corner, bottom and left
            ans = max(ans, area(min(minRow, r), max(maxRow, maxQRow), min(minCol, minQCol), max(maxCol, rowMaxCol[r])));
            // lower left corner, top and right
            ans = max(ans, area(min(minRow, minQRow), max(maxRow, r), min(minCol, rowMinCol[r]), max(maxCol, maxQCol)));
            // lower right corner, top and left
            ans = max(ans, area(min(minRow, minQRow), max(maxRow, r), min(minCol, minQCol), max(maxCol, rowMaxCol[r])));
        }
        cout << ans << endl;
        return;
    }
    ans = max(ans, area(min(minRow, minQRow), max(maxRow, maxQRow), min(minCol, minQCol), max(maxCol, maxQCol)));
    cout << ans << endl;
}
```

## Problem B: Least Common Ancestor

### Solution 1:  small-to-large tree merging, preorder traversal, stack, and postorder traversal, counts, and sets

1. Goal and inputs
Build a rooted tree from parent indices, compress node names to integers 0..M-1, then compute two values per node A[u] and D[u] and fold them into a single modular hash.
1. Name compression
Collect all names, sort and unique to form pool, then map each names[i] to idx using lower_bound. M is the number of distinct names. This gives stable, lexicographic indices.
1. Path tracking (ancestors)
ancestorCnts stores counts of name indices on the path from the root to the current node’s parent. ancestorData is a set of pairs (count, idx). It is kept consistent by removing the old pair before updating the count. Ordering is by count then idx.
1. Meaning of A[u]
If ancestorData is not empty, A[u] is set to the 1-based index of the smallest pair in ancestorData. That is the name with the minimum frequency along the path to the parent, breaking ties by smaller name index. If empty, A[u] stays 0.
1. Subtree aggregation with small to large
For each node u, dfs merges each child v’s set into u using the small to large trick: always merge the smaller child structure into the larger one. This keeps total complexity near O(N log N).
1. Subtree tracking structures
cnts[u] maps name index to its frequency in u’s processed subtree. treeData[u] is a set of pairs (count, idx) built from cnts[u]. Before increasing a count, the old pair is erased from the set, then the updated pair is inserted. After merging all children and before counting u itself, D[u] is set to the 1-based index at treeData[u].begin() if nonempty. Then u’s own name is added to cnts[u] and treeData[u].
1. Backtracking discipline
For ancestor structures, the code increments the current node’s name before visiting children and restores the path state after finishing u by decrementing and erasing if the count drops to zero. This ensures ancestor data reflects exactly the path to the current DFS node.

```cpp
const int MOD = 998'244'353;
int N, M;
vector<vector<int>> adj;
vector<string> names, pool;
// map the names to index in pool
// map node to name, then map that name to index in pool
unordered_map<string, int> nameToIndex;
vector<unordered_map<int, int>> cnts;
vector<set<pair<int, int>>> treeData;
unordered_map<int, int> ancestorCnts;
set<pair<int, int>> ancestorData;
vector<int> A, D;

void dfs(int u, int p = -1) {
    int idx = nameToIndex[names[u]];
    if (!ancestorData.empty()) {
        auto [_, index] = *ancestorData.begin();
        A[u] = index + 1;
    }
    if (ancestorCnts.count(idx)) {
        int cur = ancestorCnts[idx];
        auto it = ancestorData.find({cur, idx});
        ancestorData.erase(it);
    }
    ancestorData.emplace(++ancestorCnts[idx], idx);
    for (int v : adj[u]) {
        if (v == p) continue;
        dfs(v, u);
        if (treeData[u].size() < treeData[v].size()) {
            swap(treeData[u], treeData[v]);
            swap(cnts[u], cnts[v]);
        }
        // now add all tree data
        for (auto [c, i] : treeData[v]) {
            if (cnts[u].count(i)) {
                int cur = cnts[u][i];
                auto it = treeData[u].find({cur, i});
                treeData[u].erase(it);
            }
            cnts[u][i] += c;
            treeData[u].insert({cnts[u][i], i});
        }
    }
    if (!treeData[u].empty()) {
        auto [_, index] = *treeData[u].begin();
        D[u] = index + 1;
    }
    if (cnts[u].count(idx)) {
        int cur = cnts[u][idx];
        auto it = treeData[u].find({cur, idx});
        treeData[u].erase(it);
    }
    cnts[u][idx] += 1;
    treeData[u].insert({cnts[u][idx], idx});

    if (ancestorCnts.count(idx)) {
        int cur = ancestorCnts[idx];
        auto it = ancestorData.find({cur, idx});
        ancestorData.erase(it);
    }
    if (--ancestorCnts[idx]) {
        ancestorData.emplace(ancestorCnts[idx], idx);
    } else {
        ancestorCnts.erase(idx);
    }
}

void solve() {
    cin >> N;
    adj.assign(N, vector<int>());
    names.assign(N, "");
    pool.clear();
    treeData.assign(N, set<pair<int, int>>());
    cnts.assign(N, unordered_map<int, int>());
    ancestorCnts.clear();
    ancestorData.clear();
    nameToIndex.clear();
    A.assign(N, 0);
    D.assign(N, 0);
    for (int i = 0; i < N; ++i) {
        int p; string s;
        cin >> p >> s;
        names[i] = s;
        pool.emplace_back(s);
        if (p == -1) continue;
        p--;
        adj[p].emplace_back(i);
        adj[i].emplace_back(p);
    }
    sort(pool.begin(), pool.end());
    pool.erase(unique(pool.begin(), pool.end()), pool.end());
    for (int i = 0; i < N; ++i) {
        nameToIndex[names[i]] = lower_bound(pool.begin(), pool.end(), names[i]) - pool.begin();
    }
    M = pool.size();
    dfs(0);
    int64 ans = 0;
    for (int i = 0; i < N; ++i) {
        ans = ans * (M + 1) % MOD;
        ans = (ans + A[i]) % MOD;
        ans = ans * (M + 1) % MOD;
        ans = (ans + D[i]) % MOD;
    }
    cout << ans << endl;
}
```

## Problem E1: All Triplets Shortest Path (Part 1)

### Solution 1:

When the second node is larger than its neighbors.
0 - 2 - 1 - 3
1. That means the second node is larger than the first node.  I cannot rely on it as an intermediate to calculate the distance from the first to the fourth node.  Because it won't be calculated until later than the first node.
dist(0, 3) from dist(2, 3) nope dist(2, 3) calculated at later i = 2 from i = 0
2. I also cannot rely on the third node as the intermediate node, because the second node is larger than it so I will not have calculated the distance to it yet.
dist(0,3) from dist(0, 1), not possible because
that would require dist(0,1) to be calculated going through k = 2, but we haven't reached k = 2, we are trying to solve this with k = 1.

When the third node is larger than its neighbors
0 - 1 - 3 - 2
1. That means the third node is larger larger than the fourth node, so taking it as an intermediate, will it is going to be larger, so it won't be calculated until later.
dist(0, 2) from dist(0, 3)
2. I also cannot rely on the second node as an intermediate, because that require me to have already calculated the distance from the second to the fourth node, but it must go through the third node that is larger.
dist(0, 2) from dist(1, 2)

```cpp
int N;
vector<vector<int>> adj;

void solve() {
    cin >> N;
    adj.assign(N, vector<int>());
    for (int i = 1; i < N; ++i) {
        int u, v;
        cin >> u >> v;
        u--, v--;
        adj[u].emplace_back(v);
        adj[v].emplace_back(u);
    }
    for (int i = 0; i < N; ++i) {
        for (int j : adj[i]) {
            if (j < i) continue;
            for (int k : adj[j]) {
                if (k == i) continue;
                if (k < j && adj[k].size() > 1) {
                    cout << "Wrong" << endl;
                    return;
                }
            }
        }
    }
    cout << "Lucky" << endl;
}
```

What is the intuition behind, this how might I reason to this solution?
