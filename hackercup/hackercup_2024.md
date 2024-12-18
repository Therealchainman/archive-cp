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

## Problem C: Bunny Hopscotch

### Solution 1: 

Suppose I want to perform a range query to count all the integers in a range that are equal to x?  

given a grid, and a specific cell (r, c) in the grid.  I am calculating the distance from this cell to other cells to be the max(|r1 - r2|, |c1 - c2|), how can I quickly count for each distance the number of other cells that are that distance away, such as distance = 1, there might be at most 8 cells, for distance = 2, 16 cells, distance = 3, 24 cells.  Basically the maximum number of other cells within the boundary of the grid is going to be 4 * 2 * distance.  But some of the cells may be outside of the grid, how do I calculate with that consideration? 

I think I can count how many cells there are at a distance 1 <= d <= 800 in about O(N^2) time complexity.  But this is not aware of cells that have the same owner.  How could I possibly calculate that quickly.  I'd have to know how many cells have the same owner at each distance.  

I guess you can do this if you know which cells have same values.  

I just don't know how to query what cells have same value as the current one fast enough.  So yeah I give up.  

```cpp

```
