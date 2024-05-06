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

## Turning Red

### Solution 1: 

```py

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

## 

### Solution 1: 

```py

```

## 

### Solution 1: 

```py

```