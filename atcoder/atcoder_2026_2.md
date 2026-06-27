# Atcoder 2026 Part 2

# Atcoder Beginner Contest 464

## C - Plumage Palette

### Solution 1: 

```cpp
int N, M;

void solve() {
    cin >> N >> M;
    vector<vector<int>> colorChange(M + 1, vector<int>());
    vector<int> A(N), D(N), B(N), freq(N + 1, 0);
    int ans = 0;
    for (int i = 0; i < N; ++i) {
        cin >> A[i] >> D[i] >> B[i];
        colorChange[D[i]].emplace_back(i);
        freq[A[i]]++;
        if (freq[A[i]] == 1) {
            ans++;
        }
    }
    for (int i = 1; i <= M; ++i) {
        for (int j : colorChange[i]) {
            freq[A[j]]--;
            if (freq[A[j]] == 0) {
                ans--;
            }
            freq[B[j]]++;
            if (freq[B[j]] == 1) {
                ans++;
            }
        }
        cout << ans << endl;
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

## D - Celester

### Solution 1: 

```cpp
string S;
int N;
vector<int64> X, Y;

void solve() {
    cin >> N >> S;
    X.assign(N, 0);
    Y.assign(N - 1, 0);
    for (int i = 0; i < N; ++i) {
        cin >> X[i];
    }
    for (int i = 0; i < N - 1; ++i) {
        cin >> Y[i];
    }
    vector<int64> dp0(N, 0), dp1(N, 0); // dp0 -> 'S', dp1 -> 'R'
    for (int i = 0; i < N; ++i) {
        int64 costS = S[i] != 'S' ? X[i] : 0;
        int64 costR = S[i] != 'R' ? X[i] : 0;
        dp0[i] = -costS;
        dp1[i] = -costR;
        if (i > 0) {
            dp0[i] = max(dp0[i], dp0[i - 1] - costS);
            dp0[i] = max(dp0[i], dp1[i - 1] - costS + Y[i - 1]);
            dp1[i] = max(dp1[i], dp0[i - 1] - costR);
            dp1[i] = max(dp1[i], dp1[i - 1] - costR);
        }
    }
    int64 ans = max(dp0[N - 1], dp1[N - 1]);
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

## E - Fill-Rect Query

### Solution 1: 2D suffix maximum

Queries are stored at their bottom-right corners.

        c →
      0   1   2   3   4
r   +---+---+---+---+---+
↓ 0 |   |   |   |   |   |
    +---+---+---+---+---+
  1 |   | ? | ← | ← | q5|
    +---+---+---+---+---+
  2 |   | ↑ |   |   |   |
    +---+---+---+---+---+
  3 |   | ↑ |   | q8|   |
    +---+---+---+---+---+
  4 |   | ↑ |   |   |q12|
    +---+---+---+---+---+

For cell ?, look at every query endpoint southeast of it.

best[1][1] = max(
    seed[1][1],
    best[2][1],  // row beneath
    best[1][2]   // column to the right
)

```cpp
int R, C, Q;
vector<vector<char>> grid;

void solve() {
    cin >> R >> C >> Q;
    grid.assign(R, vector<char>(C, 'A'));
    vector<vector<int>> best(R + 1, vector<int>(C + 1, -1));
    vector<char> qvalues(Q, 'A');
    for (int i = 0; i < Q; ++i) {
        int r, c;
        char ch;
        cin >> r >> c >> ch;
        --r, --c;
        best[r][c] = i;
        qvalues[i] = ch;
    }
    for (int r = R - 1; r >= 0; --r) {
        for (int c = C - 1; c >= 0; --c) {
            best[r][c] = max(best[r][c], best[r + 1][c]);
            best[r][c] = max(best[r][c], best[r][c + 1]);
        }
    }
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < C; ++c) {
            if (best[r][c] == -1) cout << 'A';
            else cout << qvalues[best[r][c]];
        }
        cout << endl;
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

## F - Random Vault Heist

### Solution 1: Expected value, combinatorics, random permutation trick, meet-in-the-middle subset-sum counting

A stopped random permutation process solved by conditional expectation over prefix states, with meet-in-the-middle to aggregate valid subsets.

The important reusable pattern is:

Random sequential choice from remaining items
    -> random permutation

Expected final value
    -> sum of expected contributions

Stopping condition based on prefix sum
    -> count valid prefix sets

N <= 40 and subset sums involved
    -> meet-in-the-middle


onditional expectation is used when you do not know the exact value yet, but you know some partial information.

Here, the partial information is:

The robber has already stolen set S.

Once you know that, the next amount is still random, because the robber still chooses one of the remaining safes uniformly.

So instead of saying:

probability * exact value

you say:

probability of reaching this situation * average value from this situation

That average value is the conditional expectation.

What conditional expectation normally means

Normal expectation is:

E[Y] = sum over outcomes:
    P(outcome) * value(outcome)

Conditional expectation is:

E[Y | event happened]

which means:

the expected value of Y, assuming we already know some event happened

for each valid stolen-so-far set S:
    add expected value of the next safe

Given prefix set S, what is the expected next contribution?

The correction is that the editorial does not compute contribution per individual safe. It computes contribution per valid prefix set S, using the expected next amount from the remaining safes.

```cpp
const int64 MOD = 998244353;

int64 exponentiation(int64 b, int64 p, int64 m) {
    int64 res = 1;
    while (p > 0) {
        if (p & 1) res = (res * b) % m;
        b = (b * b) % m;
        p >>= 1;
    }
    return res;
}

int64 inv(int64 i, int64 m) {
    return exponentiation(i, m - 2, m);
}

int64 norm_mod(int64 x) {
    x %= MOD;
    if (x < 0) x += MOD;
    return x;
}

// res[k] = all subset sums using exactly k elements from a.
vector<vector<int64>> generate_subset_sums_by_size(const vector<int64>& a) {
    int n = (int)a.size();
    vector<vector<int64>> res(n + 1);
    res[0].push_back(0);
    for (int64 x : a) {
        for (int k = n - 1; k >= 0; --k) {
            int old_size = (int)res[k].size();
            for (int i = 0; i < old_size; ++i) {
                res[k + 1].push_back(res[k][i] + x);
            }
        }
    }
    return res;
}

void solve() {
    int N;
    int64 X;
    cin >> N >> X;

    vector<int64> A(N);
    int64 total_sum = 0;
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
        total_sum += A[i];
    }

    int h = N / 2;
    vector<int64> left(A.begin(), A.begin() + h);
    vector<int64> right(A.begin() + h, A.end());

    auto left_sums = generate_subset_sums_by_size(left);
    auto right_sums = generate_subset_sums_by_size(right);

    // combinations C(N, k) mod MOD
    vector<vector<int64>> comb(N + 1, vector<int64>(N + 1, 0));
    for (int n = 0; n <= N; ++n) {
        comb[n][0] = 1;
        for (int k = 1; k <= n; ++k) {
            comb[n][k] = (comb[n - 1][k - 1] + comb[n - 1][k]) % MOD;
        }
    }

    // q[k] = 1 / ((N - k) * C(N, k)) for valid prefix size k (k < N)
    vector<int64> q(N + 1, 0);
    for (int k = 0; k < N; ++k) {
        int64 denom = (int64)(N - k) % MOD * comb[N][k] % MOD;
        q[k] = inv(denom, MOD);
    }

    int64 total_sum_mod = norm_mod(total_sum);
    int64 ans = 0;

    for (int right_count = 0; right_count < (int)right_sums.size(); ++right_count) {
        auto& rs = right_sums[right_count];
        sort(rs.begin(), rs.end());

        // prefix sums of sorted right subset sums (mod MOD)
        vector<int64> pref(rs.size() + 1, 0);
        for (int i = 0; i < (int)rs.size(); ++i) {
            pref[i + 1] = (pref[i] + norm_mod(rs[i])) % MOD;
        }

        for (int left_count = 0; left_count < (int)left_sums.size(); ++left_count) {
            int total_count = left_count + right_count;
            if (total_count >= N) continue;

            for (int64 left_sum : left_sums[left_count]) {
                int64 limit = X - left_sum;
                // count right sums sr with sl + sr < X, i.e. sr < limit
                int c = lower_bound(rs.begin(), rs.end(), limit) - rs.begin();
                if (c == 0) continue;

                int64 c_mod = c % MOD;
                int64 left_sum_mod = norm_mod(left_sum);
                // sum over valid S of (A_all - sum(S)) = c*(A_all - sl) - sum(sr)
                int64 numerator_sum =
                    (c_mod * norm_mod(total_sum_mod - left_sum_mod) % MOD - pref[c]) % MOD;
                numerator_sum = norm_mod(numerator_sum);

                ans = (ans + q[total_count] * numerator_sum) % MOD;
            }
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

## G - Celester 2

### Solution 1: 

Why? You can move zeros using:

01 <-> 10

Then once two zeros become adjacent, use:

00 -> 11

So each paired zero-pair increases the number of 1s by 2, which increases happiness by 1.

Therefore:

To increase happiness by x, choose x disjoint pairs of zeros with minimum total distance.

In terms of D, choose x elements, but no two chosen elements can be adjacent, because adjacent distances share a zero.

4. Greedy with priority queue and linked list

Naively this looks like DP on a path, but we need answers for all k, and N is up to 1e6.

The editorial uses a known greedy for minimum-cost matching on a line.

Each D[i] is an edge between two neighboring zeros.

Repeatedly:

Pick the smallest alive D[i].
Add its cost to the operation count.
Increase happiness by 1.
Remove D[i-1], D[i], D[i+1].
Insert a replacement value:
D[i-1] + D[i+1] - D[i]

The replacement is the clever part.

If you picked the middle pair first, but later want to instead use the two outer pairs, the new cost lets you “undo” the earlier choice correctly.

Example:

D = [2, 1, 2]

Pick 1.

Cost so far:

1

Replacement:

2 + 2 - 1 = 3

If you later pick replacement 3, total cost becomes:

1 + 3 = 4

That equals choosing the two outer pairs:

2 + 2 = 4

So the greedy can safely make the locally cheapest choice while still preserving future alternatives.

Really you need to solve for the minimum operations to achieve each additional happiness, so +1, +2, +3, etc.

reduction into this problem. 

You are given `M` elements placed on a 1D number line in increasing order.

Instead of giving the coordinates directly, you are given an array `D`, where:

```text
D[i] = distance between element i and element i + 1
```

You may choose disjoint pairs of elements. Each element can be used in at most one pair.

The cost of choosing a pair is the distance between the two elements on the number line.

For each budget `k = 0, 1, ..., N`, determine the maximum number of disjoint pairs that can be chosen with total cost at most `k`.

ALTERNATIVE:

You are given a weighted path graph with `M` vertices. The edge weight `D[i]` is the distance between vertex `i` and vertex `i + 1`.

A valid selection is a matching: a set of edges such that no two selected edges share an endpoint.

For each budget `k`, find the maximum number of edges that can be selected with total weight at most `k`.


```cpp
void solve() {
    const int INF = 100000000;

    int N;
    string S;
    cin >> N >> S;

    // Pad with a leading 'S' and trailing 'R' so boundary pairs are handled.
    string A = "S" + S + "R";

    // Count "different" adjacent pairs (which already contribute), and record
    // positions where adjacent chars are the same.
    int diffCount = 0;
    vector<int> samePositions;
    for (int i = 0; i <= N; i++) {
        if (A[i] == A[i + 1]) {
            samePositions.emplace_back(i);
        } else {
            diffCount++;
        }
    }

    int happiness = diffCount / 2;

    // Build the gap array D between consecutive "same" positions, padded with
    // INF sentinels on both ends so the greedy never pairs across the border.
    vector<int> D;
    D.emplace_back(INF);
    D.emplace_back(INF);
    for (int i = 1; i < samePositions.size(); i++) {
        D.emplace_back(samePositions[i] - samePositions[i - 1]);
    }
    D.emplace_back(INF);
    D.emplace_back(INF);

    int M = D.size();

    // Doubly linked list over edges + min-heap on cost, with lazy deletion.
    vector<int> nxt(M), prv(M);
    vector<bool> alive(M, true);
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> minheap;
    for (int i = 0; i < M; i++) {
        prv[i] = i - 1;
        nxt[i] = i + 1;
        minheap.emplace(D[i], i);
    }

    vector<int> ans(N + 1, 0);
    int usedOps = 0;
    ans[0] = happiness;

    int maxHappiness = N / 2;

    // Greedily realize the cheapest improvement; each picked edge merges with
    // its neighbors (slope-trick style D[idx] = D[l] + D[r] - D[idx]).
    while (happiness < maxHappiness) {
        auto [cost, idx] = minheap.top();
        minheap.pop();

        if (!alive[idx]) continue;

        usedOps += cost;
        happiness++;

        ans[usedOps] = max(ans[usedOps], happiness);

        int l = prv[idx];
        int r = nxt[idx];

        D[idx] = D[l] + D[r] - D[idx];
        minheap.emplace(D[idx], idx);

        // Erase both neighbors l and r from the doubly linked list.
        alive[l] = false;
        prv[nxt[l]] = prv[l];
        nxt[prv[l]] = nxt[l];

        alive[r] = false;
        prv[nxt[r]] = prv[r];
        nxt[prv[r]] = nxt[r];
    }

    // Take prefix maximum so ans[k] is the best achievable using at most k ops.
    for (int k = 1; k <= N; k++) {
        ans[k] = max(ans[k], ans[k - 1]);
    }

    for (int k = 0; k <= N; k++) {
        cout << ans[k] << ' ';
    }
    cout << endl;
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
