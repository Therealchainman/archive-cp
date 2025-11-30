# WWPIT

# WWPIT Spring 2025 Programming Contest

## Cycles

### Solution 1: algebra, math, prime sieve, divisors, prefix and suffix counts

```cpp
int N;
vector<int> A;
const int MAXN = 1e5 + 5;
int spf[MAXN];

// nloglog(n)
void sieve(int n) {
    for (int i = 0; i < n; i++) {
        spf[i] = i;
    }
    for (int64 i = 2; i < n; i++) {
        if (spf[i] != i) continue;
        for (int64 j = i * i; j < n; j += i) {
            if (spf[j] != j) continue;
            spf[j] = i;
        }
    }
}

vector<int> divisors(int x) {
    vector<pair<int,int>> pf;
    while (x > 1) {
        int p = spf[x], cnt = 0;
        while (x % p == 0) { x /= p; ++cnt; }
        pf.emplace_back(p, cnt); // prime, multiplicity
    }
    vector<int> divs{1};
    for (auto [p, c] : pf) {
        int sz = (int)divs.size();
        int mult = 1;
        for (int e = 1; e <= c; ++e) {
            mult *= p;
            for (int i = 0; i < sz; ++i)
                divs.emplace_back(divs[i] * mult);
        }
    }
    return divs;
}

void solve() {
    cin >> N;
    A.assign(N, 0);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    int64 ans = 0;
    map<int, int> pcount, scount;
    for (int i = 0; i < N; i++) {
        scount[A[i]]++;
    }
    for (int i = 0; i < N; i++) {
        int aj = A[i];
        scount[aj]--;
        for (int ak : divisors(aj)) {
            int ai = aj / ak + 1;
            ans += pcount[ai] * scount[ak];
        }
        pcount[aj]++;
    }
    cout << ans << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    sieve(MAXN);
    int T;
    cin >> T;
    while (T--) {
        solve();
    }
    return 0;
}

```

## Levels and Coins

### Solution 1: 

```cpp
```

## Storage Sync

### Solution 1: 

```cpp
```

## Draining Lake

### Solution 1: greedy, splitting in half, manhattan distance, diamond

```cpp

```

### Analysis

With these visualizations an approach begins to make sense. The idea is to use the Manhattan distance to eliminate candidates. The first phase is to make random guesses until we have a number of candidates that is less than or equal to the number of perfect-split queries we can make. Then we can use perfect-split queries to find the drain. The perfect-split queries is that there are certain guesses you can make that will split the remaining candidates into two groups of equal size. The idea is to find the guess that minimizes the maximum size of the two groups. This is a greedy approach, but it works well in practice.

```py
import matplotlib.pyplot as plt
import random
from math import ceil, log2
import numpy as np

def simulate(N, true_pos):
    def manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    candidates = [(i, j) for i in range(N) for j in range(N)]
    used = set()
    query_no = 0

    # Number of perfect-split queries needed
    Qsplit = min(2 * ceil(log2(N)), N)

    def do_query(guess):
        nonlocal query_no, candidates
        i = query_no + 1
        dist_true = manhattan(true_pos, guess)
        resp = 0 if dist_true <= i else 1

        # Filter candidates based on response
        new_cands = []
        for d in candidates:
            d0 = manhattan(d, guess)
            if (resp == 0 and d0 <= i) or (resp == 1 and d0 > i):
                new_cands.append(d)
        candidates = new_cands
        query_no += 1

    def visualize(guess):
        xs = [c[1] for c in candidates]
        ys = [c[0] for c in candidates]
        plt.figure()
        plt.scatter(xs, ys)
        gx, gy = guess[1], guess[0]
        plt.scatter([gx], [gy], marker='x', s=100)
        plt.title(f"Query {query_no + 1}: guess={guess}, remaining={len(candidates)}")
        plt.gca().invert_yaxis()
        plt.xticks(range(N))
        plt.yticks(range(N))
        plt.grid(True)
        plt.show()

    def visualize_diamond(N, guess, t):
        """
        Visualize the diamond (Manhattan radius) of size t around cell `guess` in an N x N grid.
        
        Parameters:
        - N: int, size of the grid (N x N)
        - guess: tuple (row, col), 0-indexed
        - t: int, Manhattan radius
        """
        grid = np.zeros((N, N))
        x0, y0 = guess

        # Compute Manhattan distance from guess to every cell
        for i in range(N):
            for j in range(N):
                if abs(i - x0) + abs(j - y0) <= t:
                    grid[i, j] = 1  # inside diamond

        plt.figure(figsize=(6,6))
        plt.imshow(grid, origin='lower', extent=[0, N, 0, N])
        plt.scatter([y0 + 0.5], [x0 + 0.5], c='red', marker='x', s=100, label='Center')
        plt.title(f"Diamond region of radius {t} around ({x0},{y0})")
        plt.xticks(np.arange(N+1))
        plt.yticks(np.arange(N+1))
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.gca().invert_yaxis()
        plt.show()

    # # Phase 1: random guesses for first (N - Qsplit) turns
    # for _ in range(N - Qsplit):
    #     unused = [(i, j) for i in range(N) for j in range(N) if (i, j) not in used]
    #     guess = random.choice(unused)
    #     visualize(guess)
    #     do_query(guess)
    #     used.add(guess)
    #     if len(candidates) <= 1:
    #         break

    # Phase 2: perfect-split queries
    while query_no < N and len(candidates) > 1:
        best_guess = None
        best_max = float('inf')
        # Find the guess that most evenly splits the remaining candidates
        for x in range(N):
            for y in range(N):
                if (x, y) in used:
                    continue
                cnt0 = sum(1 for d in candidates if manhattan(d, (x, y)) <= query_no + 1)
                cnt1 = len(candidates) - cnt0
                mx = max(cnt0, cnt1)
                if mx < best_max:
                    best_max = mx
                    best_guess = (x, y)
                    if mx * 2 <= len(candidates):
                        break
            if best_guess and best_max * 2 <= len(candidates):
                break
        visualize(best_guess)
        visualize_diamond(N, best_guess, query_no + 1)
        do_query(best_guess)
        used.add(best_guess)

    visualize(candidates[0])

    # Final result
    if len(candidates) == 1:
        print("Found drain at", candidates[0])
    else:
        print("Remaining candidates:", candidates)

if __name__ == "__main__":
    N = int(input("Enter grid size N: "))
    x = int(input("Enter true drain row x (0-indexed): "))
    y = int(input("Enter true drain col y (0-indexed): "))
    simulate(N, (x, y))

```