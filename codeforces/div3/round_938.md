# Codeforces Round 938 Div 3

## G. GCD on a grid

### Solution 1:  gcd, factors, highly composite number
highly composite number is 240

```py
int R, C;
vector<vector<int>> grid, dp;

void solve() {
    cin >> R >> C;
    grid.assign(R, vector<int>(C));
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            cin >> grid[i][j];
        }
    }
    int g = __gcd(grid[0][0], grid[R - 1][C - 1]);
    vector<int> factors;
    for (int i = 1; i * i <= g; i++) {
        if (g % i == 0) {
            factors.push_back(i);
            if (i != g / i) factors.push_back(g / i);
        }
    }
    sort(factors.begin(), factors.end(), greater<int>());
    for (int f : factors) {
        dp.assign(R, vector<int>(C, 0));
        dp[0][0] = 1;
        for (int i = 0; i < R; i++) {
            for (int j = 0; j < C; j++) {
                if (i > 0) dp[i][j] |= dp[i - 1][j] & (grid[i][j] % f == 0);
                if (j > 0) dp[i][j] |= dp[i][j - 1] & (grid[i][j] % f == 0);
            }
        }
        if (dp[R - 1][C - 1]) {
            cout << f << endl;
            return;
        }
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

## H. The Most Reckless Defense

### Solution 1: dp bitmask, prefix sum

```py
import math
from itertools import product

def squared_radius(r1, c1, r2, c2):
    return (r1 - r2) ** 2 + (c1 - c2) ** 2

BITS = 11
PATH = "#"
def main():
    R, C, K = map(int, input().split())
    grid = [list(input()) for _ in range(R)]
    towers = [None] * K
    # ceiling radius
    UB = BITS * BITS
    crad = [0] * UB
    crad[1] = 1
    for i in range(2, UB):
        crad[i] = crad[i - 1]
        if crad[i] * crad[i] < i: crad[i] += 1
    for i in range(K):
        r, c, p = map(int, input().split())
        r -= 1; c -= 1
        towers[i] = (r, c, p)
    freq = [[0] * BITS for _ in range(K)]
    for r, c in product(range(R), range(C)):
        if grid[r][c] == PATH:
            for i in range(K):
                r2, c2, p = towers[i]
                dist_squared = squared_radius(r, c, r2, c2)
                if dist_squared >= BITS * BITS: continue
                cdist = crad[dist_squared]
                if cdist < BITS: freq[i][cdist] += 1
    psum = [[0] * BITS for _ in range(K)]
    for i in range(K):
        for j in range(1, BITS):
            psum[i][j] = psum[i][j - 1] + freq[i][j]
    dp = [-math.inf] * (1 << BITS)
    dp[0] = 0
    # O(NM*2^BITS*BITS)
    for i in range(K):
        ndp = [-math.inf] * (1 << BITS)
        p = towers[i][2]
        for mask in range(1 << BITS):
            ndp[mask] = max(ndp[mask], dp[mask])
            for j in range(BITS):
                if (mask >> j) & 1: continue 
                nmask = mask | (1 << j)
                ndp[nmask] = max(ndp[nmask], dp[mask] + psum[i][j] * p - pow(3, j))
        dp = ndp
    print(max(dp))

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```
