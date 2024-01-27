# Codeforces Educational 161 Div 2

## B. Forming Triangles

### Solution 1:  triangle inequality, math, combinatorics

```py
def main():
    n = int(input())
    cnt = [0] * (n + 1)
    for num in map(int, input().split()):
        cnt[num] += 1
    ans = psum = 0
    for c in map(lambda x: cnt[x], range(n + 1)):
        ans += c * (c - 1) * (c - 2) // 6 # pick 3 from count of items
        ans += c * (c - 1) // 2 * psum  # pick 2 from count of items and 1 from others
        psum += c
    print(ans)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## C. Closest Cities

### Solution 1:  prefix sum

```py
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    m = int(input())
    psum = [0] * n
    ssum = [0] * n
    psum[1] = ssum[-2] = 1
    for i in range(1, n - 1):
        psum[i + 1] = psum[i] + (1 if abs(arr[i + 1] - arr[i]) < abs(arr[i] - arr[i - 1]) else abs(arr[i] - arr[i + 1]))
    for i in range(n - 2, 0, -1):
        ssum[i - 1] = ssum[i] + (1 if abs(arr[i - 1] - arr[i]) < abs(arr[i] - arr[i + 1]) else abs(arr[i] - arr[i - 1]))
    for _ in range(m):
        x, y = map(int, input().split())
        x -= 1
        y -= 1
        ans = psum[y] - psum[x] if y > x else ssum[y] - ssum[x]
        print(ans)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## D. Berserk Monsters

### Solution 1:  doubly linked list, set, prv, nxt arrays

```py
def main():
    n = int(input())
    attack = [0] + list(map(int, input().split()))
    defense = [0] + list(map(int, input().split()))
    prv, nxt = [0] * (n + 2), [0] * (n + 2)
    for i in range(1, n + 1):
        prv[i] = i - 1 if i > 0 else 0
        nxt[i] = i + 1 if i < n else n + 1
    ans = [0] * n
    in_bounds = lambda idx: 1 <= idx <= n
    def kill(idx):
        dmg = 0
        if in_bounds(prv[idx]): dmg += attack[prv[idx]]
        if in_bounds(nxt[idx]): dmg += attack[nxt[idx]]
        return dmg > defense[idx]
    alive = [1] * (n + 1)
    marked = set(range(1, n + 1))
    def populate():
        dead = []
        for i in marked:
            if kill(i):
                dead.append(i)
                alive[i] = 0
        return dead
    dead = populate()
    for r in range(n):
        marked.clear()
        for i in dead:
            prv[nxt[i]] = prv[i]
            nxt[prv[i]] = nxt[i]
            if in_bounds(prv[i]) and alive[prv[i]]:
                marked.add(prv[i])
            if in_bounds(nxt[i]) and alive[nxt[i]]:
                marked.add(nxt[i])
            ans[r] += 1
        dead = populate()
    print(*ans)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## E. Increasing Subsequences

### Solution 1:  bitmasks, bit manipulation

```py
def main():
    X = bin(int(input()))[2:]
    ans = []
    for i in range(1, len(X)):
        ans.append(i)
        if X[i] == "1":
            ans.append(0)
    print(len(ans))
    print(*ans)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## F. Replace on Segment

### Solution 1: interval dynamic programming, cyclic dependency handling, modified intervals, recursive

```cpp
const int N = 105;
int n, x;
int nxt1[N][N], nxt2[N][N], prv1[N][N], prv2[N][N], dp1[N][N][N], dp2[N][N][N];
vector<int> arr;

int remove(int left, int right, int k);

int add(int left, int right, int k) {
    left = nxt1[left][k];
    right = prv1[right][k];
    if (left > right) return 0;
    if (dp1[left][right][k] != -1) return dp1[left][right][k];
    int res = LONG_LONG_MAX;
    for (int i = left; i < right; i++) {
        res = min(res, add(left, i, k) + add(i + 1, right, k));
    }
    res = min(res, remove(left, right, k) + 1);
    return dp1[left][right][k] = res;
}

int remove(int left, int right, int k) {
    left = nxt2[left][k];
    right = prv2[right][k];
    if (left > right) return 0;
    if (dp2[left][right][k] != -1) return dp2[left][right][k];
    int res = LONG_LONG_MAX;
    for (int i = left; i < right; i++) {
        res = min(res, remove(left, i, k) + remove(i + 1, right, k));
    }
    for (int m = 1; m <= x; m++) {
        if (m == k) continue;
        res = min(res, add(left, right, m));
    }
    return dp2[left][right][k] = res;
}

void solve() {
    cin >> n >> x;
    arr.resize(n);
    for (int i = 0; i < n; i++) {
        cin >> arr[i];
    }
    memset(dp1, -1, sizeof(dp1));
    memset(dp2, -1, sizeof(dp2));
    for (int k = 1; k <= x; k++) {
        int last1 = n, last2 = n;
        for (int i = n - 1; i >= 0; i--) {
            if (arr[i] != k) last1 = i;
            else last2 = i;
            nxt1[i][k] = last1;
            nxt2[i][k] = last2;
        }
        int first1 = -1, first2 = -1;
        for (int i = 0; i < n; i++) {
            if (arr[i] != k) first1 = i;
            else first2 = i;
            prv1[i][k] = first1;
            prv2[i][k] = first2;
        }
    }
    int ans = LONG_LONG_MAX;
    for (int k = 1; k <= x; k++) {
        ans = min(ans, add(0, n - 1, k));
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

### Solution 2:  iterative dp

```cpp
const int N = 105;
int n, x;
int arr[N], dp1[N][N][N], dp2[N][N][N];
/*
dp(left, right, k)
dp1 is add
dp2 is remove
*/

void solve() {
    cin >> n >> x;
    for (int i = 0; i < n; i++) {
        cin >> arr[i];
    }
    memset(dp1, N, sizeof(dp1));
    memset(dp2, N, sizeof(dp2));
    for (int i = 0; i < n; i++) {
        for (int j = 1; j <= x; j++) {
            dp1[i][i][j] = arr[i] == j ? 0 : 1;
            dp2[i][i][j] = arr[i] == j ? 1 : 0;
        }
    }
    for (int len = 2; len <= n; len++) {
        for (int left = 0; left + len - 1 < n; left++) {
            int right = left + len - 1;
            vector<pair<int, int>> vec;
            for (int k = 1; k <= x; k++) {
                for (int i = left; i < right; i++) {
                    dp1[left][right][k] = min(dp1[left][right][k], dp1[left][i][k] + dp1[i + 1][right][k]);
                    dp2[left][right][k] = min(dp2[left][right][k], dp2[left][i][k] + dp2[i + 1][right][k]);
                    /* next part is the following, need the transformations*/
                }
                /* transformation from all being not equal to k to being all equal to k */
                dp1[left][right][k] = min(dp1[left][right][k], dp2[left][right][k] + 1);

                vec.emplace_back(dp1[left][right][k], k);
            }
            for (int k = 1; k <= x; k++) {
                for (auto &[cnt, m]: vec) {
                    if (m == k) continue;
                    dp2[left][right][k] = min(dp2[left][right][k], dp1[left][right][m]);
                }
            }
        }
    }
    int ans = N;
    for (int k = 1; k <= x; k++) {
        ans = min(ans, dp1[0][n - 1][k]);
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

```py
sys.setrecursionlimit(10**6)
import math
UNVISITED = -1

def main():
    n, x = map(int, input().split())
    arr = list(map(int, input().split()))
    dp1 = [[[UNVISITED] * (x + 1) for _ in range(n)] for _ in range(n)]
    dp2 = [[[UNVISITED] * (x + 1) for _ in range(n)] for _ in range(n)]
    nxt1, prv1 = [[0] * (x + 1) for _ in range(n)], [[0] * (x + 1) for _ in range(n)]
    nxt2, prv2 = [[0] * (x + 1) for _ in range(n)], [[0] * (x + 1) for _ in range(n)]
    for k in range(1, x + 1):
        last1 = last2 = n
        for i in reversed(range(n)):
            if arr[i] != k: last1 = i
            else: last2 = i
            nxt1[i][k] = last1
            nxt2[i][k] = last2
        first1 = first2 = -1
        for i in range(n):
            if arr[i] != k: first1 = i
            else: first2 = i
            prv1[i][k] = first1
            prv2[i][k] = first2
    # add all k
    def add(left, right, k):
        left = nxt1[left][k]
        right = prv1[right][k]
        if left > right: return 0
        if dp1[left][right][k] != UNVISITED: return dp1[left][right][k]
        res = math.inf
        # split
        for i in range(left, right):
            res = min(res, add(left, i, k) + add(i + 1, right, k))
        # transformation
        res = min(res, remove(left, right, k) + 1)
        dp1[left][right][k] = res
        return res
    # remove all k
    def remove(left, right, k):
        left = nxt2[left][k]
        right = prv2[right][k]
        if left > right: return 0
        if dp2[left][right][k] != UNVISITED: return dp2[left][right][k]
        res = math.inf
        # split
        for i in range(left, right):
            res = min(res, remove(left, i, k) + remove(i + 1, right, k))
        # transformation
        for m in range(1, x + 1):
            if m == k: continue
            res = min(res, add(left, right, m))
        dp2[left][right][k] = res
        return res
    ans = min([add(0, n - 1, k) for k in range(1, x + 1)])
    print(ans)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```