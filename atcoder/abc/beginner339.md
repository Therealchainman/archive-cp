# Atcoder Beginner Contest 339

## C - Perfect Bus

### Solution 1:  min, sum, find the lowest point under 0

```py
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    mn = cur = 0
    for x in arr:
        cur += x
        mn = min(mn, cur)
    cur = abs(mn)
    for x in arr:
        cur += x
    print(cur)

if __name__ == '__main__':
    main()
```

## D - Synchronized Players

### Solution 1:  bfs

```py
from itertools import product
from collections import deque
import math

def main():
    n = int(input())
    grid = [input() for _ in range(n)]
    pos = []
    for r, c in product(range(n), repeat = 2):
        if grid[r][c] == "P": pos.extend((r, c))
    dq = deque([tuple(pos)])
    dist = [[[[math.inf] * n for _ in range(n)] for _ in range(n)] for _ in range(n)]
    dist[pos[0]][pos[1]][pos[2]][pos[3]] = 0
    in_bounds = lambda r, c: 0 <= r < n and 0 <= c < n
    while dq:
        r1, c1, r2, c2 = dq.popleft()
        if (r1, c1) == (r2, c2): return print(dist[r1][c1][r2][c2])
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr1, nc1, nr2, nc2 = r1 + dr, c1 + dc, r2 + dr, c2 + dc
            if not in_bounds(nr1, nc1) or grid[nr1][nc1] == "#": nr1, nc1 = r1, c1
            if not in_bounds(nr2, nc2) or grid[nr2][nc2] == "#": nr2, nc2 = r2, c2
            if dist[nr1][nc1][nr2][nc2] < math.inf: continue
            dist[nr1][nc1][nr2][nc2] = dist[r1][c1][r2][c2] + 1
            dq.append((nr1, nc1, nr2, nc2))
    print(-1)
if __name__ == '__main__':
    main()
```

## E - Smooth Subsequence

### Solution 1:  segment tree to get range max queries, point updates

```cpp
const int neutral = 0;

struct SegmentTree {
    int size;
    vector<int> nodes;

    void init(int num_nodes) {
        size = 1;
        while (size < num_nodes) size *= 2;
        nodes.resize(size * 2, neutral);
    }

    int func(int x, int y) {
        return max(x, y);
    }

    void ascend(int segment_idx) {
        while (segment_idx > 0) {
            int left_segment_idx = 2 * segment_idx, right_segment_idx = 2 * segment_idx + 1;
            nodes[segment_idx] = func(nodes[left_segment_idx], nodes[right_segment_idx]);
            segment_idx >>= 1;
        }
    }

    void update(int segment_idx, int val) {
        segment_idx += size;
        nodes[segment_idx] = val;
        segment_idx >>= 1;
        ascend(segment_idx);
    }

    int query(int left, int right) {
        left += size, right += size;
        int res = neutral;
        while (left <= right) {
            if (left & 1) {
                res = max(res, nodes[left]);
                left++;
            }
            if (~right & 1) {
                res = max(res, nodes[right]);
                right--;
            }
            left >>= 1, right >>= 1;
        }
        return res;
    }
};

const int MAXN = 5 * 1e5 + 5;
int N, D, dp[MAXN];
vector<int> arr;

signed main() {
    cin >> N >> D;
    arr.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> arr[i];
    }
    SegmentTree seg;
    seg.init(MAXN);
    memset(dp, 0, sizeof(dp));
    int ans = 0;
    for (int i = 0; i < N; i++) {
        int l = max(0LL, arr[i] - D), r = min(MAXN - 1, arr[i] + D);
        int mx = seg.query(l, r);
        ans = max(ans, mx + 1);
        seg.update(arr[i], mx + 1);
    }
    cout << ans << endl;
    return 0;
}
```

## F - Product Equality

### Solution 1:  multiplication of large numbers, modular arithmetic, hash table

I think this solution is wrong some reason, just read through and you will remember.

```py
import random
P = 20
MODS = [random.randint(10**9, 2 * 10**9) for _ in range(P)]

def main():
    n = int(input())
    arr = [[0] * P for _ in range(n)]
    counter = [Counter() for _ in range(P)]
    for i in range(n):
        a = int(input())
        for j in range(P):
            ma = a % MODS[j]
            arr[i][j] = ma
            counter[j][ma] += 1
    ans = 0
    for i in range(n):
        for j in range(i, n):
            val = arr[i][0] * arr[j][0]
            cand = counter[0][val % MODS[0]]
            if all(counter[k][(arr[i][k] * arr[j][k]) % MODS[k]] == cand for k in range(1, P)):
                ans += cand
                if i != j: ans += cand
    print(ans)

if __name__ == '__main__':
    main()
```

## G - Smaller Sum

### Solution 1:  merge sort tree, online queries, cumulative sum of all elements less than or equal to X,

```cpp
const int N = 2e5 + 10;
vector<int> tree[4 * N], psum[4 * N];
int n, arr[N], a, b, c;

struct MergeSortTree {
    void build(int u, int left, int right) {
        if (left == right) {
            tree[u].push_back(arr[left]);
            psum[u].push_back(arr[left]);
            return;
        }
        int mid = (left + right) >> 1;
        int left_segment = u << 1;
        int right_segment = left_segment | 1;
        build(left_segment, left, mid);
        build(right_segment, mid + 1, right);
        merge(tree[left_segment].begin(), tree[left_segment].end(), tree[right_segment].begin(), tree[right_segment].end(), back_inserter(tree[u]));
        int l = 0, r = 0, nl = tree[left_segment].size(), nr = tree[right_segment].size(), cur = 0;
        while (l < nl or r < nr) {
            if (l < nl and r < nr) {
                if (tree[left_segment][l] <= tree[right_segment][r]) {
                    cur += tree[left_segment][l];
                    l += 1;
                } else {
                    cur += tree[right_segment][r];
                    r += 1;
                }
            } else if (l < nl) {
                cur += tree[left_segment][l];
                l += 1;
            } else {
                cur += tree[right_segment][r];
                r += 1;
            }
            psum[u].push_back(cur);
        }
    }
    // not greater than k, so <= k we want
    int query(int u, int left, int right, int i, int j, int k) {
        if (i > right || left > j) return 0; // NO OVERLAP
        if (i <= left && right <= j) { // COMPLETE OVERLAP
            int idx = upper_bound(tree[u].begin(), tree[u].end(), k) - tree[u].begin();
            return idx > 0 ? psum[u][idx - 1] : 0;
        }
        // PARTIAL OVERLAP
        int mid = (left + right) >> 1;
        int left_segment = u << 1;
        int right_segment = left_segment | 1;
        return query(left_segment, left, mid, i, j, k) + query(right_segment, mid + 1, right, i, j, k);
    }
};

signed main() {
    cin >> n;
    for (int i = 0; i < n; i++) {
        cin >> arr[i];
    }
    MergeSortTree mst;
    mst.build(1, 0, n - 1);
    int q, ans = 0;
    cin >> q;
    while (q--) {
        cin >> a >> b >> c;
        int L = a ^ ans, R = b ^ ans, K = c ^ ans;
        ans = mst.query(1, 0, n - 1, L - 1, R - 1, K);
        cout << ans << endl;
    }
    return 0;
}
```

