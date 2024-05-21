# Codeforces Round 946 Div 3

## D. Ingenuity-2

### Solution 1:  greedy, constructive, two pointers

```py
ROVER = "R"
HELI = "H"
 
def main():
    n = int(input())
    instr = input()
    norths = [i for i in range(n) if instr[i] == "N"]
    souths = [i for i in range(n) if instr[i] == "S"]
    ans = [None] * n
    pos1 = pos2 = 0
    while norths:
        if pos1 <= pos2:
            ans[norths.pop()] = ROVER
            pos1 += 1
        else:
            ans[norths.pop()] = HELI
            pos2 += 1
    while souths:
        if pos1 > pos2:
            ans[souths.pop()] = ROVER
            pos1 -= 1
        else:
            ans[souths.pop()] = HELI
            pos2 -= 1
    if pos1 != pos2: return print("NO")
    easts = [i for i in range(n) if instr[i] == "E"]
    wests = [i for i in range(n) if instr[i] == "W"]
    pos1 = pos2 = 0
    while easts:
        if pos1 <= pos2:
            ans[easts.pop()] = HELI
            pos1 += 1
        else:
            ans[easts.pop()] = ROVER
            pos2 += 1
    while wests:
        if pos1 > pos2:
            ans[wests.pop()] = HELI
            pos1 -= 1
        else:
            ans[wests.pop()] = ROVER
            pos2 -= 1
    if pos1 != pos2: return print("NO")
    ans = "".join(ans)
    if ans.count(ROVER) == 0 or ans.count(HELI) == 0: return print("NO")
    print(ans)  
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## E. Money Buys Happiness

### Solution 1:  dynamic programming, analyze constraints, minimize cost for each happiness level, prefix sum

```cpp
const int INF = 1e18;

void solve() {
    int m, x;
    cin >> m >> x;
    vector<int> costs(m);
    vector<int> happy(m);
    int H = 0; 
    for (int i = 0; i < m; ++i) {
        cin >> costs[i] >> happy[i];
        H += happy[i];
    }
    vector<int> dp(H + 1, INF), ndp(H + 1, INF);
    dp[0] = 0; // (happiness) -> cheapest cost)
    int psum = 0;
    for (int i = 0; i < m; ++i) {
        ndp.assign(H + 1, INF);
        for (int j = 0; j < H; j++) {
            if (dp[j] + costs[i] <= psum) {
                ndp[j + happy[i]] = min(ndp[j + happy[i]], dp[j] + costs[i]);
            }
            ndp[j] = min(ndp[j], dp[j]);
        }
        swap(dp, ndp);
        psum += x;
    }
    int ans = 0;
    for (int i = 0; i <= H; i++) {
        if (dp[i] < INF) {
            ans = max(ans, i);
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

## F. Cutting Game

### Solution 1:  treat x and y coordinates separately, sorting, marking array, two pointers

```cpp
int A, B, N, M, alice, bob;
vector<pair<int, int>> row, col;
vector<bool> marked;
char cut;

void solve() {
    cin >> A >> B >> N >> M;
    row.resize(N);
    col.resize(N);
    marked.assign(N, false);
    for (int i = 0; i < N; i++) {
        int x, y;
        cin >> x >> y;
        row[i] = {x, i};
        col[i] = {y, i};
    }
    sort(row.begin(), row.end());
    sort(col.begin(), col.end());
    alice = 0;
    bob = 0;
    // two pointers for row and column independently
    int row_l = 0, row_r = N - 1, col_l = 0, col_r = N - 1;
    // dynamic grid, boundaries of current grid
    int row_top = 1, row_bot = A, col_left = 1, col_right = B;
    for (int i = 0; i < M; i++) {
        int c;
        cin >> cut >> c;
        if (cut == 'D') {
            row_bot -= c;
            while (row_r >= 0 && row[row_r].first > row_bot) {
                if (!marked[row[row_r].second]) {
                    marked[row[row_r].second] = true;
                    if (i & 1) bob++;
                    else alice++;
                }
                row_r--;
            }
        } else if (cut == 'U') {
            row_top += c;
            while (row_l < N && row[row_l].first < row_top) {
                if (!marked[row[row_l].second]) {
                    marked[row[row_l].second] = true;
                    if (i & 1) bob++;
                    else alice++;
                }
                row_l++;
            }
        } else if (cut == 'R') {
            col_right -= c;
            while (col_r >= 0 && col[col_r].first > col_right) {
                if (!marked[col[col_r].second]) {
                    marked[col[col_r].second] = true;
                    if (i & 1) bob++;
                    else alice++;
                }
                col_r--;
            }
        } else if (cut == 'L') {
            col_left += c;
            while (col_l < N && col[col_l].first < col_left) {
                if (!marked[col[col_l].second]) {
                    marked[col[col_l].second] = true;
                    if (i & 1) bob++;
                    else alice++;
                }
                col_l++;
            }
        }
    }
    cout << alice << " " << bob << endl;
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

## Money Buys Less Happiness Now

### Solution 1:  sort, greedy, lazy segment tree

```py
def segfunc(x,y):
    return min(x, y)
class LazySegTree_RAQ:
    def __init__(self,init_val,segfunc,ide_ele):
        n = len(init_val)
        self.segfunc = segfunc
        self.ide_ele = ide_ele
        self.num = 1<<(n-1).bit_length()
        self.tree = [ide_ele]*2*self.num
        self.lazy = [0]*2*self.num
        for i in range(n):
            self.tree[self.num+i] = init_val[i]
        for i in range(self.num-1,0,-1):
            self.tree[i] = self.segfunc(self.tree[2*i], self.tree[2*i+1])
    def gindex(self,l,r):
        l += self.num
        r += self.num
        lm = l>>(l&-l).bit_length()
        rm = r>>(r&-r).bit_length()
        while r>l:
            if l<=lm:
                yield l
            if r<=rm:
                yield r
            r >>= 1
            l >>= 1
        while l:
            yield l
            l >>= 1
    def propagates(self,*ids):
        for i in reversed(ids):
            v = self.lazy[i]
            if v==0:
                continue
            self.lazy[i] = 0
            self.lazy[2*i] += v
            self.lazy[2*i+1] += v
            self.tree[2*i] += v
            self.tree[2*i+1] += v
    def add(self,l,r,x):
        ids = self.gindex(l,r)
        l += self.num
        r += self.num
        while l<r:
            if l&1:
                self.lazy[l] += x
                self.tree[l] += x
                l += 1
            if r&1:
                self.lazy[r-1] += x
                self.tree[r-1] += x
            r >>= 1
            l >>= 1
        for i in ids:
            self.tree[i] = self.segfunc(self.tree[2*i], self.tree[2*i+1]) + self.lazy[i]
    def query(self,l,r):
        self.propagates(*self.gindex(l,r))
        res = self.ide_ele
        l += self.num
        r += self.num
        while l<r:
            if l&1:
                res = self.segfunc(res,self.tree[l])
                l += 1
            if r&1:
                res = self.segfunc(res,self.tree[r-1])
            l >>= 1
            r >>= 1
        return res
    def __repr__(self):
        return f"tree: {self.tree}"
 
from itertools import accumulate
import math
def main():
    m, x = map(int, input().split())
    arr = list(map(int, input().split()))
    psum = [0] + list(accumulate([x] * (m - 1)))
    queries = sorted([(num, i) for i, num in enumerate(arr)], key = lambda x: (x[0], -x[1]))
    seg = LazySegTree_RAQ(psum, segfunc, math.inf)
    ans = 0
    for num, i in queries:
        val = seg.query(i, m)
        if num > val: continue
        seg.add(i, m, -num)
        ans += 1
    print(ans)
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```